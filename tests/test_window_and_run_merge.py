import ast
import re
import types
import unittest
from pathlib import Path
from typing import Optional

from lxml import etree


def _load_functions(path: Path, names: list[str], globals_dict: dict):
    source = path.read_text(encoding="utf-8")
    module_ast = ast.parse(source, filename=str(path))
    selected = [
        node for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    test_module = ast.Module(body=selected, type_ignores=[])
    namespace = dict(globals_dict)
    exec(compile(test_module, filename=str(path), mode="exec"), namespace)
    return {name: namespace[name] for name in names}


class WindowSplitRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import difflib
        root = Path(__file__).resolve().parents[1]
        funcs = _load_functions(
            root / "infer" / "__init__.py",
            [
                "_split_window_output_parts",
                "_recover_window_parts",
                "_align_parts_to_sources",
                "_line_similarity",
            ],
            {"re": re, "Optional": Optional, "difflib": difflib},
        )
        cls.split_parts = staticmethod(funcs["_split_window_output_parts"])
        cls.recover_parts = staticmethod(funcs["_recover_window_parts"])
        cls.pattern = re.compile("[¬↵]")

    # --- splitting -----------------------------------------------------------

    def test_split_exact_match(self):
        parts, note = self.split_parts("a¬b↵c", 3, self.pattern)
        self.assertEqual(parts, ["a", "b", "c"])
        self.assertIsNone(note)

    def test_split_returns_raw_parts_and_reports_mismatch(self):
        # The split no longer force-collapses with maxsplit (which mis-merged
        # whenever the miscounted separator was not the last one); it returns
        # every boundary and flags the count so recovery can realign.
        parts, note = self.split_parts("a¬b¬c", 2, self.pattern)
        self.assertEqual(parts, ["a", "b", "c"])
        self.assertIn("expected 2", note)

    # --- recovery / alignment ------------------------------------------------

    def test_exact_counts_assign_positionally(self):
        src = ["Concilia uniuersalia", "ſummus pontifex", "habeãt ordinem"]
        parts = ["Concilia uniuersalia", "ſummus pontifex", "habeant ordinem"]
        recovered, notes = self.recover_parts(parts, 3, src)
        self.assertEqual(recovered, parts)
        self.assertEqual(notes, [])

    def test_dropped_middle_line_keeps_source_without_shifting(self):
        # The reported bug: an out-of-distribution line ("476. col. 2. nu. 1.")
        # is dropped by the model; the recovered owned line must keep its own
        # source, NOT inherit the following line's expansion.
        src = [
            "tificem honorare debeant, oſtenditur. pa.",
            "476. col. 2. nu. 1.",                       # owned, dropped
            "Concilia uniuerſalia quem ordinem habeãt",
        ]
        parts = [
            "tificem honorare debeant, oſtenditur. pa.",
            "Concilia uniuerſalia quem ordinem habeant",  # the next line's expansion
        ]
        recovered, notes = self.recover_parts(
            parts, 3, src, owned_range=(1, 2))
        self.assertEqual(recovered[1], "476. col. 2. nu. 1.",
                         "Owned line must keep its source, not the next line.")
        self.assertEqual(recovered[0], src[0])
        self.assertEqual(recovered[2], "Concilia uniuerſalia quem ordinem habeant")
        self.assertTrue(any("OWNED" in n for n in notes))

    def test_truncated_tail_keeps_source(self):
        src = ["alpha line one", "beta line two", "gamma line three",
               "delta line four"]
        # Output cap truncates the last two lines.
        parts = ["alpha line one", "beta line two"]
        recovered, notes = self.recover_parts(parts, 4, src, owned_range=(0, 1))
        self.assertEqual(recovered[:2], parts)
        self.assertEqual(recovered[2], src[2])
        self.assertEqual(recovered[3], src[3])
        self.assertFalse(any("OWNED" in n for n in notes),
                         "Owned line (offset 0) was generated, so no OWNED note.")

    def test_extra_part_is_ignored_without_shifting(self):
        src = ["prima linea texta", "secunda linea texta", "tertia linea texta"]
        # Model emits a spurious separator splitting the FIRST line.
        parts = ["prima linea", " texta", "secunda linea texta",
                 "tertia linea texta"]
        recovered, notes = self.recover_parts(parts, 3, src, owned_range=(1, 2))
        # The two later lines must still land on their own source lines.
        self.assertEqual(recovered[1], "secunda linea texta")
        self.assertEqual(recovered[2], "tertia linea texta")
        self.assertTrue(any("extra" in n for n in notes))

    def test_unrelated_line_is_never_matched(self):
        # A single source line whose own output was dropped, offered only an
        # unrelated leftover part, must fall back to source — never adopt it.
        src = ["476. col. 2. nu. 1."]
        parts = ["Concilia uniuerſalia quem ordinem habeant"]
        recovered, notes = self.recover_parts(parts, 1, src, owned_range=(0, 1))
        self.assertEqual(recovered[0], "476. col. 2. nu. 1.")

    def test_over_split_owned_line_flagged_as_fragment(self):
        # The model emits a spurious separator inside the owned line, so its
        # output arrives as two parts; only the first fragment aligns.  The
        # owned line is matched (not dropped) but to a short fragment, which
        # must be surfaced so the over-split rate is measurable.
        src = ["alpha beta gamma delta epsilon"]
        parts = ["alpha beta gamma", "delta epsilon"]
        recovered, notes = self.recover_parts(parts, 1, src, owned_range=(0, 1))
        self.assertEqual(recovered[0], "alpha beta gamma")
        self.assertTrue(any("fragment" in n.lower() for n in notes),
                        f"expected an over-split fragment note, got {notes}")

    def test_full_match_owned_line_not_flagged_as_fragment(self):
        # A clean full match must NOT raise the fragment note.
        src = ["alpha beta gamma delta"]
        parts = ["alpha beta gamma delta"]
        recovered, notes = self.recover_parts(parts, 1, src, owned_range=(0, 1))
        self.assertEqual(recovered[0], "alpha beta gamma delta")
        self.assertFalse(any("fragment" in n.lower() for n in notes))


class SharedRunMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        funcs = _load_functions(
            root / "tei" / "tei_roundtrip.py",
            ["_merge_changes_by_shared_runs", "_contains_whitespace"],
            {"TextRun": object, "LINE_SEP": "\u00ac"},
        )
        cls.merge_changes = staticmethod(funcs["_merge_changes_by_shared_runs"])

    def test_merges_adjacent_changes_that_share_run_node(self):
        shared_node = object()
        runs = [types.SimpleNamespace(node=shared_node, plain_start=0, plain_end=6)]
        changes = [(2, 3, 2, 4), (4, 5, 5, 6)]
        merged = self.merge_changes(changes, runs, "abcdef", "abXYdZf")
        self.assertEqual(merged, [(2, 5, "XYdZ")])

    def test_keeps_changes_separate_without_shared_run_node(self):
        node_a = object()
        node_b = object()
        runs = [
            types.SimpleNamespace(node=node_a, plain_start=0, plain_end=3),
            types.SimpleNamespace(node=node_b, plain_start=3, plain_end=6),
        ]
        changes = [(1, 2, 1, 3), (4, 5, 5, 6)]
        merged = self.merge_changes(changes, runs, "abcdef", "aXYcdZf")
        self.assertEqual(merged, [(1, 2, "XY"), (4, 5, "Z")])


class CrossLineChoiceInsertionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        funcs = _load_functions(
            root / "tei" / "tei_roundtrip.py",
            ["_truncate_line_end", "_insert_choice_at_line_end"],
            {
                "ExtractedLine": object,
                "etree": etree,
                "_is_intoken_element": lambda node: node.tag == "hi",
            },
        )
        cls.truncate_line_end = staticmethod(funcs["_truncate_line_end"])
        cls.insert_choice_at_line_end = staticmethod(funcs["_insert_choice_at_line_end"])

    def test_insert_choice_skips_detached_tail_anchor_after_truncate(self):
        parent = etree.Element("p")
        lb = etree.SubElement(parent, "lb")
        lb.tail = "ab"
        inline = etree.SubElement(parent, "hi")
        inline.text = ""
        inline.tail = "x"

        line = types.SimpleNamespace(
            text_runs=[
                types.SimpleNamespace(node=lb, is_tail=True, plain_start=0, plain_end=2, text="ab"),
                types.SimpleNamespace(node=inline, is_tail=False, plain_start=5, plain_end=5, text=""),
                types.SimpleNamespace(node=inline, is_tail=True, plain_start=4, plain_end=5, text="x"),
            ],
            lb_element=lb,
        )

        self.truncate_line_end(line, at_offset=5)

        self.assertIsNone(inline.getparent())

        choice = etree.Element("choice")
        self.insert_choice_at_line_end(line, at_offset=5, choice=choice)

        self.assertIs(choice.getparent(), parent)
        self.assertEqual(list(parent), [lb, choice])


class WindowByteLenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        funcs = _load_functions(
            root / "infer" / "__init__.py",
            ["_window_byte_len"],
            {"LINE_SEP": "\u00ac", "LINE_BREAK": "\u21b5"},
        )
        cls.byte_len = staticmethod(funcs["_window_byte_len"])

    def test_counts_real_separator_bytes(self):
        lines = [
            {"id": "l0", "source_sic": "ab", "predicted_nonbreaking_next_line": "l1"},
            {"id": "l1", "source_sic": "cd", "predicted_nonbreaking_next_line": ""},
            {"id": "l2", "source_sic": "ef", "predicted_nonbreaking_next_line": ""},
        ]
        # ab(2) + ¬(2, nonbreaking l0->l1) + cd(2) + ↵(3, breaking l1->l2) + ef(2)
        self.assertEqual(self.byte_len(lines, [0, 1, 2]), 2 + 2 + 2 + 3 + 2)

    def test_single_line_has_no_separator(self):
        lines = [{"id": "l0", "source_sic": "abc",
                  "predicted_nonbreaking_next_line": ""}]
        self.assertEqual(self.byte_len(lines, [0]), 3)

    def test_multibyte_source_counted_in_bytes(self):
        # 'ſ' (U+017F) is two UTF-8 bytes.
        lines = [{"id": "l0", "source_sic": "\u017f",
                  "predicted_nonbreaking_next_line": ""}]
        self.assertEqual(self.byte_len(lines, [0]), 2)


if __name__ == "__main__":
    unittest.main()
