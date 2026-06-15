import ast
import re
import types
import unittest
from pathlib import Path
from typing import Optional


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
        root = Path(__file__).resolve().parents[1]
        funcs = _load_functions(
            root / "infer" / "__init__.py",
            ["_split_window_output_parts", "_recover_window_parts"],
            {"re": re, "Optional": Optional},
        )
        cls.split_parts = staticmethod(funcs["_split_window_output_parts"])
        cls.recover_parts = staticmethod(funcs["_recover_window_parts"])
        cls.pattern = re.compile("[¬↵]")

    def test_split_exact_match(self):
        parts, note = self.split_parts("a¬b↵c", 3, self.pattern)
        self.assertEqual(parts, ["a", "b", "c"])
        self.assertIsNone(note)

    def test_split_uses_maxsplit_recovery(self):
        parts, note = self.split_parts("a¬b¬c", 2, self.pattern)
        self.assertEqual(parts, ["a", "b¬c"])
        self.assertIn("recovered via maxsplit", note)

    def test_recover_missing_parts_from_source(self):
        recovered, notes = self.recover_parts(
            ["exp-1", "exp-2"], 3, ["src-1", "src-2", "src-3"],
        )
        self.assertEqual(recovered, ["exp-1", "exp-2", "src-3"])
        self.assertTrue(any("filled 1 missing part" in n for n in notes))


class SharedRunMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        funcs = _load_functions(
            root / "tei" / "tei_roundtrip.py",
            ["_merge_changes_by_shared_runs"],
            {"TextRun": object},
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


if __name__ == "__main__":
    unittest.main()
