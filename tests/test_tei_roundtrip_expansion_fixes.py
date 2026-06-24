"""
Regression tests for three TEI-roundtrip expansion fixes (W0100 corpus):

1. Trailing/leading word-break punctuation must stay OUTSIDE a newly-built
   <choice> (comma/period/colon after incẽſti, anteponã, beatã, …).
2. A cross-line abbreviation whose changed glyph sits at the END of line 1,
   flush against the <lb/>, must pull line 2's plain continuation into the
   token (cõmemo+lb+rem, ornamẽ+lb+ta), replicating the <lb> with @sameAs in
   the <expan>. Mirror cases (change on L2) and glyph-variant L2 starts that
   must stay single-line are also pinned.
3. A leaked model line-break sentinel (U+21AC) is stripped, and the neighbour
   line's expansion that was merged in is redistributed across the nonbreaking
   chain, with an optional audit trail.

Style mirrors tests/test_tei_roundtrip_choice.py (unittest, lxml, imports from
tei.tei_roundtrip, sys.path insert to repo root).
"""

import unittest
import re
from lxml import etree

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tei.tei_roundtrip import (
    process_tei_xml,
    _sanitize_expansion,
    _make_tag_fn,
    TEI_NS,
    MODEL_LINE_SEP,
)

import tei.tei_roundtrip as trt
trt._tag = _make_tag_fn(f"{{{TEI_NS}}}")

SENT = MODEL_LINE_SEP  # U+21AC ↬

# Glyph fragments used across the corpus.
SF = '<g ref="#char017f">ſ</g>'
DF = '<g ref="#char00df">ß</g>'
AE = '<g ref="#char00e6">æ</g>'
AT = '<g ref="#chara0303">ã</g>'
ET = '<g ref="#chare0303">ẽ</g>'
OT = '<g ref="#charo0303">õ</g>'
UT = '<g ref="#charu0303">ũ</g>'


def _doc(body: str) -> str:
    return (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
        '<teiHeader><fileDesc><titleStmt><title>Test</title></titleStmt>'
        '<publicationStmt><p/></publicationStmt>'
        '<sourceDesc><p/></sourceDesc></fileDesc></teiHeader>'
        f'<text><body><p>{body}</p></body></text></TEI>'
    )


def _chain_body(lines: list[tuple[str, str]]) -> str:
    return "\n".join(f'<lb xml:id="{lid}"/>{txt}' for lid, txt in lines)


def _fixed_pipeline(expanded: dict, boundaries: dict | None = None):
    """Return fixed expansion/boundary results. Unexpanded lines default to
    their source_sic, matching the live pipeline."""
    def pipeline(rows, pre_annotated):
        full = {r["id"]: r["source_sic"] for r in rows}
        full.update(expanded)
        return full, (boundaries or {})
    return pipeline


def _run(body, expanded, boundaries=None, report=None):
    return process_tei_xml(
        _doc(body), _fixed_pipeline(expanded, boundaries),
        reconstruction_report=report,
    )


def _abbr_expan(xml: str) -> tuple[str, str]:
    m = re.search(r'<abbr>(.*?)</abbr>\s*<expan[^>]*>(.*?)</expan>', xml, re.S)
    if not m:
        return ("", "")
    strip = lambda s: re.sub(r'<[^>]+>', '', s)
    return strip(m.group(1)), strip(m.group(2))


def _choices(xml: str):
    root = etree.fromstring(xml.encode("utf-8"))
    return root.findall(f".//{{{TEI_NS}}}choice")


class TestNewlyBuiltChoicePunctuation(unittest.TestCase):
    """Issue 1 — edge punctuation stays outside a model-generated <choice>."""

    def _assert_punct_outside(self, body, expanded, ch):
        abbr, expan = _abbr_expan(_run(body, expanded))
        self.assertTrue(abbr, "no <choice> was built")
        self.assertFalse(abbr.endswith(ch),
            f"{ch!r} was swallowed into <abbr>: {abbr!r}")
        self.assertFalse(expan.endswith(ch),
            f"{ch!r} was swallowed into <expan>: {expan!r}")

    def test_comma_after_single_glyph(self):
        self._assert_punct_outside(
            f'<lb xml:id="L"/>inc{ET}{SF}ti, res', {"L": "incenſti, res"}, ",")

    def test_comma_plain_token(self):
        self._assert_punct_outside(
            f'<lb xml:id="L"/>virtutem inc{ET}duntur, qui',
            {"L": "virtutem incenduntur, qui"}, ",")

    def test_period(self):
        self._assert_punct_outside(
            f'<lb xml:id="L"/>laudibus antepon{AT}. Nam',
            {"L": "laudibus anteponam. Nam"}, ".")

    def test_colon(self):
        self._assert_punct_outside(
            f'<lb xml:id="L"/>crearit, beat{AT}: qu{AE}',
            {"L": "crearit, beatam: quæ"}, ":")

    def test_comma_multi_glyph_token(self):
        self._assert_punct_outside(
            f'<lb xml:id="L"/>pr{AE}{SF}tanti{DF}im{AT}, qu{AE}',
            {"L": "præſtantiſſimam, quæ"}, ",")

    def test_colon_multi_glyph_token(self):
        self._assert_punct_outside(
            f'<lb xml:id="L"/>clari{DF}im{AT}: qui',
            {"L": "clariſſimam: qui"}, ":")

    def test_produced_punctuation_stays_inside(self):
        # &c → etc. genuinely produces a period; it must remain in the token.
        abbr, expan = _abbr_expan(
            _run(f'<lb xml:id="L"/>uidelicet <g ref="#char0026">&amp;</g>c sequitur',
                 {"L": "uidelicet etc. sequitur"}))
        self.assertTrue(expan.endswith("."),
            f"produced period wrongly trimmed: {expan!r}")


class TestCrossLineL1Change(unittest.TestCase):
    """Issue 2 — change at the end of L1 pulls in L2's plain continuation."""

    def test_l1_glyph_inside_word_crosses(self):
        out = _run(f'<lb xml:id="A"/>Quid c{OT}memo\n<lb xml:id="B"/>rem {SF}anctos',
                   {"A": "Quid commemo", "B": "rem ſanctos"}, {"A": "B"})
        abbr, expan = _abbr_expan(out)
        self.assertEqual(expan, "commemorem")
        self.assertIn("sameAs", out)

    def test_l1_glyph_at_word_end_crosses(self):
        # Changed glyph is the LAST char before the <lb/> (trailing-newline case).
        out = _run(f'<lb xml:id="A"/>cui domus ornam{ET}\n<lb xml:id="B"/>ta tu{AE} nota',
                   {"A": "cui domus ornamen", "B": "ta tuæ nota"}, {"A": "B"})
        abbr, expan = _abbr_expan(out)
        self.assertEqual(expan, "ornamenta")
        self.assertIn("sameAs", out)

    def test_l2_change_still_crosses(self):
        # Control: change on L2 (præ+lb+stãtes) must keep working.
        out = _run(f'<lb xml:id="A"/><g ref="#char0026">&amp;</g> pr{AE}\n'
                   f'<lb xml:id="B"/>st{AT}tes ni{SF}i',
                   {"A": "& præ", "B": "stantes niſi"}, {"A": "B"})
        _, expan = _abbr_expan(out)
        self.assertEqual(expan, "præstantes")
        self.assertIn("sameAs", out)

    def test_glyph_variant_l2_start_stays_single_line(self):
        # Guard: L1 real abbr + L2 glyph variant (magnũ / cleſia) → single-line.
        out = _run(f'<lb xml:id="A"/>magn{UT}\n<lb xml:id="B"/>cle{SF}ia finis',
                   {"A": "magnum", "B": "clesia finis"}, {"A": "B"})
        chs = _choices(out)
        self.assertEqual(len(chs), 1, "expected exactly one single-line choice")
        self.assertIsNone(chs[0].find(f"{{{TEI_NS}}}abbr/{{{TEI_NS}}}lb"),
            "single-line choice must NOT contain an <lb> inside <abbr>")


class TestMultiTokenSharedNode(unittest.TestCase):
    """Pre-existing bug: two separate abbreviations on one line whose tokens
    share a tree node (a <g> tail spanning across whitespace into the next
    token) must not duplicate a character when the first token's expansion
    changes length. Right-to-left surgery truncates the shared node; the
    left change must read the current node, not a stale run snapshot."""

    def _reading(self, xml):
        root = etree.fromstring(xml.encode("utf-8"))
        for abbr in root.findall(f".//{{{TEI_NS}}}abbr"):
            abbr.getparent().remove(abbr)
        body = root.find(f".//{{{TEI_NS}}}body")
        return re.sub(r"\s+", " ", "".join(body.itertext())).strip()

    def test_two_glyph_tokens_first_changes_length(self):
        out = _run(f'<lb xml:id="L"/>c{OT}tra dicit N{OT} in',
                   {"L": "contra dicit Non in"})
        self.assertEqual(self._reading(out), "contra dicit Non in")
        self.assertEqual(len(_choices(out)), 2)

    def test_tokens_in_reading_order(self):
        out = _run(f'<lb xml:id="L"/>N{OT} c{OT}tra dicit',
                   {"L": "Non contra dicit"})
        self.assertEqual(self._reading(out), "Non contra dicit")

    def test_punct_token_then_glyph_token(self):
        out = _run(f'<lb xml:id="L"/>inc{ET}ti, et magn{UT} res',
                   {"L": "incenti, et magnum res"})
        self.assertEqual(self._reading(out), "incenti, et magnum res")


class TestSentinelSanitizer(unittest.TestCase):
    """Issue 3a — _sanitize_expansion keeps the segment matching the source."""

    def test_forward_leak(self):
        self.assertEqual(
            _sanitize_expansion("fidem at", "fidem at" + SENT + "tinent quæ Ro"),
            "fidem at")

    def test_reversed_leak(self):
        self.assertEqual(
            _sanitize_expansion("telligere nõ",
                                "contra Non in" + SENT + "telligere non"),
            "telligere non")

    def test_passthrough_without_sentinel(self):
        self.assertEqual(_sanitize_expansion("plenariũ", "plenarium"), "plenarium")

    def test_unrelated_segments_leave_unexpanded(self):
        self.assertEqual(
            _sanitize_expansion("quoddam uerbum",
                                "totally other" + SENT + "unrelated clause"),
            "quoddam uerbum")


class TestSentinelReconstruction(unittest.TestCase):
    """Issue 3b — chain repair recovers the partner line and audits it."""

    def _reading(self, xml):
        """Body text with each <choice> read as its <expan> (abbr dropped),
        whitespace-collapsed — i.e. the expanded reading of the document."""
        root = etree.fromstring(xml.encode("utf-8"))
        for abbr in root.findall(f".//{{{TEI_NS}}}abbr"):
            abbr.getparent().remove(abbr)
        body = root.find(f".//{{{TEI_NS}}}body")
        return re.sub(r"\s+", " ", "".join(body.itertext())).strip()

    def test_reversed_leak_recovers_previous_line(self):
        # L21 deliberately has two glyph tokens (the second of which,
        # cõtra→contra, changes length) to also exercise the shared-node path.
        report = []
        out = _run(
            _chain_body([("L21", f'c{OT}tra dicit N{OT} in'),
                         ("L22", f'telligere n{OT} auaris')]),
            {"L22": "contra dicit Non in" + SENT + "telligere non auaris"},
            {"L21": "L22"}, report)
        self.assertNotIn(SENT, out)
        self.assertIn("contra dicit Non in", self._reading(out))
        self.assertTrue(
            any(e["line_id"] == "L21" and e["from_line_id"] == "L22"
                and e["kind"] == "reconstructed" for e in report),
            f"missing L21←L22 reconstruction entry; got {report}")

    def test_forward_leak_recovers_next_line(self):
        report = []
        out = _run(
            _chain_body([("r36", 'Concilium in his quę ad fidem at'),
                         ("r37", f'tinent quæ tam patr{UT} Ro'),
                         ("r38", f'mani c{OT}ſenſu')]),
            {"r36": "Concilium in his quę ad fidem at" + SENT
                    + "tinent quæ tam patrum Ro",
             "r38": "mani conſenſu"},
            {"r36": "r37", "r37": "r38"}, report)
        self.assertNotIn(SENT, out)
        self.assertIn("tinent quæ tam patrum Ro", self._reading(out))
        # r38's own (independent) expansion must be preserved, not overwritten.
        self.assertIn("mani conſenſu", self._reading(out))
        self.assertTrue(
            any(e["line_id"] == "r37" and e["from_line_id"] == "r36"
                and e["kind"] == "reconstructed" for e in report))

    def test_clean_chain_untouched_empty_report(self):
        report = []
        out = _run(
            _chain_body([("A", f'inc{ET}ſti,'), ("B", f'nou{AT} res')]),
            {"A": "incenſti,", "B": "novam res"}, {"A": "B"}, report)
        self.assertEqual(report, [])
        self.assertNotIn(SENT, out)

    def test_report_is_optional(self):
        out = _run(
            _chain_body([("L21", f'c{OT}tra dicit'),
                         ("L22", f'telligere n{OT}')]),
            {"L22": "contra dicit" + SENT + "telligere non"}, {"L21": "L22"})
        self.assertNotIn(SENT, out)


class TestHiWrapperStraddle(unittest.TestCase):
    """Issue I — a token that STARTS inside an <hi>-rendered initial and
    straddles past </hi> into following siblings. The nest-inside-wrapper
    strategy (valid only for fully-contained tokens) duplicated the wrapper's
    tail after </hi>. The straddle must instead build a clean <choice> at the
    wrapper's position, cloning the rendered initial into <abbr> (it is dropped
    only from <expan>), with the abbr reading byte-identical to the source.
    A fully-CONTAINED wrapper token must still nest <choice> inside <hi>."""

    def _abbr_reading(self, xml):
        root = etree.fromstring(xml.encode("utf-8"))
        for expan in root.findall(f".//{{{TEI_NS}}}expan"):
            expan.getparent().remove(expan)
        body = root.find(f".//{{{TEI_NS}}}body")
        return re.sub(r"\s+", " ", "".join(body.itertext())).strip()

    def _plain(self, body):
        root = etree.fromstring(_doc(body).encode("utf-8"))
        b = root.find(f".//{{{TEI_NS}}}body")
        return re.sub(r"\s+", " ", "".join(b.itertext())).strip()

    def test_straddle_drops_hi_no_duplication(self):
        body = (
            f'<lb xml:id="L2002"/><hi rendition="#initCaps">L</hi>'
            f'Ateran{ET}{SF}is eccle{SF}i{AE} dignitas, o{SF}t{ET}'
            f'<lb xml:id="L2003"/>ditur. pag. 229. col. 1. nu. 4.'
        )
        out = _run(body,
                   {"L2002": "LAteranenſis eccleſiæ dignitas, oſten",
                    "L2003": "ditur. pag. 229. col. 1. nu. 4."},
                   {"L2002": "L2003"})
        # abbr side must round-trip to the original (no "Ateran" echo); compare
        # whitespace-insensitively since a break="no" lb rejoins split words.
        self.assertEqual(self._abbr_reading(out).replace(" ", ""),
                         self._plain(body).replace(" ", ""))
        # the rendered initial is PRESERVED inside <abbr>, dropped only in <expan>
        abbr_xml = re.search(r'<abbr>(.*?)</abbr>', out, re.S).group(1)
        expan_xml = re.search(r'<expan[^>]*>(.*?)</expan>', out, re.S).group(1)
        self.assertIn('<hi rendition="#initCaps">L</hi>Ateran', abbr_xml)
        self.assertNotIn("<hi", expan_xml)
        abbr, expan = _abbr_expan(out)
        self.assertEqual(abbr, "LAteranẽſis")
        self.assertEqual(expan, "LAteranenſis")

    def test_contained_wrapper_still_nests(self):
        body = f'<lb xml:id="L"/>uir <hi rendition="#initCaps">c{OT}</hi> ait'
        out = _run(body, {"L": "uir con ait"})
        # choice nests INSIDE the wrapper, which is preserved
        self.assertIn('<hi rendition="#initCaps"><choice', out)
        self.assertIn("</choice></hi>", out)
        self.assertEqual(self._abbr_reading(out).replace(" ", ""),
                         self._plain(body).replace(" ", ""))


class TestRunawayExpansionOverrun(unittest.TestCase):
    """Issue II — the model occasionally over-runs a line, emitting the NEXT
    line's text as part of this line's expansion with no U+21AC sentinel. The
    cross-line merge would otherwise swallow that runaway clause into the
    <expan> and split the following tail. Such a token must be left UNEXPANDED.
    A genuine cross-line abbreviation must still expand."""

    OG = '<g ref="#charo0300">ò</g>'   # ò
    EG = '<g ref="#chare0328">ę</g>'   # ę

    def _abbr_reading(self, xml):
        root = etree.fromstring(xml.encode("utf-8"))
        for expan in root.findall(f".//{{{TEI_NS}}}expan"):
            expan.getparent().remove(expan)
        body = root.find(f".//{{{TEI_NS}}}body")
        return re.sub(r"\s+", " ", "".join(body.itertext())).strip()

    def _plain(self, body):
        root = etree.fromstring(_doc(body).encode("utf-8"))
        b = root.find(f".//{{{TEI_NS}}}body")
        return re.sub(r"\s+", " ", "".join(b.itertext())).strip()

    def test_runaway_overrun_leaves_token_unexpanded(self):
        body = _chain_body([
            ("L1040", f'Iuri{SF}dictionis pote{SF}tas, etiam in foro c{OT}{SF}cien'),
            ("L1041", f'ti{AE}, qu{self.OG}d omnibus {SF}acerdotibus {self.EG}qualiter'),
            ("L1042", 'non conueniat. ibid. nu. 6.'),
        ])
        out = _run(body,
                   {"L1040": ("Iuriſdictionis poteſtas, etiam in foro "
                              "conſcienſiæ, quòd omnibus ſacerdotibus ęqualiter"),
                    "L1041": "tiæ, quòd omnibus ſacerdotibus ęqualiter",
                    "L1042": "non conueniat. ibid. nu. 6."},
                   {"L1040": "L1041"})
        # runaway → no <choice> at all; source characters fully intact (the
        # boundary classifier may set break="no", which only drops whitespace).
        self.assertEqual(len(_choices(out)), 0)
        self.assertEqual(self._abbr_reading(out).replace(" ", ""),
                         self._plain(body).replace(" ", ""))

    def test_genuine_cross_line_still_expands(self):
        body = _chain_body([("A", f'Quid c{OT}memo'), ("B", "rem ait")])
        out = _run(body, {"A": "Quid commemo", "B": "rem ait"}, {"A": "B"})
        abbr, expan = _abbr_expan(out)
        self.assertEqual(abbr, "cõmemorem")
        self.assertEqual(expan, "commemorem")


if __name__ == "__main__":
    unittest.main()
