"""
Regression tests for the TEI roundtrip handling of pre-existing <choice> elements.

These tests verify that:
1. Pre-existing <choice><abbr>...</abbr><expan>...</expan></choice> constructs
   are preserved intact when the model's expansion matches.
2. The model's resp is merged into the existing <expan>@resp.
3. Punctuation and tail text remain outside the <choice> element.
4. No nested <choice> elements are produced for matching expansions.
5. No elements are duplicated.
6. Namespace declarations appear only on the root element.
7. Empty xmlns and xml:id attributes are never inserted.
"""

import unittest
from lxml import etree

import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tei.tei_roundtrip import (
    process_tei_xml,
    extract_lines,
    apply_expansions,
    _detect_namespace,
    _make_tag_fn,
    TEI_NS,
    XML_NS,
    EXPANSION_MODEL,
)

# Set the module-level _tag before tests
import tei.tei_roundtrip as trt
trt._tag = _make_tag_fn(f"{{{TEI_NS}}}")


def _make_tei_doc(body_content: str) -> str:
    """Wrap body content in a minimal TEI document."""
    return (
        f'<TEI xmlns="http://www.tei-c.org/ns/1.0">'
        f'<teiHeader><fileDesc><titleStmt><title>Test</title></titleStmt>'
        f'<publicationStmt><p/></publicationStmt>'
        f'<sourceDesc><p/></sourceDesc></fileDesc></teiHeader>'
        f'<text><body><p>'
        f'<lb xml:id="lb-001"/>{body_content}'
        f'</p></body></text></TEI>'
    )


def _noop_pipeline(rows, pre_annotated):
    """Pipeline that returns text unchanged (no expansions, no boundaries)."""
    expanded = {row["id"]: row["source_sic"] for row in rows}
    return expanded, {}


def _expanding_pipeline(expansions_map):
    """Create a pipeline that applies specific expansions."""
    def pipeline(rows, pre_annotated):
        expanded = {}
        for row in rows:
            text = row["source_sic"]
            for orig, repl in expansions_map.items():
                text = text.replace(orig, repl)
            expanded[row["id"]] = text
        return expanded, {}
    return pipeline


class TestPreExistingChoicePreserved(unittest.TestCase):
    """Test that pre-existing choice elements are not mangled."""

    def test_choice_with_matching_expansion_preserved(self):
        """Case 1: model expands to same as existing expan → just add resp."""
        body = (
            'tam ex tertio '
            '<choice resp="#auto" xml:id="W0100-00-0493-ce-45eb">'
            '<abbr>politicor<g ref="#charu0303">\u0169</g></abbr>'
            '<expan resp="#CR #auto">politicorum</expan>'
            '</choice>, qu<g ref="#chara0300">\u00e0</g>m'
        )
        xml_input = _make_tei_doc(body)

        # Model expands "politicorũ" → "politicorum" (same as existing)
        pipeline = _expanding_pipeline({"politicor\u0169": "politicorum"})
        result = process_tei_xml(xml_input, pipeline)

        # The existing choice should be preserved, not nested
        tree = etree.fromstring(result.encode())
        ns = {"tei": TEI_NS}

        choices = tree.findall(f".//{{{TEI_NS}}}choice")
        # Should only have one choice (the original), not nested
        # Filter out appInfo choices
        body_choices = [
            c for c in choices
            if c.get(f"{{{XML_NS}}}id") == "W0100-00-0493-ce-45eb"
        ]
        self.assertEqual(len(body_choices), 1, "Should have exactly one choice element")

        choice = body_choices[0]
        # Check that abbr content is preserved
        abbr = choice.find(f"{{{TEI_NS}}}abbr")
        self.assertIsNotNone(abbr)
        abbr_text = "".join(abbr.itertext())
        self.assertEqual(abbr_text, "politicor\u0169")

        # Check that expan has merged resp
        expan = choice.find(f"{{{TEI_NS}}}expan")
        self.assertIsNotNone(expan)
        self.assertIn("#CR", expan.get("resp", ""))
        self.assertIn(f"#{EXPANSION_MODEL}", expan.get("resp", ""))

        # Check that comma is in the tail (outside choice)
        self.assertTrue(choice.tail and choice.tail.startswith(","),
                       f"Comma should be in choice tail, got: {choice.tail!r}")

    def test_choice_unchanged_when_no_expansion(self):
        """Case 4: model doesn't change anything → choice completely untouched."""
        body = (
            'in tribus '
            '<choice resp="#auto" xml:id="W0100-00-0025-ce-0563">'
            '<abbr>c<g ref="#charo0303">\u00f5</g><g ref="#char017f">\u017f</g>i<g ref="#char017f">\u017f</g>tat</abbr>'
            '<expan resp="#CR #auto">con<g ref="#char017f">\u017f</g>i<g ref="#char017f">\u017f</g>tat</expan>'
            '</choice>, vi'
        )
        xml_input = _make_tei_doc(body)

        # Pipeline returns text unchanged
        result = process_tei_xml(xml_input, _noop_pipeline)

        tree = etree.fromstring(result.encode())
        choices = [
            c for c in tree.findall(f".//{{{TEI_NS}}}choice")
            if c.get(f"{{{XML_NS}}}id") == "W0100-00-0025-ce-0563"
        ]
        self.assertEqual(len(choices), 1, "Should have exactly one choice")

        choice = choices[0]
        # Comma should still be in tail
        self.assertTrue(choice.tail and "," in choice.tail,
                       f"Comma should be in choice tail, got: {choice.tail!r}")

        # No spurious abbr/expan outside choice
        body_el = tree.find(f".//{{{TEI_NS}}}body")
        body_str = etree.tostring(body_el, encoding="unicode")
        # Count occurrences of <abbr> - should only be inside the one choice
        self.assertEqual(body_str.count("<abbr"), 1,
                        f"Should have exactly one <abbr> element")
        self.assertEqual(body_str.count("<expan"), 1,
                        f"Should have exactly one <expan> element")

    def test_no_nested_choice_for_cross_line_with_existing_choice(self):
        """Case 3: existing choice before a line break is not corrupted."""
        body = (
            'man<g ref="#char017f">\u017f</g>uetudine\n'
            '<choice resp="#auto" xml:id="W0100-00-0663-ce-68cf">\n'
            '    <abbr>corri<lb xml:id="W0100-00-0663-lb-2046" rendition="#hyphen" break="no"/>g<g ref="#chara0303">\u00e3</g>t</abbr>\n'
            '    <expan resp="#CR #auto">corri<lb sameAs="#W0100-00-0663-lb-2046" rendition="#hyphen" break="no" xml:id="W0100-00-0663-lb-s456"/>gant</expan>\n'
            '</choice>. V<g ref="#charn0303">\u00f1</g>\n'
        )
        xml_input = _make_tei_doc(body)

        # Model expands "corrigãt" → "corrigant" (same as existing)
        pipeline = _expanding_pipeline({"corrig\u00e3t": "corrigant"})
        result = process_tei_xml(xml_input, pipeline)

        tree = etree.fromstring(result.encode())
        choices = [
            c for c in tree.findall(f".//{{{TEI_NS}}}choice")
            if c.get(f"{{{XML_NS}}}id") == "W0100-00-0663-ce-68cf"
        ]
        self.assertEqual(len(choices), 1, "Should have exactly one choice for this id")

        # No spurious abbr/expan outside any choice
        body_el = tree.find(f".//{{{TEI_NS}}}body")
        body_str = etree.tostring(body_el, encoding="unicode")

        # Count standalone abbr/expan not inside a choice
        # The body should contain the choice element intact
        self.assertNotIn("</choice><abbr>", body_str,
                        "No abbr elements should appear outside choice")


class TestNamespaceCleanup(unittest.TestCase):
    """Test that namespace handling is correct."""

    def test_no_redundant_namespace_declarations(self):
        """Namespace should only appear on root element."""
        body = (
            'test <choice resp="#auto" xml:id="c1">'
            '<abbr>t<g ref="#x">\u0169</g>st</abbr>'
            '<expan resp="#CR #auto">test</expan>'
            '</choice> end'
        )
        xml_input = _make_tei_doc(body)
        result = process_tei_xml(xml_input, _noop_pipeline)

        # Count xmlns declarations - should only appear once (on root)
        xmlns_count = result.count('xmlns="http://www.tei-c.org/ns/1.0"')
        self.assertEqual(xmlns_count, 1,
                        f"Namespace should appear only once, found {xmlns_count} times")

    def test_no_empty_xmlns_attributes(self):
        """No empty xmlns="" attributes should appear."""
        body = 'simple text without abbreviations'
        xml_input = _make_tei_doc(body)
        result = process_tei_xml(xml_input, _noop_pipeline)
        self.assertNotIn('xmlns=""', result)

    def test_no_empty_xml_id_attributes(self):
        """No empty xml:id="" attributes should appear."""
        body = 'simple text without abbreviations'
        xml_input = _make_tei_doc(body)
        result = process_tei_xml(xml_input, _noop_pipeline)
        self.assertNotIn('xml:id=""', result)


class TestPunctuationPreserved(unittest.TestCase):
    """Test that punctuation stays outside choice elements."""

    def test_comma_not_swallowed_into_choice(self):
        """Trailing comma after choice should remain in tail."""
        body = (
            'tam ex tertio '
            '<choice resp="#auto" xml:id="c1">'
            '<abbr>politicor<g ref="#charu0303">\u0169</g></abbr>'
            '<expan resp="#CR #auto">politicorum</expan>'
            '</choice>, qu<g ref="#chara0300">\u00e0</g>m'
        )
        xml_input = _make_tei_doc(body)
        pipeline = _expanding_pipeline({"politicor\u0169": "politicorum"})
        result = process_tei_xml(xml_input, pipeline)

        # Verify comma is NOT inside any choice/abbr/expan
        tree = etree.fromstring(result.encode())
        choices = [
            c for c in tree.findall(f".//{{{TEI_NS}}}choice")
            if c.get(f"{{{XML_NS}}}id") == "c1"
        ]
        self.assertEqual(len(choices), 1)
        choice = choices[0]

        # abbr should not contain comma
        abbr = choice.find(f"{{{TEI_NS}}}abbr")
        abbr_text = "".join(abbr.itertext())
        self.assertNotIn(",", abbr_text)

        # expan should not contain comma
        expan = choice.find(f"{{{TEI_NS}}}expan")
        expan_text = "".join(expan.itertext())
        self.assertNotIn(",", expan_text)

    def test_period_not_swallowed_into_choice(self):
        """Trailing period after choice should remain in tail."""
        body = (
            'text '
            '<choice resp="#auto" xml:id="c2">'
            '<abbr>congreg<g ref="#chara0303">\u00e3</g>ti</abbr>'
            '<expan resp="#CRPY #auto">congreganti</expan>'
            '</choice>. Super quo'
        )
        xml_input = _make_tei_doc(body)
        pipeline = _expanding_pipeline({"congreg\u00e3ti": "congreganti"})
        result = process_tei_xml(xml_input, pipeline)

        tree = etree.fromstring(result.encode())
        choices = [
            c for c in tree.findall(f".//{{{TEI_NS}}}choice")
            if c.get(f"{{{XML_NS}}}id") == "c2"
        ]
        self.assertEqual(len(choices), 1)
        choice = choices[0]

        # Period should be in tail, not in abbr or expan
        self.assertTrue(choice.tail and choice.tail.startswith("."),
                       f"Period should be in choice tail, got: {choice.tail!r}")


class TestDisagreementFlaggedForReview(unittest.TestCase):
    """Test that model disagreement flags for manual inspection."""

    def test_disagreeing_expansion_flagged_cert_low(self):
        """When model expands differently, flag with @cert='low' and no model resp."""
        body = (
            'tam ex tertio '
            '<choice resp="#auto" xml:id="W0100-00-0493-ce-45eb">'
            '<abbr>politicor<g ref="#charu0303">\u0169</g></abbr>'
            '<expan resp="#CR #auto">politicorum</expan>'
            '</choice>, qu<g ref="#chara0300">\u00e0</g>m'
        )
        xml_input = _make_tei_doc(body)

        # Model expands "politicorũ" → "politicorUNDO" (DIFFERENT from existing)
        pipeline = _expanding_pipeline({"politicor\u0169": "politicorUNDO"})
        result = process_tei_xml(xml_input, pipeline)

        tree = etree.fromstring(result.encode())

        # Find the choice element
        body_choices = [
            c for c in tree.findall(f".//{{{TEI_NS}}}choice")
            if c.get(f"{{{XML_NS}}}id") == "W0100-00-0493-ce-45eb"
        ]
        self.assertEqual(len(body_choices), 1, "Should have exactly one choice element")

        choice = body_choices[0]
        expan = choice.find(f"{{{TEI_NS}}}expan")
        self.assertIsNotNone(expan)

        # The model identifier should NOT be in resp (expansion was discarded)
        self.assertNotIn(f"#{EXPANSION_MODEL}", expan.get("resp", ""))

        # The original resp should still be there
        self.assertIn("#CR", expan.get("resp", ""))
        self.assertIn("#auto", expan.get("resp", ""))

        # Should be flagged with cert="low" for manual inspection
        self.assertEqual(expan.get("cert"), "low")

        # Existing expansion text should be unchanged
        expan_text = "".join(expan.itertext())
        self.assertEqual(expan_text, "politicorum")

    def test_agreeing_expansion_no_cert_flag(self):
        """When model agrees with existing expan, no @cert flag is added."""
        body = (
            'tam ex tertio '
            '<choice resp="#auto" xml:id="W0100-00-0493-ce-45eb">'
            '<abbr>politicor<g ref="#charu0303">\u0169</g></abbr>'
            '<expan resp="#CR #auto">politicorum</expan>'
            '</choice>, qu<g ref="#chara0300">\u00e0</g>m'
        )
        xml_input = _make_tei_doc(body)

        # Model expands "politicorũ" → "politicorum" (SAME as existing)
        pipeline = _expanding_pipeline({"politicor\u0169": "politicorum"})
        result = process_tei_xml(xml_input, pipeline)

        tree = etree.fromstring(result.encode())

        body_choices = [
            c for c in tree.findall(f".//{{{TEI_NS}}}choice")
            if c.get(f"{{{XML_NS}}}id") == "W0100-00-0493-ce-45eb"
        ]
        self.assertEqual(len(body_choices), 1)

        choice = body_choices[0]
        expan = choice.find(f"{{{TEI_NS}}}expan")
        self.assertIsNotNone(expan)

        # Model resp should be added (agreement)
        self.assertIn(f"#{EXPANSION_MODEL}", expan.get("resp", ""))

        # No cert flag (no disagreement)
        self.assertIsNone(expan.get("cert"))


class TestNoSpuriousChoiceFromGlyphVariants(unittest.TestCase):
    """Test that glyph-variant normalization does NOT produce spurious choices."""

    def test_chain_with_existing_choice_no_spurious_cross_line_choices(self):
        """
        Regression test: when a chain contains a pre-existing <choice> and the
        model only expands the real abbreviation (magnũ→magnum) but normalizes
        glyphs elsewhere (ſ→s, è→e), no spurious <choice> elements should be
        created for the glyph-normalized tokens.
        """
        body = (
            'hoc'
            '<note anchored="false" place="margin" xml:id="W0100-00-0102-nm-0675">'
            '<p xml:id="W0100-00-0102-pa-0a9b">'
            '<lb xml:id="W0100-00-0102-lb-m006"/>'
            '<hi rendition="#it">Ephe. 5.</hi></p></note>'
            '<lb xml:id="W0100-00-0102-lb-2017"/>'
            'magn<g ref="#charu0303">\u0169</g> e<g ref="#char017f">\u017f</g>t in '
            '<choice xml:id="W0100-00-0102-ce-0e0d">'
            '<abbr>Ch<g ref="#charr0303">r\u0303</g>o</abbr>'
            '<expan resp="#auto">Christo</expan></choice>'
            ', <g ref="#char0026">&amp;</g> ec'
            '<lb xml:id="W0100-00-0102-lb-2018"/>'
            'cle<g ref="#char017f">\u017f</g>ia. apert<g ref="#chare0300">\u00e8</g>'
            ' de<g ref="#char017f">\u017f</g>ignauit'
            '<lb xml:id="W0100-00-0102-lb-2019"/>'
            'matrimonium carnale'
        )
        xml_input = _make_tei_doc(body)

        # Pipeline that expands only "magnũ"→"magnum" and "Chr̃o"→"Christo",
        # and normalizes ſ→s, è→e (as the ByT5 model tends to do)
        def pipeline(rows, pre_annotated):
            expanded = {}
            boundaries = {}
            for row in rows:
                text = row["source_sic"]
                # Expand real abbreviation
                text = text.replace("magn\u0169", "magnum")
                # Expand existing choice's abbr text (model sees abbr text)
                text = text.replace("Chr\u0303o", "Christo")
                # Model also normalizes glyphs (NOT real expansions)
                text = text.replace("\u017f", "s")  # long-s → s
                text = text.replace("\u00e8", "e")  # è → e
                expanded[row["id"]] = text
            # Boundary prediction: lb-2017 → lb-2018 (nonbreaking, for ec-clesia)
            boundaries["W0100-00-0102-lb-2017"] = "W0100-00-0102-lb-2018"
            return expanded, boundaries
        result = process_tei_xml(xml_input, pipeline)

        tree = etree.fromstring(result.encode())

        # Find all <choice> elements in the body
        all_choices = tree.findall(f".//{{{TEI_NS}}}choice")

        # We expect:
        # 1. The pre-existing choice W0100-00-0102-ce-0e0d (preserved)
        # 2. A new choice for magnũ→magnum
        # 3. A cross-line choice for ec/clesia→ecclesia
        # But NOT any spurious choices for "apertè", "deſignauit", etc.

        # Filter to body choices only (exclude appInfo)
        body_el = tree.find(f".//{{{TEI_NS}}}body")
        body_choices = body_el.findall(f".//{{{TEI_NS}}}choice")

        # Should have at most 3 choices (existing + magnũ + ec/clesia)
        self.assertLessEqual(
            len(body_choices), 3,
            f"Expected at most 3 <choice> elements but found {len(body_choices)}. "
            f"Spurious choices were introduced for glyph-variant-only changes."
        )

        # The existing choice should still be present
        existing = [c for c in body_choices
                    if c.get(f"{{{XML_NS}}}id") == "W0100-00-0102-ce-0e0d"]
        self.assertEqual(len(existing), 1, "Pre-existing choice was lost")

        # Verify no expan text contains garbage like "ecmagnum" or "Christo, &"
        for choice in body_choices:
            expan = choice.find(f"{{{TEI_NS}}}expan")
            if expan is not None:
                expan_text = "".join(expan.itertext())
                self.assertNotIn(
                    "ecmagnum", expan_text,
                    "Spurious content in <expan>: earlier text leaked in"
                )

    def test_glyph_only_change_filtered(self):
        """Single-line glyph-only changes (ſ→s) should NOT create choice."""
        body = (
            'cle<g ref="#char017f">\u017f</g>ia'
        )
        xml_input = _make_tei_doc(body)

        # Model normalizes ſ→s
        def pipeline(rows, pre_annotated):
            expanded = {}
            for row in rows:
                text = row["source_sic"]
                text = text.replace("\u017f", "s")
                expanded[row["id"]] = text
            return expanded, {}
        result = process_tei_xml(xml_input, pipeline)

        tree = etree.fromstring(result.encode())
        body_el = tree.find(f".//{{{TEI_NS}}}body")
        body_choices = body_el.findall(f".//{{{TEI_NS}}}choice")

        # No choice should be introduced for a glyph-only change
        self.assertEqual(
            len(body_choices), 0,
            "Glyph-only change (ſ→s) should not introduce a <choice> element"
        )




class TestCrossLineAbbreviationExpansion(unittest.TestCase):
    """
    Regression tests for cross-line abbreviation expansion (the pr\u00e6st\u00e3tes bug).

    PR #10 fixed _expand_left/_expand_right to stop at LINE_SEP, which
    correctly prevents spurious cross-line choices from glyph variants.
    But it also broke genuine cross-line abbreviations that span a
    non-breaking line boundary, because the diff was being split into two
    single-line changes instead of one cross-line change.

    The fix: after glyph-variant filtering, boundary-adjacent single-line
    changes are merged back into a single cross-line change that
    _apply_cross_line_choice can handle.
    """

    def _make_doc(self, body: str) -> str:
        return (
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            '<teiHeader><fileDesc><titleStmt><title>T</title></titleStmt>'
            '<publicationStmt><p/></publicationStmt>'
            '<sourceDesc><p/></sourceDesc></fileDesc></teiHeader>'
            '<text><body><p>'
            '<lb xml:id="W0100-00-0009-lb-0034"/>'
            + body +
            '</p></body></text></TEI>'
        )

    def _body_str(self, result: str) -> str:
        tree = etree.fromstring(result.encode())
        body_el = tree.find(f"{{{TEI_NS}}}text/{{{TEI_NS}}}body")
        return etree.tostring(body_el, encoding="unicode")

    # The input from the bug report
    INPUT_BODY = (
        'altas, <g ref="#char0026">&amp;</g> pr<g ref="#char00e6">\u00e6</g>'
        '<lb xml:id="W0100-00-0009-lb-0035"/>'
        'st<g ref="#chara0303">\u00e3</g>tes ni<g ref="#char017f">\u017f</g>i magnus'
    )

    def _pipeline_a(self, rows, pre_annotated):
        """Model correctly expands both halves: pr\u00e6 → prae, st\u00e3tes → stantes."""
        expanded = {r["id"]: r["source_sic"] for r in rows}
        expanded["W0100-00-0009-lb-0034"] = "altas, & prae"
        expanded["W0100-00-0009-lb-0035"] = "stantes ni\u017fi magnus"
        return expanded, {"W0100-00-0009-lb-0034": "W0100-00-0009-lb-0035"}

    def _pipeline_b(self, rows, pre_annotated):
        """Model expands only the \u00e3 part: st\u00e3tes → stantes, pr\u00e6 unchanged."""
        expanded = {r["id"]: r["source_sic"] for r in rows}
        expanded["W0100-00-0009-lb-0034"] = "altas, & pr\u00e6"   # \u00e6 NOT expanded
        expanded["W0100-00-0009-lb-0035"] = "stantes ni\u017fi magnus"
        return expanded, {"W0100-00-0009-lb-0034": "W0100-00-0009-lb-0035"}

    def _get_choices(self, result: str) -> list:
        tree = etree.fromstring(result.encode())
        body = tree.find(f"{{{TEI_NS}}}text/{{{TEI_NS}}}body")
        return body.findall(f".//{{{TEI_NS}}}choice")

    def test_scenario_a_produces_single_cross_line_choice(self):
        """
        When the model expands both halves (pr\u00e6→prae AND st\u00e3tes→stantes),
        the result should be ONE cross-line <choice> wrapping the full
        pr\u00e6[lb]st\u00e3tes token, not two separate single-line choices.
        """
        xml_input = self._make_doc(self.INPUT_BODY)
        result = process_tei_xml(xml_input, self._pipeline_a)
        choices = self._get_choices(result)

        self.assertEqual(
            len(choices), 1,
            f"Expected exactly 1 cross-line <choice>, got {len(choices)}. "
            f"Body: {self._body_str(result)}"
        )
        choice = choices[0]
        abbr = choice.find(f"{{{TEI_NS}}}abbr")
        expan = choice.find(f"{{{TEI_NS}}}expan")
        self.assertIsNotNone(abbr)
        self.assertIsNotNone(expan)

        # <abbr> must contain the <lb/> element (cross-line structure)
        lb_in_abbr = abbr.find(f"{{{TEI_NS}}}lb")
        self.assertIsNotNone(lb_in_abbr,
            "<abbr> must contain the <lb/> element for a cross-line choice")

        # <abbr> text must cover both pr\u00e6 and st\u00e3tes
        abbr_text = "".join(abbr.itertext())
        self.assertIn("pr", abbr_text)
        self.assertIn("\u00e6", abbr_text,   # æ
            "æ should be inside <abbr>")
        self.assertIn("st", abbr_text)
        self.assertIn("\u00e3", abbr_text,   # ã
            "ã should be inside <abbr>")

        # <expan> must contain "prae" and "stantes" (the model's expanded forms)
        expan_text = "".join(expan.itertext())
        self.assertIn("prae", expan_text,
            f"'prae' missing from <expan>; got: {expan_text!r}")
        self.assertIn("stantes", expan_text,
            f"'stantes' missing from <expan>; got: {expan_text!r}")

        # <expan> must also contain a <lb sameAs> element (cross-line structure)
        lb_in_expan = expan.find(f"{{{TEI_NS}}}lb")
        self.assertIsNotNone(lb_in_expan,
            "<expan> must contain a <lb sameAs=…> element")
        self.assertIn("W0100-00-0009-lb-0035",
                      lb_in_expan.get("sameAs", ""),
                      "<lb> in <expan> should have sameAs pointing to the original lb")

    def test_scenario_b_produces_cross_line_choice_not_single_line(self):
        """
        When the model expands only the L2 part (st\u00e3tes→stantes) but leaves L1
        unchanged (pr\u00e6 stays pr\u00e6), the result should still be a cross-line
        <choice> wrapping both parts — not a single-line L2-only choice that
        leaves pr\u00e6 outside any markup.
        """
        xml_input = self._make_doc(self.INPUT_BODY)
        result = process_tei_xml(xml_input, self._pipeline_b)
        choices = self._get_choices(result)
        body_s = self._body_str(result)

        self.assertEqual(
            len(choices), 1,
            f"Expected exactly 1 cross-line <choice>, got {len(choices)}. Body: {body_s}"
        )
        choice = choices[0]
        abbr = choice.find(f"{{{TEI_NS}}}abbr")
        expan = choice.find(f"{{{TEI_NS}}}expan")
        self.assertIsNotNone(abbr)
        self.assertIsNotNone(expan)

        # <abbr> must contain the <lb/> (cross-line, not just L2)
        lb_in_abbr = abbr.find(f"{{{TEI_NS}}}lb")
        self.assertIsNotNone(lb_in_abbr,
            "<abbr> must contain the <lb/> element; currently only L2 is wrapped")

        # <abbr> text must cover L1 part (pr\u00e6)
        abbr_text = "".join(abbr.itertext())
        self.assertIn("pr", abbr_text,
            "'pr' (L1 word start) must be inside <abbr>")

        # <expan> must cover L2 expansion
        expan_text = "".join(expan.itertext())
        self.assertIn("stantes", expan_text,
            f"'stantes' missing from <expan>; got: {expan_text!r}")

    def test_glyph_variant_at_l2_start_does_not_produce_cross_line_choice(self):
        """
        Regression: a glyph-variant-only change at the very start of L2
        (e.g. cleſia→clesia) must NOT produce a cross-line <choice> even
        though it is adjacent to the LINE_SEP boundary.  The glyph-variant
        filter must neutralise it before the boundary-merge logic runs.
        """
        body = (
            'ec'
            '<lb xml:id="lb-A"/>'
            'cle<g ref="#char017f">\u017f</g>ia. finis'
        )
        xml_input = self._make_doc(body)

        def pipeline(rows, pre_annotated):
            expanded = {r["id"]: r["source_sic"] for r in rows}
            # L1 unchanged, L2: ſ→s only (glyph variant)
            expanded["W0100-00-0009-lb-0034"] = "ec"
            expanded["lb-A"] = "clesia. finis"
            return expanded, {"W0100-00-0009-lb-0034": "lb-A"}

        result = process_tei_xml(xml_input, pipeline)
        choices = self._get_choices(result)
        self.assertEqual(
            len(choices), 0,
            "A glyph-variant-only change at L2 start must NOT create a <choice>. "
            f"Body: {self._body_str(result)}"
        )

    def test_real_l1_abbr_plus_glyph_l2_start_stays_single_line(self):
        """
        Regression: if L1's last word is a real abbreviation (magn\u0169→magnum)
        that ends at the LINE_SEP, but L2's first word is only a glyph variant
        (cleſia→clesia), only the L1 abbreviation should get a single-line
        <choice>.  No cross-line choice should be created.
        """
        body = (
            'magn<g ref="#charu0303">\u0169</g>'
            '<lb xml:id="lb-B"/>'
            'cle<g ref="#char017f">\u017f</g>ia finis'
        )
        xml_input = self._make_doc(body)

        def pipeline(rows, pre_annotated):
            expanded = {r["id"]: r["source_sic"] for r in rows}
            expanded["W0100-00-0009-lb-0034"] = "magnum"  # real abbr
            expanded["lb-B"] = "clesia finis"              # glyph only
            return expanded, {"W0100-00-0009-lb-0034": "lb-B"}

        result = process_tei_xml(xml_input, pipeline)
        choices = self._get_choices(result)
        body_s = self._body_str(result)

        # Should have exactly one choice, for magn\u0169→magnum only (single-line L1)
        self.assertEqual(len(choices), 1,
            f"Expected 1 choice (single-line L1 abbr), got {len(choices)}. Body: {body_s}")

        expan = choices[0].find(f"{{{TEI_NS}}}expan")
        expan_text = "".join(expan.itertext())
        self.assertIn("magnum", expan_text)
        # Cross-line structure must NOT be present
        self.assertIsNone(
            choices[0].find(f"{{{TEI_NS}}}abbr/{{{TEI_NS}}}lb"),
            "Single-line choice must NOT have a <lb> inside <abbr>"
        )




class TestPreExistingCrossLineChoicePreserved(unittest.TestCase):
    """
    Regression tests for pre-existing cross-line <choice> elements that were
    annotated by the rule-based system and should be left structurally intact.

    Root cause of the bug:
    When a <lb break="no"/> sits inside a pre-existing <choice>/<abbr> (e.g.
    <abbr>repe<lb/>riat̃</abbr>), the extraction code had two defects:

    1. _walk_into for <choice> called _inner_text(abbr) without respecting
       next_lb, so L1 collected the FULL abbr text ("reperiat̃") plus the
       choice's tail (" in "), when it should only have collected "repe".

    2. _collect_text_after_lb, starting from the lb inside <abbr>, climbed
       up to <abbr> and then walked into the sibling <expan>, collecting
       "reperiatur" as if it were L2 text — pure garbage.

    The garbled plain_text caused the diff to find a spurious large change
    spanning both halves plus surrounding text, which was then wrapped in a
    new nested <choice> inside the existing <abbr>.

    Fix: detect lb elements inside <choice>/<abbr> and handle them specially
    in both _walk_into (L1: collect only pre-lb text) and
    _collect_text_after_lb (L2: collect post-lb abbr content as from_choice,
    skip <expan>, then resume after <choice>).

    A secondary fix: strip LINE_SEP from exp_text before comparing it to
    existing_text in the cross-line pre-existing-choice merge, since the
    merged change carries "repe¬riatur" while existing_text is "reperiatur".
    """

    # Exact input from the bug report
    INPUT_BODY = (
        'Eccle<g ref="#char017f">\u017f</g>ia quot nominibus compellata '
        '<choice resp="#auto" xml:id="W0100-00-0019-ce-04eb">'
        '<abbr>repe<lb xml:id="W0100-00-0019-lb-2029" rendition="#hyphen" break="no"/>'
        'ria<g ref="#chart0303">t\u0303</g></abbr>'
        '<expan resp="#CR #auto">repe'
        '<lb sameAs="#W0100-00-0019-lb-2029" rendition="#hyphen" break="no"'
        ' xml:id="W0100-00-0019-lb-s704"/>riatur</expan>'
        '</choice> in <g ref="#char017f">\u017f</g>criptura '
        '<g ref="#char017f">\u017f</g>acra. pa. 53. col. 2. in prin.'
    )

    def _make_doc(self, body: str) -> str:
        return (
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            '<teiHeader><fileDesc><titleStmt><title>T</title></titleStmt>'
            '<publicationStmt><p/></publicationStmt>'
            '<sourceDesc><p/></sourceDesc></fileDesc></teiHeader>'
            '<text><body><p>'
            '<item xml:id="W0100-00-0019-it-05f1">'
            '<lb xml:id="W0100-00-0019-lb-2028"/>'
            + body +
            '</item></p></body></text></TEI>'
        )

    def _body_el(self, result: str) -> etree._Element:
        tree = etree.fromstring(result.encode())
        return tree.find(f"{{{TEI_NS}}}text/{{{TEI_NS}}}body")

    def _pipeline_agree(self, rows, pre_annotated):
        """Model agrees with the rule-based expansion (riat̃ → riatur)."""
        expanded = {}
        boundaries = {}
        for row in rows:
            lid = row["id"]
            txt = row["source_sic"]
            # Expand riat̃ → riatur, resolve glyph variants
            txt = txt.replace("riat\u0303", "riatur")
            txt = txt.replace("\u017f", "s")
            expanded[lid] = txt
        # lb-2029 is pre-annotated (break="no" in source); pipeline returns it too
        boundaries["W0100-00-0019-lb-2028"] = "W0100-00-0019-lb-2029"
        return expanded, boundaries

    def test_no_nested_choice_inside_abbr(self):
        """
        A pre-existing cross-line <choice> must never get a new nested
        <choice> inserted inside its <abbr>.
        """
        result = process_tei_xml(self._make_doc(self.INPUT_BODY), self._pipeline_agree)
        body = self._body_el(result)

        outer_choice = body.find(f".//{{{TEI_NS}}}choice[@{{http://www.w3.org/XML/1998/namespace}}id='W0100-00-0019-ce-04eb']")
        self.assertIsNotNone(outer_choice, "Original <choice> must still exist")

        abbr = outer_choice.find(f"{{{TEI_NS}}}abbr")
        self.assertIsNotNone(abbr)

        nested_choices = abbr.findall(f".//{{{TEI_NS}}}choice")
        self.assertEqual(
            len(nested_choices), 0,
            f"No nested <choice> must appear inside the existing <abbr>, "
            f"got: {etree.tostring(abbr, encoding='unicode')}"
        )

    def test_abbr_content_unchanged(self):
        """
        The <abbr> content (repe + lb + riat̃) must be preserved exactly
        — the rule-based annotation is trusted and must not be touched.
        """
        result = process_tei_xml(self._make_doc(self.INPUT_BODY), self._pipeline_agree)
        body = self._body_el(result)

        outer_choice = body.find(f".//{{{TEI_NS}}}choice[@{{http://www.w3.org/XML/1998/namespace}}id='W0100-00-0019-ce-04eb']")
        abbr = outer_choice.find(f"{{{TEI_NS}}}abbr")
        abbr_text = "".join(abbr.itertext())
        self.assertIn("repe", abbr_text, "<abbr> must still contain 'repe'")
        self.assertIn("ria", abbr_text, "<abbr> must still contain 'ria'")
        self.assertIn("t\u0303", abbr_text, "<abbr> must still contain the abbreviated 't̃'")

        lb_in_abbr = abbr.find(f"{{{TEI_NS}}}lb")
        self.assertIsNotNone(lb_in_abbr, "The <lb> must remain inside <abbr>")

    def test_model_resp_added_to_existing_expan(self):
        """
        When the model agrees with the pre-existing expansion (riatur),
        the model identifier must be appended to the existing <expan @resp>.
        No new <choice> or <expan> should be created.
        """
        result = process_tei_xml(self._make_doc(self.INPUT_BODY), self._pipeline_agree)
        body = self._body_el(result)

        outer_choice = body.find(f".//{{{TEI_NS}}}choice[@{{http://www.w3.org/XML/1998/namespace}}id='W0100-00-0019-ce-04eb']")
        expans = outer_choice.findall(f"{{{TEI_NS}}}expan")
        self.assertEqual(len(expans), 1, "Exactly one <expan> must remain")

        expan = expans[0]
        resp = expan.get("resp", "")
        self.assertIn("#CR", resp, "Original #CR resp must be preserved")
        self.assertIn("#auto", resp, "Original #auto resp must be preserved")
        self.assertIn(f"#{EXPANSION_MODEL}", resp,
                      f"Model identifier #{EXPANSION_MODEL} must be added to <expan @resp>")

        expan_text = "".join(expan.itertext())
        self.assertIn("riatur", expan_text, "<expan> must still contain 'riatur'")

    def test_main_text_after_choice_preserved(self):
        """
        Text after the </choice> (' in ſcriptura ſacra…') must not be
        swallowed into the choice or garbled.
        """
        result = process_tei_xml(self._make_doc(self.INPUT_BODY), self._pipeline_agree)
        body = self._body_el(result)
        full_text = "".join(body.itertext())

        # The XML preserves <g>ſ</g> elements, so itertext() gives the raw ſ,
        # not the model-resolved 's'. Check for stable substrings instead.
        self.assertIn("pa. 53. col. 2. in prin.", full_text,
                      "Trailing text must be preserved after </choice>")
        # ' in ' should appear between the choice and scriptura
        self.assertIn(" in ", full_text,
                      "The ' in ' text after </choice> must be preserved")


class TestCrossLineAbbrWithWhitespaceAroundLb(unittest.TestCase):
    """
    Regression tests for the ſuccedãt bug.

    In real TEI, an <lb/> is almost always flanked by layout whitespace
    (a CR/CRLF newline plus indentation) in the source, so the extracted
    plain_text of the line *before* the break ends with that whitespace —
    e.g. "…ſtatui ſucce\\n".  The boundary classifier (correctly) predicts the
    break as nonbreaking and the model strips the whitespace in its own
    output, but the roundtrip code matched change edges against the raw
    LINE_SEP position.  A single trailing newline therefore made the
    leftward extension in _merge_sep_adjacent_changes a no-op, and the
    cross-line abbreviation collapsed into an L2-only single-line <choice>
    (with the trailing period wrongly swallowed inside it).

    PR #11's Scenario B test placed L1's last word directly against the
    <lb/> with no intervening whitespace, so it never exercised this path —
    which is why it did not catch the bug.

    Target for "…ſtatui ſucce\\n" / "<lb/>" / "dãt. Item…":

        …ſtatui <choice><abbr><g>ſ</g>ucce<lb break="no" …/>d<g>ã</g>t</abbr>
                        <expan>ſucce<lb sameAs=… break="no" …/>dant</expan>
                </choice>. Item…

    i.e. ONE cross-line <choice> wrapping the whole ſucce[lb]dãt token, with
    the period left outside the <choice>.
    """

    def _make_doc(self, body: str) -> str:
        return (
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            '<teiHeader><fileDesc><titleStmt><title>T</title></titleStmt>'
            '<publicationStmt><p/></publicationStmt>'
            '<sourceDesc><p/></sourceDesc></fileDesc></teiHeader>'
            '<text><body><p>' + body + '</p></body></text></TEI>'
        )

    def _body_el(self, result: str):
        tree = etree.fromstring(result.encode())
        return tree.find(f"{{{TEI_NS}}}text/{{{TEI_NS}}}body")

    def _choices(self, result: str) -> list:
        return self._body_el(result).findall(f".//{{{TEI_NS}}}choice")

    def _assert_cross_line_succedat(self, result: str):
        """Shared assertions for the ſucce[lb]dãt → succedant target shape."""
        choices = self._choices(result)
        self.assertEqual(
            len(choices), 1,
            f"Expected exactly 1 cross-line <choice>, got {len(choices)}. "
            f"Body: {etree.tostring(self._body_el(result), encoding='unicode')}"
        )
        choice = choices[0]
        abbr = choice.find(f"{{{TEI_NS}}}abbr")
        expan = choice.find(f"{{{TEI_NS}}}expan")
        self.assertIsNotNone(abbr)
        self.assertIsNotNone(expan)

        # <abbr> must contain the <lb/> (cross-line, not L2-only)
        lb_in_abbr = abbr.find(f"{{{TEI_NS}}}lb")
        self.assertIsNotNone(
            lb_in_abbr,
            "<abbr> must contain the <lb/> element; the whole ſucce[lb]dãt "
            "token must be wrapped, not just the L2 half."
        )

        # <abbr> text must cover BOTH halves of the token with no stray
        # layout whitespace from around the <lb/>.
        abbr_text = "".join(abbr.itertext())
        self.assertEqual(
            abbr_text, "\u017fucced\u00e3t",
            f"<abbr> must read exactly 'ſuccedãt' (the full token, no stray "
            f"whitespace); got {abbr_text!r}"
        )

        # <expan> must read 'ſuccedant' and contain the <lb sameAs=…>.
        expan_text = "".join(expan.itertext())
        self.assertEqual(
            expan_text, "\u017fuccedant",
            f"<expan> must read 'ſuccedant'; got {expan_text!r}"
        )
        lb_in_expan = expan.find(f"{{{TEI_NS}}}lb")
        self.assertIsNotNone(lb_in_expan, "<expan> must contain a <lb sameAs=…>")
        self.assertEqual(lb_in_expan.get("break"), "no")

        # The surviving long-s must be wrapped as <g ref="#char017f"> inside
        # <expan>, mirroring the diplomatic markup in <abbr> — not emitted as a
        # bare U+017F character.
        expan_g = expan.findall(f"{{{TEI_NS}}}g")
        self.assertEqual(
            len(expan_g), 1,
            f"<expan> must contain exactly one <g> (the surviving long-s); "
            f"got {len(expan_g)}: {etree.tostring(expan, encoding='unicode')}"
        )
        self.assertEqual(expan_g[0].get("ref"), "#char017f")
        self.assertEqual(expan_g[0].text, "\u017f")
        # That <g> must precede the <lb/> (it is part of the L1 half).
        self.assertLess(
            list(expan).index(expan_g[0]), list(expan).index(lb_in_expan),
            "The <g>ſ</g> must come before the <lb/> in <expan>.")
        # The expanded 'dant' (L2 half) carries no glyphs.
        self.assertEqual(lb_in_expan.tail, "dant")

        # The period must be OUTSIDE the <choice> (in its tail), not inside.
        self.assertTrue(
            (choice.tail or "").lstrip().startswith("."),
            f"The period after 'dãt' must follow </choice>, not sit inside it; "
            f"choice.tail={choice.tail!r}"
        )
        self.assertNotIn(
            ".", abbr_text, "Trailing period must not be inside <abbr>")
        self.assertNotIn(
            ".", expan_text, "Trailing period must not be inside <expan>")

    def _pipeline(self, l1_id, l2_id, l1_exp, l2_exp):
        def pipeline(rows, pre_annotated):
            exp = {r["id"]: r["source_sic"] for r in rows}
            exp[l1_id] = l1_exp
            exp[l2_id] = l2_exp
            return exp, {l1_id: l2_id}
        return pipeline

    def test_newline_before_lb_two_line_chain(self):
        """
        The minimal reproduction: a trailing newline on L1 (between 'ſucce'
        and the <lb/>) must NOT prevent the cross-line <choice>.
        """
        body = (
            '<lb xml:id="W0100-00-0014-lb-2009"/>'
            'Cardinales Apo<g ref="#char017f">\u017f</g>tolorum '
            '<g ref="#char017f">\u017f</g>tatui <g ref="#char017f">\u017f</g>ucce\n'
            '<lb xml:id="W0100-00-0014-lb-2010"/>'
            'd<g ref="#chara0303">\u00e3</g>t. Item <g ref="#char0026">&amp;</g> quid'
        )
        result = process_tei_xml(
            self._make_doc(body),
            self._pipeline(
                "W0100-00-0014-lb-2009", "W0100-00-0014-lb-2010",
                "Cardinales Apo\u017ftolorum \u017ftatui \u017fucce\n",
                "dant. Item & quid",
            ),
        )
        self._assert_cross_line_succedat(result)

        # The <lb> inside <expan> must derive its xml:id from the original
        # (…-lb-2010 → …-lb-s2010) per the corpus convention.
        expan = self._choices(result)[0].find(f"{{{TEI_NS}}}expan")
        lb_in_expan = expan.find(f"{{{TEI_NS}}}lb")
        self.assertEqual(
            lb_in_expan.get("sameAs"), "#W0100-00-0014-lb-2010")
        self.assertEqual(
            lb_in_expan.get(f"{{{XML_NS}}}id"), "W0100-00-0014-lb-s2010")

    def test_real_four_line_chain_from_jsonl(self):
        """
        The bug as it actually occurs: a four-line nonbreaking chain (the
        inferred JSONL had lb-2008 → 2009 → 2010 → 2011, each continuing line
        ending with '\\n').  The ſucce[lb]dãt token sits at the 2009→2010
        junction; an unrelated single-line abbr (Apoſtolorũ→Apoſtolorum) sits
        on 2008.  Only the cross-line token should become a cross-line choice.
        """
        body = (
            '<lb xml:id="W0100-00-0014-lb-2008"/>'
            'Apo<g ref="#char017f">\u017f</g>tolor<g ref="#charu0303">\u0169</g> '
            'triplex con<g ref="#char017f">\u017f</g>ideratio. Et quomo\n'
            '<lb xml:id="W0100-00-0014-lb-2009"/>'
            'do Cardinales Apo<g ref="#char017f">\u017f</g>tolorum '
            '<g ref="#char017f">\u017f</g>tatui <g ref="#char017f">\u017f</g>ucce\n'
            '<lb xml:id="W0100-00-0014-lb-2010"/>'
            'd<g ref="#chara0303">\u00e3</g>t. Item <g ref="#char0026">&amp;</g> '
            'quid in illis eorum dignitas re\n'
            '<lb xml:id="W0100-00-0014-lb-2011"/>'
            'quirat. <g ref="#char0026">&amp;</g>c. pag. 139. col. 2. nu. 3.'
        )

        def pipeline(rows, pre_annotated):
            exp = {
                "W0100-00-0014-lb-2008":
                    "Apo\u017ftolorum triplex con\u017fideratio. Et quomo\n",
                "W0100-00-0014-lb-2009":
                    "do Cardinales Apo\u017ftolorum \u017ftatui \u017fucce\n",
                "W0100-00-0014-lb-2010":
                    "dant. Item & quid in illis eorum dignitas re\n",
                "W0100-00-0014-lb-2011":
                    "quirat. &c. pag. 139. col. 2. nu. 3.",
            }
            boundaries = {
                "W0100-00-0014-lb-2008": "W0100-00-0014-lb-2009",
                "W0100-00-0014-lb-2009": "W0100-00-0014-lb-2010",
                "W0100-00-0014-lb-2010": "W0100-00-0014-lb-2011",
            }
            return exp, boundaries

        result = process_tei_xml(self._make_doc(body), pipeline)

        choices = self._choices(result)
        cross = [c for c in choices
                 if c.find(f"{{{TEI_NS}}}abbr/{{{TEI_NS}}}lb") is not None]
        single = [c for c in choices
                  if c.find(f"{{{TEI_NS}}}abbr/{{{TEI_NS}}}lb") is None]

        # Exactly one cross-line choice (ſuccedãt) and one single-line choice
        # (Apoſtolorũ).  The 're\nquirat' continuation is NOT an abbreviation
        # (model left both halves unchanged), so it must not become a choice.
        self.assertEqual(len(cross), 1,
                         f"Expected 1 cross-line choice, got {len(cross)}")
        self.assertEqual(len(single), 1,
                         f"Expected 1 single-line choice, got {len(single)}")

        cross_abbr = "".join(cross[0].find(f"{{{TEI_NS}}}abbr").itertext())
        self.assertEqual(cross_abbr, "\u017fucced\u00e3t")
        single_abbr = "".join(single[0].find(f"{{{TEI_NS}}}abbr").itertext())
        self.assertEqual(single_abbr, "Apo\u017ftolor\u0169")

    def test_leading_whitespace_on_l2_after_lb(self):
        """
        Symmetric case: whitespace *after* the <lb/> (leading layout
        whitespace on L2) must also be handled, and must not leak into the
        <abbr> token.
        """
        body = (
            '<lb xml:id="lb-A"/>'
            'foo bar <g ref="#char017f">\u017f</g>ucce\n'
            '<lb xml:id="lb-B"/>\n    '
            'd<g ref="#chara0303">\u00e3</g>t. baz'
        )
        result = process_tei_xml(
            self._make_doc(body),
            self._pipeline("lb-A", "lb-B",
                           "foo bar \u017fucce\n", "\n    dant. baz"),
        )
        self._assert_cross_line_succedat(result)

    def test_glyph_only_l2_start_with_whitespace_no_choice(self):
        """
        A glyph-variant-only change (ſ→s) at the start of L2, with whitespace
        around the <lb/>, must NOT be promoted to a cross-line choice — the
        glyph filter must still neutralise it.
        """
        body = (
            '<lb xml:id="lb-A"/>ec\n'
            '<lb xml:id="lb-B"/>cle<g ref="#char017f">\u017f</g>ia. finis'
        )
        result = process_tei_xml(
            self._make_doc(body),
            self._pipeline("lb-A", "lb-B", "ec\n", "clesia. finis"),
        )
        self.assertEqual(
            len(self._choices(result)), 0,
            "A glyph-variant-only L2-start change must not create a <choice>.")

    def test_real_l1_abbr_plus_glyph_l2_with_whitespace_stays_single_line(self):
        """
        With whitespace around the <lb/>: if L1's last word is a real abbr
        (magnũ→magnum) but L2's first word is only a glyph variant
        (cleſia→clesia), only the single-line L1 choice should appear — no
        cross-line choice.
        """
        body = (
            '<lb xml:id="lb-A"/>magn<g ref="#charu0303">\u0169</g>\n'
            '<lb xml:id="lb-B"/>cle<g ref="#char017f">\u017f</g>ia finis'
        )
        result = process_tei_xml(
            self._make_doc(body),
            self._pipeline("lb-A", "lb-B", "magnum\n", "clesia finis"),
        )
        choices = self._choices(result)
        self.assertEqual(len(choices), 1,
                         f"Expected 1 single-line choice, got {len(choices)}")
        self.assertIsNone(
            choices[0].find(f"{{{TEI_NS}}}abbr/{{{TEI_NS}}}lb"),
            "Single-line L1 choice must not have a <lb> inside <abbr>.")


    def test_single_line_expan_also_wraps_glyphs(self):
        """
        Consistency: the single-line expansion path must also wrap surviving
        special characters in <g> inside <expan>.  Here 'ſã' (long-s glyph +
        combining-tilde glyph) expands to 'ſan': the long-s survives and must
        be emitted as <g ref="#char017f">ſ</g> in <expan>, not as a bare
        U+017F character.
        """
        body = (
            '<lb xml:id="lb-A"/>'
            'incipit <g ref="#char017f">\u017f</g>'
            '<g ref="#chara0303">\u00e3</g> finit'
        )
        # ſã -> ſan : long-s survives (glyph), ã -> an (real replace)
        def pipeline(rows, pre_annotated):
            exp = {r["id"]: r["source_sic"] for r in rows}
            exp["lb-A"] = "incipit \u017fan finit"
            return exp, {}
        result = process_tei_xml(self._make_doc(body), pipeline)
        choices = self._choices(result)
        self.assertEqual(len(choices), 1)
        expan = choices[0].find(f"{{{TEI_NS}}}expan")
        # itertext still reads the full expansion
        self.assertEqual("".join(expan.itertext()), "\u017fan")
        # and the long-s is a <g ref="#char017f">, not a bare character
        g = expan.findall(f"{{{TEI_NS}}}g")
        self.assertEqual(len(g), 1, "single-line <expan> must wrap the long-s in <g>")
        self.assertEqual(g[0].get("ref"), "#char017f")
        self.assertEqual(g[0].text, "\u017f")


if __name__ == "__main__":
    unittest.main()
