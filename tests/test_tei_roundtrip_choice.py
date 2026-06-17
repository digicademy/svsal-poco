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


if __name__ == "__main__":
    unittest.main()
