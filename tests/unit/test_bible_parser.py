"""
Tests for the Bible parser module.
"""

import unittest

from src.data.preprocessing.bible_parser import (
    preprocess_bible_text,
    parse_kikuyu_verse,
    parse_english_verse,
    align_verses,
)
from src.data.preprocessing.structure.bible_structure import get_missing_verses, get_extra_verses, is_valid_reference


class BibleParserTests(unittest.TestCase):
    """Test cases for Bible parser module."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample Kikuyu text from Genesis 1
        self.kikuyu_sample = """
        Kĩambĩrĩria 1:1
        1
        KĨAMBĨRĨRIA
        Kĩambĩrĩria 1:12
        1 Kĩambĩrĩria-inĩ kĩa maũndũ mothe, Ngai nĩombire
        igũrũ na thĩ. 2Hĩndĩ ĩyo thĩ ndĩarĩ ũrĩa yatariĩ na ndĩarĩ
        nakĩndũ, nayo ndumayarĩigũrũrĩa kũrĩa kũriku, nake
        Roho waNgaiaareerete igũrũ rĩa maaĩ.
        3 Nake Ngai agĩathana, akiuga atĩrĩ, "Nĩkũgĩe ũtheri,"
        na gũkĩgĩa ũtheri. 4Ngai akĩona atĩ ũtheri ũcio warĩ
        mwega, nake akĩgayũkania ũtheri na nduma. 5Ngai
        agĩĩta ũtheri ũcio "mũthenya," nayo ndumaakĩmĩĩta
        "ũtukũ." Na gũkĩgĩa hwaĩ-inĩ, na gũkĩgĩa rũciinĩ.
        Ũcio ũgĩtuĩka mũthenya wa mbere.
        """

        # Sample English text from Genesis 1
        self.english_sample = """
        Genesis 1:1
        1
        Genesis 1:30
        Genesis
        The Creation of the World
        1 In the beginning God created the heavens and the earth.
        2 Now the earth was without shape and empty, and darkness was over the surface of the watery
        deep, but the Spirit of God was moving over the surface of the water. 3God said, "Let there be light."
        Andtherewaslight! 4Godsawthatthelightwasgood, soGodseparatedthelightfromthedarkness.
        5 God called the light "day" and the darkness "night." There was evening, and there was morning,
        marking the first day.
        """

    def test_preprocess_bible_text(self):
        """Test preprocessing Bible text."""
        # Test Kikuyu preprocessing
        processed_kikuyu = preprocess_bible_text(self.kikuyu_sample, "kikuyu")
        self.assertIn("Kĩambĩrĩria 1:1", processed_kikuyu)
        self.assertIn("1 Kĩambĩrĩria-inĩ kĩa maũndũ mothe", processed_kikuyu)

        # Check that inline verse numbers are properly separated
        self.assertIn("2 Hĩndĩ", processed_kikuyu)
        self.assertNotIn("2Hĩndĩ", processed_kikuyu)

        # Test English preprocessing
        processed_english = preprocess_bible_text(self.english_sample, "english")
        self.assertIn("Genesis 1:1", processed_english)
        self.assertIn("1 In the beginning God created", processed_english)

        # Check that inline verse numbers are properly separated
        self.assertIn("3 God", processed_english)
        self.assertNotIn("3God", processed_english)

    def test_parse_kikuyu_verse(self):
        """Test parsing Kikuyu Bible verses."""
        parsed = parse_kikuyu_verse(self.kikuyu_sample)

        # Check if Genesis is parsed correctly
        self.assertIn("Kĩambĩrĩria", parsed)

        # Check if chapter 1 is present
        self.assertIn(1, parsed.get("Kĩambĩrĩria", {}))

        # Check if verse 1 is parsed correctly
        verse_1 = parsed.get("Kĩambĩrĩria", {}).get(1, {}).get(1, "")
        self.assertIn("Kĩambĩrĩria-inĩ kĩa maũndũ mothe", verse_1)

        # Check for embedded verse numbers
        verse_2 = parsed.get("Kĩambĩrĩria", {}).get(1, {}).get(2, "")
        self.assertTrue(verse_2)  # Verse 2 should exist
        self.assertIn("Hĩndĩ ĩyo thĩ ndĩarĩ ũrĩa yatariĩ", verse_2)

        # Validate using our structure
        missing = get_missing_verses({"Genesis": parsed.get("Kĩambĩrĩria", {})}, "Genesis")
        self.assertLessEqual(len(missing), 1528)  # We expect most verses to be missing in this small sample

        extras = get_extra_verses({"Genesis": parsed.get("Kĩambĩrĩria", {})}, "Genesis")
        self.assertEqual(len(extras), 0)  # There should be no extra verses

    def test_parse_english_verse(self):
        """Test parsing English Bible verses."""
        parsed = parse_english_verse(self.english_sample)

        # Check if Genesis is parsed correctly
        self.assertIn("Genesis", parsed)

        # Check if chapter 1 is present
        self.assertIn(1, parsed.get("Genesis", {}))

        # Check if verse 1 is parsed correctly
        verse_1 = parsed.get("Genesis", {}).get(1, {}).get(1, "")
        self.assertIn("In the beginning God created", verse_1)

        # Check for embedded verse numbers
        verse_3 = parsed.get("Genesis", {}).get(1, {}).get(3, "")
        self.assertTrue(verse_3)  # Verse 3 should exist
        self.assertIn("God said", verse_3)

        # Validate using our structure
        missing = get_missing_verses(parsed, "Genesis")
        self.assertLessEqual(len(missing), 1528)  # We expect most verses to be missing in this small sample

        extras = get_extra_verses(parsed, "Genesis")
        self.assertEqual(len(extras), 0)  # There should be no extra verses

    def test_align_verses(self):
        """Test aligning verses between languages."""
        kikuyu_dict = parse_kikuyu_verse(self.kikuyu_sample)
        english_dict = parse_english_verse(self.english_sample)

        aligned = align_verses(kikuyu_dict, english_dict)

        # We should have at least some aligned verses
        self.assertTrue(len(aligned) > 0)

        # Check the format of aligned verses
        for reference, kikuyu_text, english_text in aligned:
            self.assertTrue(reference)  # Reference should not be empty
            self.assertTrue(kikuyu_text)  # Kikuyu text should not be empty
            self.assertTrue(english_text)  # English text should not be empty

            # Extract book, chapter, verse from reference
            parts = reference.split()
            book = parts[0]
            chapter_verse = parts[1].split(":")
            chapter = int(chapter_verse[0])
            verse = int(chapter_verse[1])

            # Validate the reference
            self.assertEqual(book, "Kĩambĩrĩria")
            self.assertTrue(1 <= chapter <= 50)
            self.assertTrue(is_valid_reference("Genesis", chapter, verse))


if __name__ == "__main__":
    unittest.main()
