"""
Tests for Bible utilities module.

This module contains unit tests for the Bible utilities.
"""

import unittest
from unittest.mock import patch

import pandas as pd
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocessing.bible_utils import (
    parse_reference,
    get_stats_by_book,
    compare_translations,
    get_missing_books,
)


class TestBibleUtils(unittest.TestCase):
    """Test class for Bible utilities."""

    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame(
            {
                "Reference": ["Kĩambĩrĩria 1:1", "Kĩambĩrĩria 1:2", "Woima 20:3", "Thaburi 119:1"],
                "Kikuyu": [
                    "Kĩambĩrĩria-inĩ Ngai nĩombire igũrũ na thĩ.",
                    "Nayo thĩ yarĩ ĩtarĩ kĩndũ, na ĩtarĩ na mũhianĩre; nakuo nduma yarĩ igũrũ rĩa kũrĩa kũriku.",
                    "Ndũkaagĩe na ngai ingĩ o na ĩmwe mbere yakwa.",
                    "Kũrathimwo-nĩ arĩa matarĩ na ũcuuke njĩra-inĩ yao.",
                ],
                "English": [
                    "In the beginning God created the heavens and the earth.",
                    "Now the earth was formless and empty, darkness was over the surface of the deep.",
                    "You shall have no other gods before me.",
                    "Blessed are those whose ways are blameless.",
                ],
            }
        )

    def test_parse_reference(self):
        """Test parsing Bible references."""
        # Test a simple reference
        book, chapter, verse = parse_reference("Kĩambĩrĩria 1:1")
        self.assertEqual(book, "Kĩambĩrĩria")
        self.assertEqual(chapter, 1)
        self.assertEqual(verse, 1)

        # Test a reference with a multi-word book name
        book, chapter, verse = parse_reference("Rwĩmbo rwa Solomoni 1:5")
        self.assertEqual(book, "Rwĩmbo rwa Solomoni")
        self.assertEqual(chapter, 1)
        self.assertEqual(verse, 5)

        # Test a reference with a number in the book name
        book, chapter, verse = parse_reference("1 Samũeli 15:22")
        self.assertEqual(book, "1 Samũeli")
        self.assertEqual(chapter, 15)
        self.assertEqual(verse, 22)

    @patch("src.data.preprocessing.bible_utils.kikuyu_to_english_book_name")
    @patch("src.data.preprocessing.bible_utils.get_book_by_name")
    def test_get_stats_by_book(self, mock_get_book, mock_kikuyu_to_english):
        """Test getting statistics by book."""
        # Set up mocks
        mock_kikuyu_to_english.side_effect = lambda name: {
            "Kĩambĩrĩria": "Genesis",
            "Woima": "Exodus",
            "Thaburi": "Psalms",
        }.get(name)

        mock_get_book.side_effect = lambda name: {
            "Genesis": {
                "book_name": "Genesis",
                "kikuyu_name": "Kĩambĩrĩria",
                "chapters": [{"chapter_no": 1, "num_verses": 31}, {"chapter_no": 2, "num_verses": 25}],
            },
            "Exodus": {
                "book_name": "Exodus",
                "kikuyu_name": "Woima",
                "chapters": [{"chapter_no": 20, "num_verses": 17}],
            },
            "Psalms": {
                "book_name": "Psalms",
                "kikuyu_name": "Thaburi",
                "chapters": [{"chapter_no": 119, "num_verses": 176}],
            },
        }.get(name)

        # Run function
        result = get_stats_by_book(self.sample_data)

        # Verify results
        self.assertEqual(len(result), 3)  # Three books
        self.assertIn("kikuyu_name", result.columns)
        self.assertIn("english_name", result.columns)
        self.assertIn("verses_available", result.columns)
        self.assertIn("total_verses", result.columns)
        self.assertIn("coverage_percent", result.columns)

    @patch("src.data.preprocessing.bible_utils.get_stats_by_book")
    def test_compare_translations(self, mock_get_stats):
        """Test comparing translations."""
        # Set up mock
        mock_get_stats.return_value = pd.DataFrame(
            {
                "kikuyu_name": ["Kĩambĩrĩria", "Woima", "Thaburi"],
                "english_name": ["Genesis", "Exodus", "Psalms"],
                "verses_available": [2, 1, 1],
                "total_verses": [1533, 1213, 2461],
                "coverage_percent": [0.13, 0.08, 0.04],
            }
        )

        # Run function
        result = compare_translations(self.sample_data)

        # Verify results
        self.assertEqual(result["total_verse_pairs"], 4)
        self.assertIn("avg_kikuyu_len", result)
        self.assertIn("avg_english_len", result)
        self.assertIn("avg_len_ratio", result)
        self.assertIn("avg_kikuyu_words", result)
        self.assertIn("avg_english_words", result)
        self.assertIn("avg_word_ratio", result)
        self.assertIn("book_coverage", result)

    @patch("src.data.preprocessing.bible_utils.kikuyu_to_english_book_name")
    @patch("src.data.preprocessing.bible_utils.BIBLE_BOOKS")
    def test_get_missing_books(self, mock_bible_books, mock_kikuyu_to_english):
        """Test getting missing books."""
        # Set up mocks
        mock_kikuyu_to_english.side_effect = lambda name: {
            "Kĩambĩrĩria": "Genesis",
            "Woima": "Exodus",
            "Thaburi": "Psalms",
        }.get(name)

        mock_bible_books.__iter__.return_value = [
            {"book_name": "Genesis", "kikuyu_name": "Kĩambĩrĩria"},
            {"book_name": "Exodus", "kikuyu_name": "Woima"},
            {"book_name": "Leviticus", "kikuyu_name": "Alawii"},
            {"book_name": "Numbers", "kikuyu_name": "Ndari"},
            {"book_name": "Psalms", "kikuyu_name": "Thaburi"},
        ]

        # Run function
        result = get_missing_books(self.sample_data)

        # Verify results
        self.assertEqual(len(result), 2)
        self.assertIn("Leviticus", result)
        self.assertIn("Numbers", result)


if __name__ == "__main__":
    unittest.main()
