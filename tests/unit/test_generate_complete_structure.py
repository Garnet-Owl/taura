"""
Tests for Bible structure generation utilities.

This module contains unit tests for the Bible structure generation utilities.
"""

import unittest
from unittest.mock import patch, mock_open
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.preprocessing.structure.generate_complete_structure import generate_bible_structure, KIKUYU_NAMES


class TestGenerateCompleteStructure(unittest.TestCase):
    """Test class for Bible structure generation utilities."""

    def setUp(self):
        """Set up test data."""
        # Create a sample bible_books.json content
        self.sample_json = [
            {
                "book_name": "Genesis",
                "num_chapters": 2,
                "chapters": [{"chapter_no": 1, "num_verses": 31}, {"chapter_no": 2, "num_verses": 25}],
            },
            {"book_name": "Exodus", "num_chapters": 1, "chapters": [{"chapter_no": 1, "num_verses": 22}]},
        ]

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    @patch("pathlib.Path")
    def test_generate_bible_structure(self, mock_path, mock_json_load, mock_file):
        """Test generating Bible structure from JSON."""
        # Set up mocks
        mock_json_load.return_value = self.sample_json
        mock_path.return_value.parent = "mock_dir"

        # Run function
        result = generate_bible_structure()

        # Verify the function succeeded
        self.assertTrue(result)

        # Verify file was opened for writing
        mock_file.assert_called()

        # Get the content written to the file
        file_content = mock_file().write.call_args[0][0]

        # Verify the content contains expected patterns
        self.assertIn("BIBLE_BOOKS = [", file_content)
        self.assertIn('"book_name": "Genesis"', file_content)
        self.assertIn('"kikuyu_name": "Kĩambĩrĩria"', file_content)
        self.assertIn('"book_name": "Exodus"', file_content)
        self.assertIn('"kikuyu_name": "Woima"', file_content)
        self.assertIn("def get_book_by_name", file_content)
        self.assertIn("def kikuyu_to_english_book_name", file_content)

    def test_kikuyu_names_completeness(self):
        """Test that all 66 Bible books have Kikuyu names."""
        # Set of expected book names from full Bible
        expected_books = {
            "Genesis",
            "Exodus",
            "Leviticus",
            "Numbers",
            "Deuteronomy",
            "Joshua",
            "Judges",
            "Ruth",
            "1 Samuel",
            "2 Samuel",
            "1 Kings",
            "2 Kings",
            "1 Chronicles",
            "2 Chronicles",
            "Ezra",
            "Nehemiah",
            "Esther",
            "Job",
            "Psalms",
            "Proverbs",
            "Ecclesiastes",
            "Song of Solomon",
            "Isaiah",
            "Jeremiah",
            "Lamentations",
            "Ezekiel",
            "Daniel",
            "Hosea",
            "Joel",
            "Amos",
            "Obadiah",
            "Jonah",
            "Micah",
            "Nahum",
            "Habakkuk",
            "Zephaniah",
            "Haggai",
            "Zechariah",
            "Malachi",
            "Matthew",
            "Mark",
            "Luke",
            "John",
            "Acts",
            "Romans",
            "1 Corinthians",
            "2 Corinthians",
            "Galatians",
            "Ephesians",
            "Philippians",
            "Colossians",
            "1 Thessalonians",
            "2 Thessalonians",
            "1 Timothy",
            "2 Timothy",
            "Titus",
            "Philemon",
            "Hebrews",
            "James",
            "1 Peter",
            "2 Peter",
            "1 John",
            "2 John",
            "3 John",
            "Jude",
            "Revelation",
        }

        # Get the set of book names from KIKUYU_NAMES
        provided_books = set(KIKUYU_NAMES.keys())

        # Verify all expected books are present
        missing_books = expected_books - provided_books
        self.assertEqual(len(missing_books), 0, f"Missing Kikuyu names for: {missing_books}")

        # Verify no extra books are present
        extra_books = provided_books - expected_books
        self.assertEqual(len(extra_books), 0, f"Unexpected books: {extra_books}")


if __name__ == "__main__":
    unittest.main()
