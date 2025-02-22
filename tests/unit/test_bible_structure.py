"""
Tests for the Bible structure module.
"""

import unittest
from src.data.preprocessing.structure.bible_structure import (
    get_book_by_name,
    get_book_by_kikuyu_name,
    get_verse_count,
    is_valid_reference,
    get_verse_id,
    get_all_books,
    get_all_kikuyu_books,
    english_to_kikuyu_book_name,
    kikuyu_to_english_book_name,
    get_missing_verses,
    get_extra_verses,
)


class BibleStructureTests(unittest.TestCase):
    """Test cases for Bible structure module."""

    def test_get_book_by_name(self):
        """Test getting a book by its English name."""
        book = get_book_by_name("Genesis")
        self.assertIsNotNone(book)
        self.assertEqual(book["book_name"], "Genesis")
        self.assertEqual(book["num_chapters"], 50)

        # Test case-insensitive lookup
        book = get_book_by_name("genesis")
        self.assertIsNotNone(book)
        self.assertEqual(book["book_name"], "Genesis")

        # Test non-existent book
        book = get_book_by_name("NonExistentBook")
        self.assertIsNone(book)

    def test_get_book_by_kikuyu_name(self):
        """Test getting a book by its Kikuyu name."""
        book = get_book_by_kikuyu_name("Kĩambĩrĩria")
        self.assertIsNotNone(book)
        self.assertEqual(book["book_name"], "Genesis")
        self.assertEqual(book["kikuyu_name"], "Kĩambĩrĩria")

        # Test case-insensitive lookup
        book = get_book_by_kikuyu_name("kĩambĩrĩria")
        self.assertIsNotNone(book)
        self.assertEqual(book["book_name"], "Genesis")

        # Test non-existent book
        book = get_book_by_kikuyu_name("NonExistentBook")
        self.assertIsNone(book)

    def test_get_verse_count(self):
        """Test getting the number of verses in a chapter."""
        # Genesis chapter 1 has 31 verses
        verse_count = get_verse_count("Genesis", 1)
        self.assertEqual(verse_count, 31)

        # Genesis chapter 2 has 25 verses
        verse_count = get_verse_count("Genesis", 2)
        self.assertEqual(verse_count, 25)

        # Test non-existent chapter
        verse_count = get_verse_count("Genesis", 100)
        self.assertIsNone(verse_count)

        # Test non-existent book
        verse_count = get_verse_count("NonExistentBook", 1)
        self.assertIsNone(verse_count)

    def test_is_valid_reference(self):
        """Test validating Bible references."""
        # Valid references
        self.assertTrue(is_valid_reference("Genesis", 1, 1))
        self.assertTrue(is_valid_reference("Genesis", 1, 31))
        self.assertTrue(is_valid_reference("Genesis", 50, 26))

        # Invalid chapter
        self.assertFalse(is_valid_reference("Genesis", 51, 1))

        # Invalid verse
        self.assertFalse(is_valid_reference("Genesis", 1, 32))

        # Invalid book
        self.assertFalse(is_valid_reference("NonExistentBook", 1, 1))

    def test_get_verse_id(self):
        """Test generating verse IDs."""
        verse_id = get_verse_id("Genesis", 1, 1)
        self.assertEqual(verse_id, "Genesis.1.1")

        verse_id = get_verse_id("Genesis", 50, 26)
        self.assertEqual(verse_id, "Genesis.50.26")

    def test_get_all_books(self):
        """Test getting all book names."""
        books = get_all_books()
        self.assertIsInstance(books, list)
        self.assertIn("Genesis", books)

    def test_get_all_kikuyu_books(self):
        """Test getting all Kikuyu book names."""
        books = get_all_kikuyu_books()
        self.assertIsInstance(books, list)
        self.assertIn("Kĩambĩrĩria", books)

    def test_english_to_kikuyu_book_name(self):
        """Test converting English book names to Kikuyu."""
        kikuyu_name = english_to_kikuyu_book_name("Genesis")
        self.assertEqual(kikuyu_name, "Kĩambĩrĩria")

        # Test case-insensitive lookup
        kikuyu_name = english_to_kikuyu_book_name("genesis")
        self.assertEqual(kikuyu_name, "Kĩambĩrĩria")

        # Test non-existent book
        kikuyu_name = english_to_kikuyu_book_name("NonExistentBook")
        self.assertIsNone(kikuyu_name)

    def test_kikuyu_to_english_book_name(self):
        """Test converting Kikuyu book names to English."""
        english_name = kikuyu_to_english_book_name("Kĩambĩrĩria")
        self.assertEqual(english_name, "Genesis")

        # Test non-existent book
        english_name = kikuyu_to_english_book_name("NonExistentBook")
        self.assertIsNone(english_name)

    def test_get_missing_verses(self):
        """Test identifying missing verses."""
        # Create a sample of parsed verses with some missing
        parsed_verses = {
            "Genesis": {
                1: {
                    1: "In the beginning...",
                    # Verse 2 is missing
                    3: "And God said...",
                },
                # Chapter 2 is missing entirely
                3: {1: "Now the serpent...", 2: "The woman said..."},
            }
        }

        missing = get_missing_verses(parsed_verses, "Genesis")
        self.assertIn((1, 2), missing)  # Verse 1:2 is missing
        self.assertIn((2, 1), missing)  # All of chapter 2 is missing

        # Check at least one from chapter 3
        self.assertIn((3, 3), missing)  # Verse 3:3 is missing

    def test_get_extra_verses(self):
        """Test identifying extra verses."""
        # Create a sample of parsed verses with some extras
        parsed_verses = {
            "Genesis": {
                1: {
                    1: "In the beginning...",
                    2: "The earth was...",
                    32: "This verse doesn't exist",  # Genesis 1 only has 31 verses
                },
                2: {1: "Thus the heavens...", 2: "By the seventh day..."},
                51: {  # Genesis only has 50 chapters
                    1: "This chapter doesn't exist"
                },
            }
        }

        extras = get_extra_verses(parsed_verses, "Genesis")
        self.assertIn((1, 32), extras)  # Verse 1:32 doesn't exist
        self.assertIn((51, 1), extras)  # Chapter 51 doesn't exist


if __name__ == "__main__":
    unittest.main()
