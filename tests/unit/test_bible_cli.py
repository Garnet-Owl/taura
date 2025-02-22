"""
Integration tests for Bible CLI.

This module contains integration tests for the Bible CLI utilities.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocessing.bible_cli import main


class TestBibleCLI(unittest.TestCase):
    """Test class for Bible CLI integration tests."""

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

        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Save sample data to a CSV file
        self.sample_csv = os.path.join(self.test_dir, "sample_data.csv")
        self.sample_data.to_csv(self.sample_csv, index=False)

        # Create another sample CSV for comparison tests
        self.other_data = pd.DataFrame(
            {
                "Reference": ["Kĩambĩrĩria 1:1", "Mathayo 1:1"],
                "Kikuyu": [
                    "Kĩambĩrĩria-inĩ Ngai nĩombire igũrũ na thĩ.",
                    "Ibuku rĩa rũciaro rwa Jesũ Kristũ, mũrũ wa Daudi, mũrũ wa Iburahĩmu.",
                ],
                "English": [
                    "In the beginning God created the heavens and the earth.",
                    "The book of the genealogy of Jesus Christ, the son of David, the son of Abraham.",
                ],
            }
        )
        self.other_csv = os.path.join(self.test_dir, "other_data.csv")
        self.other_data.to_csv(self.other_csv, index=False)

        # Output paths for tests
        self.output_json = os.path.join(self.test_dir, "output.json")
        self.output_csv = os.path.join(self.test_dir, "output.csv")

    def tearDown(self):
        """Clean up temporary files."""
        # Remove any created files
        for file_path in [self.sample_csv, self.other_csv, self.output_json, self.output_csv]:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Remove the temporary directory
        os.rmdir(self.test_dir)

    @patch("sys.argv")
    @patch("src.data.preprocessing.bible_cli.generate_bible_structure")
    def test_generate_structure(self, mock_generate, mock_argv):
        """Test the generate-structure command."""
        # Mock command-line arguments
        mock_argv.__getitem__.side_effect = lambda i: ["bible_cli.py", "generate-structure"][i]

        # Mock the generate_bible_structure function
        mock_generate.return_value = True

        # Run the main function
        result = main()

        # Verify the function was called
        mock_generate.assert_called_once()

        # Verify the result
        self.assertEqual(result, 0)

    @patch("sys.argv")
    def test_analyze(self, mock_argv):
        """Test the analyze command."""
        # Mock command-line arguments
        mock_argv.__getitem__.side_effect = lambda i: [
            "bible_cli.py",
            "analyze",
            "--input",
            self.sample_csv,
            "--output",
            self.output_json,
        ][i]

        # Run the main function
        result = main()

        # Verify the result
        self.assertEqual(result, 0)

        # Verify output file was created
        self.assertTrue(os.path.exists(self.output_json))

    @patch("sys.argv")
    def test_compare(self, mock_argv):
        """Test the compare command."""
        # Mock command-line arguments
        mock_argv.__getitem__.side_effect = lambda i: [
            "bible_cli.py",
            "compare",
            "--source",
            self.sample_csv,
            "--target",
            self.other_csv,
            "--output",
            self.output_csv,
        ][i]

        # Run the main function
        result = main()

        # Verify the result
        self.assertEqual(result, 0)

        # Verify output file was created
        self.assertTrue(os.path.exists(self.output_csv))

        # Verify content of output file
        output_df = pd.read_csv(self.output_csv)
        self.assertEqual(len(output_df), 3)  # 3 rows should be in the difference

    @patch("sys.argv")
    @patch("builtins.print")
    def test_missing_books(self, mock_print, mock_argv):
        """Test the missing-books command."""
        # Mock command-line arguments
        mock_argv.__getitem__.side_effect = lambda i: ["bible_cli.py", "missing-books", "--input", self.sample_csv][i]

        # Run the main function
        result = main()

        # Verify the result
        self.assertEqual(result, 0)

    @patch("sys.argv")
    def test_stats_by_book(self, mock_argv):
        """Test the stats-by-book command."""
        # Mock command-line arguments
        mock_argv.__getitem__.side_effect = lambda i: [
            "bible_cli.py",
            "stats-by-book",
            "--input",
            self.sample_csv,
            "--output",
            self.output_csv,
        ][i]

        # Run the main function
        result = main()

        # Verify the result
        self.assertEqual(result, 0)

        # Verify output file was created
        self.assertTrue(os.path.exists(self.output_csv))


if __name__ == "__main__":
    unittest.main()
