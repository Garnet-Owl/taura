"""
Script to generate the full Bible structure module.

This script is a one-time utility to generate a complete bible_structure.py
using all 66 books from the bible_books.json file.
"""

from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.preprocessing.structure.generate_complete_structure import generate_bible_structure


def main():
    """Generate the complete Bible structure module."""
    print("Generating complete Bible structure module...")
    success = generate_bible_structure()

    if success:
        print("Successfully generated Bible structure with all 66 books.")
        print("New structure saved to:")
        print(Path(__file__).parent / "structure" / "bible_structure.py")
    else:
        print("Failed to generate Bible structure.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
