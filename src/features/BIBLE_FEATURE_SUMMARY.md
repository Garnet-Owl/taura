# Taura Project - Bible Translation Features

This document summarizes the Bible translation features implemented for the Taura Kikuyu-English translation project.

## Overview

The Taura project now includes comprehensive support for working with Bible translations between Kikuyu and English. These features enable the project to:

1. Represent the complete structure of the Bible (all 66 books) with proper Kikuyu names
2. Parse and validate Bible references in both languages
3. Extract parallel texts from Bible PDFs for training data
4. Analyze and compare translation datasets
5. Command-line tools for working with Bible data

## Implementation Details

### 1. Complete Bible Structure

- All 66 books of the Bible from Genesis to Revelation
- Mapping between English and Kikuyu book names
- Complete chapter and verse structure for validation
- Utilities for reference validation and conversion

### 2. Bible Parsing Utilities

- PDF text extraction with PyMuPDF
- Kikuyu and English text preprocessing
- Reference parsing and normalization
- Verse alignment between languages

### 3. Analysis and Comparison Tools

- Translation statistics (length, word count ratios)
- Coverage analysis by book
- Missing books and verses identification
- Dataset comparison utilities

### 4. Command-Line Interface

- Generate complete Bible structure
- Analyze translation datasets
- Compare multiple datasets
- Find missing books and verses
- Generate statistics by book

## Usage Examples

### Generating Training Data from Bible PDFs

```python
from src.data.preprocessing.bible_parser import process_bible_texts

df = process_bible_texts(
    "kikuyu_bible.pdf",
    "english_bible.pdf"
)
df.to_csv("bible_parallel.csv", index=False)
```

### Analyzing Translation Data

```python
from src.data.preprocessing.bible_utils import load_translation_data, compare_translations

df = load_translation_data("bible_parallel.csv")
stats = compare_translations(df)
print(f"Total verse pairs: {stats['total_verse_pairs']}")
print(f"Book coverage: {stats['book_coverage_percent']}%")
```

### Using the Command-Line Interface

```bash
# Generate statistics by book
python -m src.data.preprocessing.bible_cli stats-by-book --input data/bible_parallel.csv --output data/book_stats.csv

# Find missing books
python -m src.data.preprocessing.bible_cli missing-books --input data/bible_parallel.csv
```

## Testing

The Bible features are thoroughly tested with unit and integration tests:

- `test_bible_structure.py` - Tests for Bible structure utilities
- `test_bible_parser.py` - Tests for Bible parsing utilities
- `test_bible_utils.py` - Tests for Bible utility functions
- `test_bible_cli.py` - Integration tests for CLI functions
- `test_generate_complete_structure.py` - Tests for structure generation

## Next Steps

Future enhancements to the Bible translation features could include:

1. Improved parsing accuracy for difficult text formats
2. Support for additional Bible translations
3. Automatic error correction in parsed texts
4. Interactive visualization of translation coverage
5. Integration with model training pipelines for religious text specialization
