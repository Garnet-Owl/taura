"""
Main preprocessing script for the Taura project.

This script coordinates all preprocessing tasks, including:
1. Processing the Excel dataset
2. Processing Bible PDF files
3. Combining datasets
4. Splitting into train/val/test sets
"""

import argparse
from pathlib import Path

import pandas as pd
from preprocess import load_dataset as load_excel_dataset, clean_text, preprocess_data, split_data, save_splits
from bible_parser import process_bible_texts, save_bible_dataset

# Set paths
DATA_DIR = Path("../../../data")
PROCESSED_DIR = DATA_DIR / "processed"

# Create processed directory if it doesn't exist
PROCESSED_DIR.mkdir(exist_ok=True)


def combine_datasets(excel_df, bible_df):
    """
    Combine datasets from different sources.

    Args:
        excel_df: DataFrame from Excel source
        bible_df: DataFrame from Bible PDF source

    Returns:
        Combined DataFrame
    """
    # Ensure both DataFrames have the same structure
    if excel_df is not None and "English_cleaned" in excel_df.columns and "Kikuyu_cleaned" in excel_df.columns:
        excel_df = excel_df[["English_cleaned", "Kikuyu_cleaned"]]
        excel_df.columns = ["English", "Kikuyu"]

    if bible_df is not None:
        # Clean Bible texts
        bible_df["English_cleaned"] = bible_df["English"].apply(clean_text)
        bible_df["Kikuyu_cleaned"] = bible_df["Kikuyu"].apply(clean_text)
        bible_df = bible_df[["English_cleaned", "Kikuyu_cleaned"]]
        bible_df.columns = ["English", "Kikuyu"]

    # Combine the datasets
    if excel_df is not None and bible_df is not None:
        combined_df = pd.concat([excel_df, bible_df], ignore_index=True)
    elif excel_df is not None:
        combined_df = excel_df
    elif bible_df is not None:
        combined_df = bible_df
    else:
        return None

    # Remove duplicates
    combined_df = combined_df.copy()
    combined_df.drop_duplicates(inplace=True)

    return combined_df


def main(args):
    """
    Main function to run all preprocessing steps.

    Args:
        args: Command-line arguments
    """
    # Process Excel dataset if specified
    excel_df = None
    if args.excel:
        print("Processing Excel dataset...")
        raw_df = load_excel_dataset()
        excel_df = preprocess_data(raw_df)

    # Process Bible PDFs if specified
    bible_df = None
    if args.bible:
        print("Processing Bible PDFs...")
        kikuyu_pdf = DATA_DIR / args.kikuyu_pdf
        english_pdf = DATA_DIR / args.english_pdf

        # Verify PDF files exist
        if not kikuyu_pdf.exists():
            print(f"Warning: Kikuyu Bible PDF not found at {kikuyu_pdf}")
            print("Please provide the correct filename using --kikuyu-pdf")
            return

        if not english_pdf.exists():
            print(f"Warning: English Bible PDF not found at {english_pdf}")
            print("Please provide the correct filename using --english-pdf")
            return

        bible_df = process_bible_texts(str(kikuyu_pdf), str(english_pdf), max_examples=args.max_examples)

        # Save the Bible dataset separately
        if args.save_bible:
            save_bible_dataset(bible_df)

    # Combine datasets if both are available
    combined_df = combine_datasets(excel_df, bible_df)

    if combined_df is not None:
        print(f"Combined dataset shape: {combined_df.shape}")

        # Split data
        train_df, val_df, test_df = split_data(
            combined_df, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio
        )

        # Save splits
        save_splits(train_df, val_df, test_df)
    else:
        print("No data to process. Please specify at least one data source.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taura dataset preprocessing")

    # Data source options
    parser.add_argument("--excel", action="store_true", help="Process Excel dataset")
    parser.add_argument("--bible", action="store_true", help="Process Bible PDFs")

    # Bible specific options
    parser.add_argument("--save-bible", action="store_true", help="Save Bible dataset separately")
    parser.add_argument("--max-examples", type=int, default=None, help="Maximum number of examples from Bible")
    parser.add_argument(
        "--kikuyu-pdf",
        type=str,
        default="kikuyu_bible_all_testments.pdf",
        help="Filename of Kikuyu Bible PDF in data directory",
    )
    parser.add_argument(
        "--english-pdf",
        type=str,
        default="english_net_bible_all_testaments.pdf",
        help="Filename of English Bible PDF in data directory",
    )

    # Data split options
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of training data")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of test data")

    args = parser.parse_args()

    # If no sources specified, use Excel by default
    if not args.excel and not args.bible:
        args.excel = True

    main(args)
