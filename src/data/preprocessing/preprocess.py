"""
This script handles preprocessing of the Kikuyu-English dataset.
"""

import re
from pathlib import Path

import pandas as pd
import unicodedata

# Set paths
DATA_DIR = Path("../../../data")
EXCEL_FILE = DATA_DIR / "English20Kikuyu20Pairs2029.xlsx"
PROCESSED_DIR = DATA_DIR / "processed"

# Create processed directory if it doesn't exist
PROCESSED_DIR.mkdir(exist_ok=True)


def load_dataset():
    """
    Load the Kikuyu-English dataset from Excel file.
    """
    try:
        df = pd.read_excel(EXCEL_FILE)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def clean_text(text):
    """
    Clean and normalize text.

    Args:
        text (str): The input text to clean

    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def preprocess_data(df):
    """
    Preprocess the dataset.

    Args:
        df (DataFrame): Pandas DataFrame containing the data

    Returns:
        DataFrame: Preprocessed DataFrame
    """
    if df is None:
        return None

    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Rename columns if needed based on actual column names
    if "English" in processed_df.columns and "Kikuyu" in processed_df.columns:
        # Apply text cleaning
        processed_df["English_cleaned"] = processed_df["English"].apply(clean_text)
        processed_df["Kikuyu_cleaned"] = processed_df["Kikuyu"].apply(clean_text)

        # Remove empty entries
        processed_df = processed_df[(processed_df["English_cleaned"] != "") & (processed_df["Kikuyu_cleaned"] != "")]

        print(f"After cleaning, dataset shape: {processed_df.shape}")
    else:
        # Handle case where column names are different
        print(f"Warning: Expected columns 'English' and 'Kikuyu' not found. Columns: {df.columns.tolist()}")

    return processed_df


def split_data(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split the dataset into train, validation, and test sets.

    Args:
        df (DataFrame): Preprocessed DataFrame
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if df is None:
        return None, None, None

    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"

    # Shuffle the DataFrame
    shuffled_df = df.sample(frac=1, random_state=random_state)

    # Calculate split indices
    train_end = int(len(shuffled_df) * train_ratio)
    val_end = train_end + int(len(shuffled_df) * val_ratio)

    # Split the data
    train_df = shuffled_df.iloc[:train_end]
    val_df = shuffled_df.iloc[train_end:val_end]
    test_df = shuffled_df.iloc[val_end:]

    print(f"Data split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df):
    """
    Save the data splits to CSV files.

    Args:
        train_df (DataFrame): Training data
        val_df (DataFrame): Validation data
        test_df (DataFrame): Test data
    """
    if train_df is None or val_df is None or test_df is None:
        print("Error: Cannot save empty datasets")
        return

    # Save to CSV
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"Data splits saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    # Load data
    df = load_dataset()

    # Preprocess data
    processed_df = preprocess_data(df)

    # Split data
    train_df, val_df, test_df = split_data(processed_df)

    # Save splits
    save_splits(train_df, val_df, test_df)
