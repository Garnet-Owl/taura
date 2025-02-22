"""
This script explores the Kikuyu-English dataset to understand its structure and contents.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Set path to the data file
DATA_DIR = Path("../../data")
EXCEL_FILE = DATA_DIR / "English20Kikuyu20Pairs2029.xlsx"


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


def analyze_dataset(df):
    """
    Analyze the dataset and print statistics.
    """
    if df is None:
        return

    print("\n===== Dataset Analysis =====")
    print(f"Number of entries: {len(df)}")

    # Display column names
    print(f"\nColumns in the dataset: {df.columns.tolist()}")

    # Basic statistics
    print("\nSample data:")
    print(df.head())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values)

    # Analyze text length
    if "English" in df.columns and "Kikuyu" in df.columns:
        df["english_length"] = df["English"].astype(str).apply(len)
        df["kikuyu_length"] = df["Kikuyu"].astype(str).apply(len)

        print("\nText length statistics (characters):")
        print("English text length:")
        print(df["english_length"].describe())
        print("\nKikuyu text length:")
        print(df["kikuyu_length"].describe())

        # Visualize length distribution
        plt.figure(figsize=(10, 6))
        plt.scatter(df["english_length"], df["kikuyu_length"], alpha=0.5)
        plt.xlabel("English Text Length")
        plt.ylabel("Kikuyu Text Length")
        plt.title("Text Length Comparison: English vs Kikuyu")
        plt.savefig(DATA_DIR / "text_length_comparison.png")
        print("\nText length comparison plot saved to data directory.")


if __name__ == "__main__":
    df = load_dataset()
    analyze_dataset(df)
