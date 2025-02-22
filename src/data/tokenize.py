"""
This script handles tokenization for the Kikuyu-English translation model.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

# Set paths
DATA_DIR = Path("../../data")
PROCESSED_DIR = DATA_DIR / "processed"
TOKENIZED_DIR = DATA_DIR / "tokenized"
MODELS_DIR = Path("../../models")

# Create necessary directories
TOKENIZED_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Default tokenizer model (can be changed to any other suitable model)
DEFAULT_TOKENIZER = "Helsinki-NLP/opus-mt-en-mul"


class TranslationTokenizer:
    """
    Handles tokenization for Kikuyu-English translation.
    """

    def __init__(self, tokenizer_name=DEFAULT_TOKENIZER):
        """
        Initialize the tokenizer.

        Args:
            tokenizer_name (str): Name or path of the tokenizer to use
        """
        try:
            print(f"Loading tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

        # Save tokenizer configuration
        self.config = {
            "tokenizer_name": tokenizer_name,
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
            "max_length": 128,  # Default max length, can be adjusted
        }

    def encode_pair(self, english_text, kikuyu_text, max_length=None):
        """
        Encode a pair of English and Kikuyu texts.

        Args:
            english_text (str): English text
            kikuyu_text (str): Kikuyu text
            max_length (int, optional): Maximum sequence length

        Returns:
            dict: Dictionary containing the encoded inputs
        """
        if max_length is None:
            max_length = self.config["max_length"]

        # Tokenize source (English)
        english_encoding = self.tokenizer(
            english_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )

        # Tokenize target (Kikuyu)
        kikuyu_encoding = self.tokenizer(
            kikuyu_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )

        return {
            "input_ids": english_encoding.input_ids.squeeze(),
            "attention_mask": english_encoding.attention_mask.squeeze(),
            "labels": kikuyu_encoding.input_ids.squeeze(),
        }

    def tokenize_dataset(self, df, src_col="English_cleaned", tgt_col="Kikuyu_cleaned", max_length=None):
        """
        Tokenize a dataset.

        Args:
            df (DataFrame): DataFrame containing the data
            src_col (str): Column name for source language
            tgt_col (str): Column name for target language
            max_length (int, optional): Maximum sequence length

        Returns:
            list: List of encoded examples
        """
        if max_length is None:
            max_length = self.config["max_length"]

        examples = []
        for _, row in df.iterrows():
            src_text = row[src_col]
            tgt_text = row[tgt_col]

            # Skip empty texts
            if not src_text or not tgt_text:
                continue

            encoded = self.encode_pair(src_text, tgt_text, max_length)
            examples.append(encoded)

        print(f"Tokenized {len(examples)} examples")
        return examples

    def save_config(self, path=None):
        """
        Save tokenizer configuration.

        Args:
            path (Path, optional): Path to save the config
        """
        if path is None:
            path = MODELS_DIR / "tokenizer_config.json"

        with open(path, "w") as f:
            json.dump(self.config, f, indent=2)

        print(f"Tokenizer configuration saved to {path}")

    @staticmethod
    def save_examples(examples, split_name, path=None):
        """
        Save tokenized examples.

        Args:
            examples (list): List of tokenized examples
            split_name (str): Name of the split (train, val, test)
            path (Path, optional): Directory to save the examples
        """
        if path is None:
            path = TOKENIZED_DIR

        output_path = path / f"{split_name}.pt"
        torch.save(examples, output_path)
        print(f"Saved {len(examples)} tokenized examples to {output_path}")


def process_all_splits():
    """
    Process and tokenize all data splits.
    """
    # Check if processed files exist
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        print("Processed data files not found. Please run preprocess.py first.")
        return

    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded data splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Initialize tokenizer
    tokenizer = TranslationTokenizer()

    # Save tokenizer configuration
    tokenizer.save_config()

    # Tokenize each split
    train_examples = tokenizer.tokenize_dataset(train_df)
    val_examples = tokenizer.tokenize_dataset(val_df)
    test_examples = tokenizer.tokenize_dataset(test_df)

    # Save tokenized examples
    tokenizer.save_examples(train_examples, "train")
    tokenizer.save_examples(val_examples, "val")
    tokenizer.save_examples(test_examples, "test")

    print("All data splits tokenized and saved successfully.")


if __name__ == "__main__":
    process_all_splits()
