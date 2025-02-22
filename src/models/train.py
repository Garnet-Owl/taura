import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from translation_model import KikuyuEnglishTranslator

# Set paths
DATA_DIR = Path("../../data")
TOKENIZED_DIR = DATA_DIR / "tokenized"
MODELS_DIR = Path("../../models")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(MODELS_DIR / "training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TranslationDataset(Dataset):
    """
    Dataset for Kikuyu-English translation.
    """

    def __init__(self, examples):
        """
        Initialize the dataset.

        Args:
            examples (list): List of tokenized examples
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Args:
        batch (list): Batch of examples

    Returns:
        dict: Batched tensors
    """
    input_ids = torch.stack([example["input_ids"] for example in batch])
    attention_mask = torch.stack([example["attention_mask"] for example in batch])
    labels = torch.stack([example["labels"] for example in batch])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_examples(split_name):
    """
    Load tokenized examples.

    Args:
        split_name (str): Name of the split (train, val, test)

    Returns:
        list: List of tokenized examples
    """
    path = TOKENIZED_DIR / f"{split_name}.pt"
    if not path.exists():
        logger.error(f"File not found: {path}")
        return []

    examples = torch.load(path)
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


def train_epoch(model, dataloader, optimizer, scheduler, device, clip_grad=1.0):
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use (cuda/cpu)
        clip_grad: Gradient clipping value

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # Update weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Update total loss
        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluate the model.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to use (cuda/cpu)

    Returns:
        float: Average loss for the evaluation
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Update total loss
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    model_name="Helsinki-NLP/opus-mt-en-mul",
    batch_size=16,
    num_epochs=10,
    learning_rate=5e-5,
    warmup_steps=0,
    from_scratch=False,
    device=None,
):
    """
    Train the translation model.

    Args:
        model_name (str): Name or path of the model
        batch_size (int): Batch size
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        warmup_steps (int): Warmup steps for scheduler
        from_scratch (bool): Whether to train from scratch
        device (str): Device to use (cuda/cpu)

    Returns:
        KikuyuEnglishTranslator: Trained model
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Training on {device}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Training from scratch: {from_scratch}")

    # Load examples
    train_examples = load_examples("train")
    val_examples = load_examples("val")

    if not train_examples or not val_examples:
        logger.error("Failed to load examples. Please run tokenize.py first.")
        return None

    # Create datasets and dataloaders
    train_dataset = TranslationDataset(train_examples)
    val_dataset = TranslationDataset(val_examples)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    translator = KikuyuEnglishTranslator(model_name, from_scratch)
    model = translator.model

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        train_losses.append(train_loss)

        # Evaluate
        val_loss = evaluate(model, val_dataloader, device)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start_time

        logger.info(
            f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            translator.save_model(MODELS_DIR / "best_model")
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")

    # Save final model
    translator.save_model(MODELS_DIR / "final_model")
    logger.info("Final model saved")

    # Save training history
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val_loss,
        "training_time": total_time,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_name": model_name,
        "from_scratch": from_scratch,
    }

    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training history saved")

    return translator


def main():
    """
    Main function to parse arguments and train the model.
    """
    parser = argparse.ArgumentParser(description="Train a Kikuyu-English translation model")

    parser.add_argument(
        "--model", type=str, default="Helsinki-NLP/opus-mt-en-mul", help="Pretrained model to fine-tune"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--from-scratch", action="store_true", help="Train model from scratch instead of fine-tuning")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Convert string device to torch.device if provided
    device = torch.device(args.device) if args.device else None

    train_model(
        model_name=args.model,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        from_scratch=args.from_scratch,
        device=device,
    )


if __name__ == "__main__":
    main()
