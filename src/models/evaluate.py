import argparse
import json
import logging
from pathlib import Path

import torch
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from train import TranslationDataset, load_examples, collate_fn
from translation_model import KikuyuEnglishTranslator

# Set paths
DATA_DIR = Path("../../data")
TOKENIZED_DIR = DATA_DIR / "tokenized"
MODELS_DIR = Path("../../models")
RESULTS_DIR = Path("../../results")

# Create results directory
RESULTS_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(RESULTS_DIR / "evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def decode_batch(tokenizer, batch):
    """
    Decode batch of tokenized sequences.

    Args:
        tokenizer: The tokenizer
        batch: Batch of tokenized sequences

    Returns:
        list: List of decoded texts
    """
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch]


def calculate_bleu(references, hypotheses):
    """
    Calculate BLEU score.

    Args:
        references (list): List of reference translations
        hypotheses (list): List of model translations

    Returns:
        float: BLEU score
    """
    # For BLEU calculation, we need a list of references for each hypothesis
    references_list = [[ref] for ref in references]
    return corpus_bleu(hypotheses, references_list).score


def evaluate_model(
    model_path=None, tokenizer_name="Helsinki-NLP/opus-mt-en-mul", batch_size=16, device=None, test_set="test"
):
    """
    Evaluate the trained model.

    Args:
        model_path (str): Path to the saved model
        tokenizer_name (str): Name or path of the tokenizer
        batch_size (int): Batch size for evaluation
        device (str): Device to use (cuda/cpu)
        test_set (str): Name of the test set to use

    Returns:
        dict: Evaluation metrics
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Evaluating on {device}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Tokenizer: {tokenizer_name}")
    logger.info(f"Test set: {test_set}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"Tokenizer loaded: {tokenizer_name}")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return None

    # Load model
    translator = KikuyuEnglishTranslator()
    if model_path:
        translator.load_model(Path(model_path))
    else:
        # Try to load best model by default
        best_model_path = MODELS_DIR / "best_model"
        if best_model_path.exists():
            translator.load_model(best_model_path)
        else:
            logger.error("Model path not provided and best_model not found")
            return None

    model = translator.model

    # Load test examples
    test_examples = load_examples(test_set)

    if not test_examples:
        logger.error(f"Failed to load {test_set} examples. Please run tokenize.py first.")
        return None

    # Create dataset and dataloader
    test_dataset = TranslationDataset(test_examples)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Evaluation
    model.eval()
    all_input_ids = []
    all_labels = []
    all_predictions = []
    total_loss = 0

    logger.info("Starting evaluation...")

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass for loss
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            # Generate translations
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
                num_beams=5,
                early_stopping=True,
            )

            # Store for later analysis
            all_input_ids.append(batch["input_ids"].cpu())
            all_labels.append(batch["labels"].cpu())
            all_predictions.append(generated_ids.cpu())

    # Calculate average loss
    avg_loss = total_loss / len(test_dataloader)
    logger.info(f"Test Loss: {avg_loss:.4f}")

    # Decode all sequences
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    source_texts = decode_batch(tokenizer, all_input_ids)
    reference_texts = decode_batch(tokenizer, all_labels)
    prediction_texts = decode_batch(tokenizer, all_predictions)

    # Calculate BLEU score
    bleu_score = calculate_bleu(reference_texts, prediction_texts)
    logger.info(f"BLEU Score: {bleu_score:.2f}")

    # Save results
    results = {
        "test_loss": avg_loss,
        "bleu_score": bleu_score,
        "model_path": str(model_path) if model_path else str(MODELS_DIR / "best_model"),
        "tokenizer": tokenizer_name,
        "test_set": test_set,
        "batch_size": batch_size,
    }

    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save example translations
    examples = []
    for i in range(min(50, len(source_texts))):
        examples.append({"source": source_texts[i], "reference": reference_texts[i], "prediction": prediction_texts[i]})

    with open(RESULTS_DIR / "translation_examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    logger.info(f"Evaluation results saved to {RESULTS_DIR}")

    return results


def main():
    """
    Main function to parse arguments and evaluate the model.
    """
    parser = argparse.ArgumentParser(description="Evaluate a Kikuyu-English translation model")

    parser.add_argument("--model", type=str, default=None, help="Path to the saved model")
    parser.add_argument("--tokenizer", type=str, default="Helsinki-NLP/opus-mt-en-mul", help="Tokenizer to use")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--test-set", type=str, default="test", help="Name of the test set to use")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Convert string device to torch.device if provided
    device = torch.device(args.device) if args.device else None

    evaluate_model(
        model_path=args.model,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        device=device,
        test_set=args.test_set,
    )


if __name__ == "__main__":
    main()
