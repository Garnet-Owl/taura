import argparse
from pathlib import Path

from transformers import AutoTokenizer

from translation_model import KikuyuEnglishTranslator

# Set paths
MODELS_DIR = Path("../../models")


def load_translator(model_path=None, tokenizer_name=None):
    """
    Load the translator model and tokenizer.

    Args:
        model_path (str): Path to the saved model
        tokenizer_name (str): Name or path of the tokenizer

    Returns:
        tuple: (translator, tokenizer)
    """
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
            # Try final model
            final_model_path = MODELS_DIR / "final_model"
            if final_model_path.exists():
                translator.load_model(final_model_path)
            else:
                raise FileNotFoundError("No trained model found. Please train a model first.")

    # Load tokenizer
    if tokenizer_name is None:
        # Try to get from config
        config_path = MODELS_DIR / "tokenizer_config.json"
        if config_path.exists():
            import json

            with open(config_path, "r") as f:
                config = json.load(f)
            tokenizer_name = config.get("tokenizer_name", "Helsinki-NLP/opus-mt-en-mul")
        else:
            tokenizer_name = "Helsinki-NLP/opus-mt-en-mul"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return translator, tokenizer


def translate_text(text, translator, tokenizer, max_length=128, direction="en-to-kik"):
    """
    Translate a piece of text.

    Args:
        text (str): Text to translate
        translator: Translator model
        tokenizer: Tokenizer
        max_length (int): Maximum sequence length
        direction (str): Translation direction ('en-to-kik' or 'kik-to-en')

    Returns:
        str: Translated text
    """
    # For now, the model is trained for English to Kikuyu only
    if direction == "kik-to-en":
        raise NotImplementedError("Kikuyu to English translation not yet implemented")

    return translator.translate(text, tokenizer, max_length)


def translate_file(input_file, output_file, translator, tokenizer, max_length=128, direction="en-to-kik"):
    """
    Translate all lines in a file.

    Args:
        input_file (str): Path to input file
        output_file (str): Path to output file
        translator: Translator model
        tokenizer: Tokenizer
        max_length (int): Maximum sequence length
        direction (str): Translation direction ('en-to-kik' or 'kik-to-en')
    """
    with open(input_file, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    translated_lines = []
    for line in lines:
        line = line.strip()
        if line:
            translation = translate_text(line, translator, tokenizer, max_length, direction)
            translated_lines.append(translation)
        else:
            translated_lines.append("")

    with open(output_file, "w", encoding="utf-8") as f_out:
        for line in translated_lines:
            f_out.write(line + "\n")


def main():
    """
    Main function to parse arguments and perform translation.
    """
    parser = argparse.ArgumentParser(description="Translate text using the Kikuyu-English model")

    parser.add_argument("--text", type=str, default=None, help="Text to translate")
    parser.add_argument(
        "--input-file", type=str, default=None, help="Input file containing text to translate (one per line)"
    )
    parser.add_argument("--output-file", type=str, default="translations.txt", help="Output file for translated text")
    parser.add_argument("--model", type=str, default=None, help="Path to the saved model")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer to use")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument(
        "--direction", type=str, choices=["en-to-kik", "kik-to-en"], default="en-to-kik", help="Translation direction"
    )

    args = parser.parse_args()

    # Load translator and tokenizer
    translator, tokenizer = load_translator(args.model, args.tokenizer)

    # Perform translation
    if args.text:
        # Translate provided text
        translation = translate_text(args.text, translator, tokenizer, args.max_length, args.direction)
        print(f"Original: {args.text}")
        print(f"Translation: {translation}")

    elif args.input_file:
        # Translate file
        translate_file(args.input_file, args.output_file, translator, tokenizer, args.max_length, args.direction)
        print(f"Translations saved to {args.output_file}")

    else:
        # Interactive mode
        print("Enter text to translate (press Ctrl+C to exit):")
        try:
            while True:
                text = input(">>> ")
                if text.strip():
                    translation = translate_text(text, translator, tokenizer, args.max_length, args.direction)
                    print(f"Translation: {translation}")
        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()
