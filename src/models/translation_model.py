import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    MarianMTModel,
    MarianConfig,
    BartForConditionalGeneration,
    BartConfig,
    T5ForConditionalGeneration,
    T5Config,
    AutoTokenizer,
)
from pathlib import Path
import json

MODELS_DIR = Path("../../models")


def get_pretrained_model(model_name, from_scratch=False):
    """
    Get a pretrained or new model for translation.

    Args:
        model_name (str): Name or path of the model
        from_scratch (bool): Whether to initialize a new model

    Returns:
        nn.Module: The translation model
    """
    if from_scratch:
        if "marian" in model_name.lower():
            config = MarianConfig(
                vocab_size=50000,
                d_model=512,
                encoder_layers=6,
                decoder_layers=6,
                encoder_attention_heads=8,
                decoder_attention_heads=8,
                decoder_ffn_dim=2048,
                encoder_ffn_dim=2048,
            )
            model = MarianMTModel(config)
        elif "bart" in model_name.lower():
            config = BartConfig(
                vocab_size=50000,
                d_model=768,
                encoder_layers=6,
                decoder_layers=6,
                encoder_attention_heads=12,
                decoder_attention_heads=12,
            )
            model = BartForConditionalGeneration(config)
        elif "t5" in model_name.lower():
            config = T5Config(
                vocab_size=32000,
                d_model=512,
                d_ff=2048,
                num_layers=6,
                num_heads=8,
            )
            model = T5ForConditionalGeneration(config)
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")
    else:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    return model


class KikuyuEnglishTranslator:
    """
    Wrapper class for the Kikuyu-English translation model.
    """

    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-mul", from_scratch=False):
        """
        Initialize the translator.

        Args:
            model_name (str): Name or path of the model
            from_scratch (bool): Whether to initialize a new model
        """
        self.model_name = model_name
        self.model = get_pretrained_model(model_name, from_scratch)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Model initialized and moved to {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def save_model(self, path=None):
        """
        Save the model.

        Args:
            path (Path, optional): Path to save the model
        """
        if path is None:
            path = MODELS_DIR / "translator"

        path.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(path)

        # Save model configuration for easy loading
        config = {
            "model_name": self.model_name,
            "path": str(path),
        }

        with open(MODELS_DIR / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {path}")

    def load_model(self, path=None):
        """
        Load a saved model.

        Args:
            path (Path, optional): Path to load the model from
        """
        if path is None:
            # Try to load from config
            config_path = MODELS_DIR / "model_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                path = Path(config["path"])
            else:
                path = MODELS_DIR / "translator"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path} and moved to {self.device}")

    def translate(self, text, tokenizer, max_length=128):
        """
        Translate text.

        Args:
            text (str): Text to translate
            tokenizer: Tokenizer to use
            max_length (int): Maximum generated sequence length

        Returns:
            str: Translated text
        """
        self.model.eval()

        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True)

        # Decode output
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translation


def get_model_size(model):
    """
    Calculate model size in MB.

    Args:
        model: PyTorch model

    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_in_mb = (param_size + buffer_size) / 1024**2
    return size_in_mb


if __name__ == "__main__":
    # Example usage
    local_model_path = "../../models/best_model"  # Replace with your actual path
    translator = KikuyuEnglishTranslator(model_name=local_model_path)  # load local model
    print(f"Model size: {get_model_size(translator.model):.2f} MB")

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-mul")  # load same tokenizer as when training
    translation = translator.translate("Hello, how are you?", tokenizer)
    print(f"Translation: {translation}")
