"""
Wrapper script for backward compatibility.
Redirects to the preprocessing module's main script.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to enable imports from the new location
sys.path.append(str(Path(__file__).parent))

# Import the main function from the preprocessing module
from src.data.preprocessing.main import main  # This import is now safe since we fixed the circular dependency

if __name__ == "__main__":
    # Create an args object with default values
    class Args:
        excel = True
        bible = False
        save_bible = False
        max_examples = None
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

    args = Args()

    # Call the main function from the new location
    main(args)

    print("\nNote: This script is now a wrapper for backward compatibility.")
    print("For more options, use: python -m src.data.preprocessing.main --help")
