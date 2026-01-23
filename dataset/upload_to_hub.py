#!/usr/bin/env python3
"""
Upload the fitness coach dataset to HuggingFace Hub.
"""

import sys

from datasets import load_from_disk
from huggingface_hub import whoami


# Configuration
DATASET_PATH = "dataset/fitness_coach_function_calling"
REPO_NAME = "tengomucho/fitness-coach-function-calling"


def check_authentication():
    """Check if user is authenticated with HuggingFace."""
    try:
        user_info = whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return True
    except Exception:
        print("✗ Not logged in to HuggingFace")
        print("\nTo login, run:")
        print("  huggingface-cli login")
        print("\nOr set your token:")
        print("  export HUGGING_FACE_HUB_TOKEN=your_token_here")
        return False


def upload_dataset():
    """Upload the dataset to HuggingFace Hub."""
    print("=" * 60)
    print("UPLOADING DATASET TO HUGGINGFACE HUB")
    print("=" * 60)

    # Check authentication
    if not check_authentication():
        sys.exit(1)

    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}...")
    try:
        dataset = load_from_disk(DATASET_PATH)
        print("✓ Dataset loaded successfully")
        print(f"  - Total examples: {len(dataset)}")
        print(f"  - Features: {list(dataset.features.keys())}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        sys.exit(1)

    # Upload to Hub
    print(f"\nUploading dataset to {REPO_NAME}...")
    print("This may take a few minutes...")

    try:
        dataset.push_to_hub(REPO_NAME, private=False)
        print("\n✓ Dataset uploaded successfully!")
        print("\nView your dataset at:")
        print(f"  https://huggingface.co/datasets/{REPO_NAME}")
    except Exception as e:
        print(f"\n✗ Failed to upload dataset: {e}")
        print("\nIf you see an authentication error, try:")
        print("  1. Get a token from https://huggingface.co/settings/tokens")
        print("  2. Run: huggingface-cli login")
        print("  3. Or set: export HUGGING_FACE_HUB_TOKEN=your_token_here")
        sys.exit(1)


if __name__ == "__main__":
    upload_dataset()
