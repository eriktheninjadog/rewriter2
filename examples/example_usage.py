"""
Example usage of the Chinese Text Simplification pipeline.

Run from the project root:

    python examples/example_usage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the examples/ directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from src.pipeline_controller import SimplificationPipeline


def main() -> None:
    # Load configuration
    config_path = ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Override vocab path to be relative to project root
    vocab_file = str(ROOT / config["system"]["vocabulary_file"])

    # Initialise pipeline
    pipeline = SimplificationPipeline(vocab_file=vocab_file, config=config)

    # Sample texts (Traditional Chinese)
    sample_texts = [
        "美麗的風景令人心曠神怡，我們決定在此駐足欣賞。",
        "今天天氣很好，我和朋友一起去公園散步。",
        "他非常聰明，學習成績一直名列前茅。",
    ]

    for text in sample_texts:
        print("=" * 60)
        result = pipeline.simplify_text(text)
        print(f"Original:    {result['original']}")
        print(f"Simplified:  {result['simplified']}")
        if result["replacements_made"]:
            print("Replacements:")
            for orig, repl in result["replacements_made"].items():
                print(f"  {orig} → {repl}")
        print(f"Stats:       {result['stats']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
