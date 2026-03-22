# rewriter2 – Chinese Text Simplification System

A pipeline system that rewrites Traditional Chinese text using only a limited
10,000-word vocabulary, designed specifically for Chinese language learners
reading novels.

## System Overview

```
Traditional Chinese Text
        ↓
Jieba POS Segmentation  (incl. Traditional→Simplified conversion)
        ↓
OOV Detection Engine    (identify words outside the 10k vocabulary)
        ↓
Synonym Selection       (find in-vocabulary replacements)
        ↓
LLM Rewriting Engine    (Qwen-7B-Chat or rule-based fallback)
        ↓
Simplified Chinese Text (using only 10k vocabulary words)
```

## File Structure

```
chinese_simplifier/
├── README.md
├── requirements.txt
├── config.yaml
├── vocabulary_10000.txt
├── src/
│   ├── __init__.py
│   ├── vocabulary_manager.py   # Manages the 10k word list
│   ├── text_segmenter.py       # Jieba segmentation + OpenCC conversion
│   ├── oov_detector.py         # Identifies OOV tokens
│   ├── synonym_finder.py       # Finds in-vocabulary replacements
│   ├── llm_rewriter.py         # LLM / rule-based rewriting
│   └── pipeline_controller.py  # Orchestrates the full pipeline
├── templates/
│   ├── rewrite_prompt.txt      # LLM prompt template (rewriting)
│   └── validate_prompt.txt     # LLM prompt template (validation)
├── tests/
│   └── test_pipeline.py
└── examples/
    └── example_usage.py
```

## Installation

```bash
pip install -r requirements.txt
```

Core runtime dependencies:

| Package | Purpose |
|---|---|
| `jieba` | Chinese word segmentation & POS tagging |
| `opencc-python-reimplemented` | Traditional ↔ Simplified conversion |
| `synonyms` | Chinese synonym lookup |
| `transformers` / `torch` | LLM inference (Qwen-7B-Chat) |
| `pyyaml` | Configuration loading |

## Quick Start

```python
import yaml
from src.pipeline_controller import SimplificationPipeline

with open("config.yaml", encoding="utf-8") as fh:
    config = yaml.safe_load(fh)

pipeline = SimplificationPipeline(
    vocab_file="vocabulary_10000.txt",
    config=config,
)

result = pipeline.simplify_text("美麗的風景令人心曠神怡，我們決定在此駐足欣賞。")

print("Original:   ", result["original"])
print("Simplified: ", result["simplified"])
print("Replacements:", result["replacements_made"])
print("Stats:      ", result["stats"])
```

Or run the bundled example:

```bash
python examples/example_usage.py
```

## Running Tests

```bash
pip install pytest
pytest tests/
```

## Configuration

Edit `config.yaml` to adjust:

* **`system.vocabulary_file`** – path to your 10k vocabulary list.
* **`system.model`** – Hugging Face model ID (default `Qwen/Qwen-7B-Chat`).
* **`oov_detection.*`** – rules for what to keep vs. replace.
* **`synonym_selection.similarity_threshold`** – minimum similarity score for
  a synonym to be accepted.
* **`llm.max_retries`** – how many times to retry LLM generation.

## Vocabulary File Format

`vocabulary_10000.txt` contains one Simplified Chinese word per line.
Lines beginning with `#` are comments and are ignored.

## LLM Fallback

When `transformers`/`torch` are not installed (e.g. in CI or lightweight
deployments), the rewriter automatically falls back to a direct
string-substitution approach that applies the replacement map without using a
language model.

