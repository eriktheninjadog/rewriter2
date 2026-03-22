"""
Unit and integration tests for the Chinese Text Simplification pipeline.

These tests are designed to run without requiring GPU / LLM / optional
heavy dependencies (synonyms, opencc, etc.).  The LLM rewriter is exercised
via its rule-based fallback path so that CI can pass on any machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make sure the src package is importable when running tests directly
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.vocabulary_manager import VocabularyManager
from src.oov_detector import OOVDetector
from src.synonym_finder import SynonymFinder
from src.llm_rewriter import LLMRewriter
from src.pipeline_controller import SimplificationPipeline


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VOCAB_FILE = ROOT / "vocabulary_10000.txt"
CONFIG_FILE = ROOT / "config.yaml"


def _minimal_config() -> Dict:
    return {
        "system": {
            "vocabulary_file": str(VOCAB_FILE),
            "model": "Qwen/Qwen-7B-Chat",
            "temperature": 0.7,
            "max_tokens": 200,
        },
        "oov_detection": {
            "keep_numbers": True,
            "keep_dates": True,
            "keep_proper_nouns": True,
            "keep_function_words": True,
            "min_word_length_to_replace": 2,
        },
        "synonym_selection": {
            "similarity_threshold": 0.1,  # Low threshold for test predictability
            "max_candidates_per_word": 5,
            "use_multiple_sources": False,
        },
        "llm": {
            "rewrite_prompt_template": str(ROOT / "templates" / "rewrite_prompt.txt"),
            "validation_prompt_template": str(
                ROOT / "templates" / "validate_prompt.txt"
            ),
            "max_retries": 1,
        },
    }


# ---------------------------------------------------------------------------
# VocabularyManager tests
# ---------------------------------------------------------------------------


class TestVocabularyManager:
    def test_load_vocabulary_file_exists(self):
        vm = VocabularyManager(str(VOCAB_FILE))
        assert len(vm) > 0, "Vocabulary should not be empty"

    def test_contains_known_word(self):
        vm = VocabularyManager(str(VOCAB_FILE))
        # "的" is the most common Chinese particle and should be in vocabulary
        assert vm.contains("的")

    def test_does_not_contain_unknown_word(self):
        vm = VocabularyManager(str(VOCAB_FILE))
        assert not vm.contains("XXXXXXXX_NOT_A_WORD")

    def test_load_vocabulary_missing_file(self, tmp_path):
        vm = VocabularyManager(str(tmp_path / "nonexistent.txt"))
        assert len(vm) == 0

    def test_get_similar_words_returns_list(self):
        vm = VocabularyManager(str(VOCAB_FILE))
        result = vm.get_similar_words("美丽", limit=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_get_similar_words_empty_query(self):
        vm = VocabularyManager(str(VOCAB_FILE))
        assert vm.get_similar_words("") == []

    def test_add_word(self):
        vm = VocabularyManager(str(VOCAB_FILE))
        vm.add_word("测试新词")
        assert vm.contains("测试新词")

    def test_dunder_contains(self):
        vm = VocabularyManager(str(VOCAB_FILE))
        assert "的" in vm

    def test_comments_and_blank_lines_skipped(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("# comment\n\n词语\n", encoding="utf-8")
        vm = VocabularyManager(str(vocab_file))
        assert vm.contains("词语")
        assert not vm.contains("# comment")


# ---------------------------------------------------------------------------
# OOVDetector tests
# ---------------------------------------------------------------------------


class TestOOVDetector:
    def setup_method(self):
        self.vm = VocabularyManager(str(VOCAB_FILE))
        self.config = _minimal_config()
        self.detector = OOVDetector(self.vm, self.config)

    def test_detect_oov_returns_list(self):
        pairs = [("美丽", "a"), ("的", "uj")]
        result = self.detector.detect_oov_words(pairs)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_in_vocab_word_kept(self):
        # "的" is in vocabulary
        pairs = [("的", "uj")]
        result = self.detector.detect_oov_words(pairs)
        assert result[0]["action"] == "keep"

    def test_number_kept(self):
        pairs = [("2024", "m")]
        result = self.detector.detect_oov_words(pairs)
        assert result[0]["action"] == "keep"
        assert "number" in result[0]["reason"]

    def test_proper_noun_kept(self):
        pairs = [("北京", "ns")]
        result = self.detector.detect_oov_words(pairs)
        assert result[0]["action"] == "keep"
        assert "proper_noun" in result[0]["reason"]

    def test_function_word_kept(self):
        pairs = [("的", "uj")]
        result = self.detector.detect_oov_words(pairs)
        assert result[0]["action"] in ("keep", "special")

    def test_short_word_kept(self):
        # Single character words below min_word_length_to_replace should be kept
        pairs = [("了", "ul")]
        result = self.detector.detect_oov_words(pairs)
        assert result[0]["action"] == "keep"

    def test_oov_word_flagged_for_replacement(self):
        # Add a word definitely not in vocab
        pairs = [("瞬間移動術語X", "v")]
        result = self.detector.detect_oov_words(pairs)
        # Should be flagged as replace (it's long, Chinese, not in vocab)
        assert result[0]["action"] == "replace"

    def test_should_replace_returns_tuple(self):
        flag, reason = self.detector.should_replace("美丽", "a", {})
        assert isinstance(flag, bool)
        assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# SynonymFinder tests
# ---------------------------------------------------------------------------


class TestSynonymFinder:
    def setup_method(self):
        self.vm = VocabularyManager(str(VOCAB_FILE))
        self.finder = SynonymFinder(self.vm, similarity_threshold=0.0)

    def test_find_replacement_returns_none_or_string(self):
        result = self.finder.find_replacement("美丽", "a", {})
        assert result is None or isinstance(result, str)

    def test_find_replacement_caches_result(self):
        # Call twice – second call should use cache
        r1 = self.finder.find_replacement("美丽", "a", {})
        r2 = self.finder.find_replacement("美丽", "a", {})
        assert r1 == r2

    def test_get_candidates_returns_list(self):
        candidates = self.finder.get_candidates("美丽", "a")
        assert isinstance(candidates, list)

    def test_rank_candidates_sorted_desc(self):
        candidates = ["好看", "美好", "漂亮"]
        ranked = self.finder.rank_candidates("美丽", candidates, {})
        if len(ranked) > 1:
            assert ranked[0][1] >= ranked[1][1]

    def test_replacement_in_vocabulary(self):
        result = self.finder.find_replacement("瞬間移動術語X", "v", {})
        # If a replacement is found, it must be in vocabulary
        if result is not None:
            assert self.vm.contains(result)


# ---------------------------------------------------------------------------
# LLMRewriter tests
# ---------------------------------------------------------------------------


class TestLLMRewriter:
    def setup_method(self):
        self.rewriter = LLMRewriter(
            model_name="Qwen/Qwen-7B-Chat",
            prompt_template_path=str(ROOT / "templates" / "rewrite_prompt.txt"),
            validation_template_path=str(ROOT / "templates" / "validate_prompt.txt"),
            max_retries=1,
        )

    def test_build_prompt_contains_original_text(self):
        text = "今天天氣很好"
        prompt = self.rewriter.build_prompt(text, {"天氣": "天色"}, [])
        assert text in prompt

    def test_build_prompt_contains_replacements(self):
        text = "今天天氣很好"
        replacements = {"天氣": "天色"}
        prompt = self.rewriter.build_prompt(text, replacements, [])
        assert "天氣" in prompt
        assert "天色" in prompt

    def test_rewrite_no_replacements_returns_original(self):
        text = "今天天氣很好"
        result = self.rewriter.rewrite_text(text, {}, [])
        assert result == text

    def test_rewrite_fallback_applies_replacements(self):
        """Without LLM, fallback should apply replacements directly."""
        text = "今天天氣很好"
        replacements = {"天氣": "天色"}
        # Simulate LLM unavailable by patching _call_llm to raise
        with patch.object(self.rewriter, "_call_llm", side_effect=RuntimeError("no LLM")):
            result = self.rewriter.rewrite_text(text, replacements, [])
        assert "天色" in result

    def test_validate_output_strips_whitespace(self):
        assert self.rewriter.validate_output("  hello  ") == "hello"

    def test_validate_output_none_on_empty(self):
        assert self.rewriter.validate_output("") is None
        assert self.rewriter.validate_output(None) is None

    def test_apply_replacements_directly(self):
        result = LLMRewriter._apply_replacements_directly(
            "我喜歡美麗的風景", {"美麗": "好看"}
        )
        assert "好看" in result
        assert "美麗" not in result

    def test_apply_replacements_longest_first(self):
        """Longer strings should be replaced before shorter sub-strings."""
        result = LLMRewriter._apply_replacements_directly(
            "今天天氣很好", {"天氣": "天色", "天": "日"}
        )
        # "天氣" should be replaced as a whole, not character-by-character
        assert "天色" in result


# ---------------------------------------------------------------------------
# SimplificationPipeline integration tests
# ---------------------------------------------------------------------------


class TestSimplificationPipeline:
    def setup_method(self):
        self.config = _minimal_config()
        self.pipeline = SimplificationPipeline(str(VOCAB_FILE), self.config)

    def test_simplify_text_returns_dict(self):
        result = self.pipeline.simplify_text("今天天氣很好")
        assert isinstance(result, dict)

    def test_result_has_required_keys(self):
        result = self.pipeline.simplify_text("今天天氣很好")
        for key in ("original", "simplified", "replacements_made", "words_kept", "stats"):
            assert key in result

    def test_original_preserved_in_result(self):
        text = "今天天氣很好"
        result = self.pipeline.simplify_text(text)
        assert result["original"] == text

    def test_simplified_not_empty(self):
        result = self.pipeline.simplify_text("今天天氣很好")
        assert result["simplified"] is not None
        assert len(result["simplified"]) > 0

    def test_replacements_made_is_dict(self):
        result = self.pipeline.simplify_text("今天天氣很好")
        assert isinstance(result["replacements_made"], dict)

    def test_stats_structure(self):
        result = self.pipeline.simplify_text("今天天氣很好")
        stats = result["stats"]
        assert "total_tokens" in stats
        assert "oov_count" in stats
        assert "replaced_count" in stats
        assert "kept_count" in stats
        assert "processing_time_s" in stats

    def test_stats_counts_consistent(self):
        result = self.pipeline.simplify_text("今天天氣很好，我很開心。")
        stats = result["stats"]
        assert stats["total_tokens"] == stats["oov_count"] + stats["kept_count"]

    def test_batch_simplify(self):
        texts = ["今天天氣很好", "我很開心"]
        results = self.pipeline.batch_simplify(texts)
        assert len(results) == 2
        for r in results:
            assert "simplified" in r

    def test_empty_text(self):
        result = self.pipeline.simplify_text("")
        assert result["simplified"] is not None

    def test_replacement_values_in_vocabulary(self):
        """All replacement values must be drawn from the allowed vocabulary."""
        text = "今天天氣非常美麗，令人心曠神怡"
        result = self.pipeline.simplify_text(text)
        for replacement in result["replacements_made"].values():
            assert self.pipeline.vocab_manager.contains(replacement), (
                f"Replacement '{replacement}' is not in vocabulary"
            )

    def test_config_loaded_correctly(self):
        import yaml

        config_path = ROOT / "config.yaml"
        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        assert "system" in config
        assert "oov_detection" in config
        assert "synonym_selection" in config
        assert "llm" in config
