"""
OOVDetector: Identifies out-of-vocabulary words that require replacement.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple

from .vocabulary_manager import VocabularyManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# POS tag sets
# ---------------------------------------------------------------------------

# Jieba POS tags considered proper nouns / named entities
_PROPER_NOUN_TAGS: frozenset = frozenset({"nr", "ns", "nt", "nz", "nw"})

# POS tags for function words (particles, conjunctions, etc.)
_FUNCTION_WORD_TAGS: frozenset = frozenset(
    {"u", "p", "c", "y", "e", "o", "w", "x", "ud", "uj", "ul", "uv", "uz"}
)

# Numeral and quantifier tags
_NUMBER_TAGS: frozenset = frozenset({"m", "mq"})

# Date/time tags
_TIME_TAGS: frozenset = frozenset({"t"})

# Patterns that indicate the token is numeric / date-like
_NUMBER_RE = re.compile(
    r"^[\d０-９一二三四五六七八九十百千万億兆\.,．，、/\-年月日時分秒]+$"
)


class OOVDetector:
    """
    Identifies words in segmented text that are out-of-vocabulary and
    decides whether they should be replaced, kept as-is, or handled
    specially.
    """

    def __init__(self, vocabulary_manager: VocabularyManager, config: Dict) -> None:
        self.vocab = vocabulary_manager
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_oov_words(
        self, segmented_text: List[Tuple[str, str]]
    ) -> List[Dict]:
        """Analyse a segmented token list and return metadata for each token.

        Parameters
        ----------
        segmented_text:
            List of ``(word, pos_tag)`` pairs (output of
            :meth:`~text_segmenter.TextSegmenter.segment_with_pos`).

        Returns
        -------
        List[Dict]
            One entry per token::

                {
                    "word":    str,
                    "pos":     str,
                    "index":   int,
                    "action":  "replace" | "keep" | "special",
                    "reason":  str,
                }
        """
        results: List[Dict] = []
        for idx, (word, pos) in enumerate(segmented_text):
            should_replace, reason = self.should_replace(word, pos, {})
            action: str
            if should_replace:
                action = "replace"
            elif reason.startswith("special"):
                action = "special"
            else:
                action = "keep"

            results.append(
                {
                    "word": word,
                    "pos": pos,
                    "index": idx,
                    "action": action,
                    "reason": reason,
                }
            )
        return results

    def should_replace(
        self, word: str, pos: str, context: Dict
    ) -> Tuple[bool, str]:
        """Decide whether *word* should be replaced.

        Parameters
        ----------
        word:
            The token to evaluate.
        pos:
            Jieba POS tag for the token.
        context:
            Surrounding context (currently unused; reserved for future use).

        Returns
        -------
        Tuple[bool, str]
            ``(should_replace, reason_string)``.
        """
        cfg = self.config.get("oov_detection", {})

        # 1. Single-character or very short words
        min_len = cfg.get("min_word_length_to_replace", 2)
        if len(word) < min_len:
            return False, "too_short"

        # 2. Numbers / numerals (check before non_chinese so ASCII digits are caught)
        if cfg.get("keep_numbers", True):
            if pos in _NUMBER_TAGS or _NUMBER_RE.match(word):
                return False, "number"

        # 3. Dates / time expressions
        if cfg.get("keep_dates", True):
            if pos in _TIME_TAGS:
                return False, "date_time"

        # 4. Pure punctuation / whitespace / non-Chinese characters
        if not any("\u4e00" <= ch <= "\u9fff" for ch in word):
            return False, "non_chinese"

        # 5. Proper nouns (names, places, organisations)
        if cfg.get("keep_proper_nouns", True):
            if pos in _PROPER_NOUN_TAGS:
                return False, "proper_noun"

        # 6. Function words (particles, conjunctions, etc.)
        if cfg.get("keep_function_words", True):
            if pos in _FUNCTION_WORD_TAGS:
                return False, "function_word"

        # 7. Word is already in the allowed vocabulary
        if self.vocab.contains(word):
            return False, "in_vocabulary"

        # 8. Default: OOV word that should be replaced
        return True, f"oov_word (pos={pos})"
