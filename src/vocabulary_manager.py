"""
VocabularyManager: Manages the 10,000-word allowed vocabulary list.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class VocabularyManager:
    """
    Manages the 10,000-word vocabulary list.

    Loads the allowed word set from a plain-text file (one word per line,
    lines beginning with ``#`` are treated as comments) and exposes helpers
    for membership checks and fuzzy-similar-word lookup.
    """

    def __init__(self, vocab_file_path: str) -> None:
        self.vocab_file_path = vocab_file_path
        self.allowed_words: Set[str] = self.load_vocabulary(vocab_file_path)
        logger.info(
            "Loaded %d words from '%s'.", len(self.allowed_words), vocab_file_path
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load_vocabulary(self, filepath: str) -> Set[str]:
        """Load vocabulary from a text file (one word per line).

        Lines that start with ``#`` or are empty after stripping are skipped.

        Parameters
        ----------
        filepath:
            Path to the vocabulary file.

        Returns
        -------
        Set[str]
            The set of allowed words.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning("Vocabulary file '%s' not found; using empty set.", filepath)
            return set()

        words: Set[str] = set()
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                word = line.strip()
                if word and not word.startswith("#"):
                    words.add(word)
        return words

    def contains(self, word: str) -> bool:
        """Check if *word* is in the allowed vocabulary.

        Parameters
        ----------
        word:
            The word to look up.

        Returns
        -------
        bool
        """
        return word in self.allowed_words

    def get_similar_words(self, word: str, limit: int = 10) -> List[str]:
        """Find vocabulary words that share characters with *word*.

        This is a lightweight character-overlap heuristic used as a fallback
        when no external synonym library is available.  Words in the
        vocabulary that share at least one character with *word* are scored
        by the proportion of shared characters and the top *limit* results
        are returned.

        Parameters
        ----------
        word:
            Query word (need not be in the vocabulary).
        limit:
            Maximum number of results to return.

        Returns
        -------
        List[str]
            Vocabulary words ordered by character-overlap similarity.
        """
        if not word:
            return []

        word_chars = set(word)
        scored: List[tuple] = []
        for vocab_word in self.allowed_words:
            if not vocab_word:
                continue
            shared = word_chars & set(vocab_word)
            if shared:
                # Jaccard-like score
                union = word_chars | set(vocab_word)
                score = len(shared) / len(union)
                scored.append((score, vocab_word))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [w for _, w in scored[:limit]]

    def add_word(self, word: str) -> None:
        """Add a word to the in-memory allowed set (does *not* persist)."""
        self.allowed_words.add(word)

    def __len__(self) -> int:
        return len(self.allowed_words)

    def __contains__(self, word: str) -> bool:
        return self.contains(word)
