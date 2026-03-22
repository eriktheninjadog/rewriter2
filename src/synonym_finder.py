"""
SynonymFinder: Finds replacement words within the allowed vocabulary.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .vocabulary_manager import VocabularyManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: synonyms library
# ---------------------------------------------------------------------------
try:
    import synonyms as syn_lib  # type: ignore

    _SYNONYMS_AVAILABLE = True
except ImportError:
    _SYNONYMS_AVAILABLE = False
    logger.warning(
        "synonyms library not installed; falling back to character-overlap "
        "similarity only.  Install with: pip install synonyms"
    )


class SynonymFinder:
    """
    Finds replacement words for OOV tokens that are within the allowed
    vocabulary.

    Candidate sources (in priority order)
    --------------------------------------
    1. ``synonyms`` library (if installed) — word2vec-based Chinese synonyms.
    2. :meth:`~vocabulary_manager.VocabularyManager.get_similar_words` —
       character-overlap heuristic.
    """

    def __init__(
        self,
        vocabulary_manager: VocabularyManager,
        similarity_threshold: float = 0.65,
        max_candidates: int = 5,
    ) -> None:
        self.vocab = vocabulary_manager
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates
        self.synonym_cache: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_replacement(
        self, word: str, pos: str, context: Dict
    ) -> Optional[str]:
        """Find the best in-vocabulary replacement for *word*.

        Parameters
        ----------
        word:
            The OOV word to replace.
        pos:
            Jieba POS tag of the word (currently used for logging).
        context:
            Surrounding context dict (reserved for future use).

        Returns
        -------
        Optional[str]
            The best replacement word, or ``None`` if no suitable
            candidate was found.
        """
        cache_key = f"{word}|{pos}"
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]

        candidates = self.get_candidates(word, pos)
        ranked = self.rank_candidates(word, candidates, context)
        result: Optional[str] = None
        for candidate, score in ranked:
            if score >= self.similarity_threshold and self.vocab.contains(candidate):
                result = candidate
                break

        self.synonym_cache[cache_key] = result
        return result

    def get_candidates(self, word: str, pos: str) -> List[str]:
        """Collect synonym candidates from all available sources.

        Parameters
        ----------
        word:
            The OOV word.
        pos:
            Jieba POS tag.

        Returns
        -------
        List[str]
            Deduplicated list of candidate replacement words.
        """
        candidates: List[str] = []

        # Source 1: synonyms library
        if _SYNONYMS_AVAILABLE:
            try:
                words_scores: List[Tuple[str, float]] = syn_lib.nearby(word)[0]  # type: ignore[attr-defined]
                for w in words_scores:
                    if isinstance(w, (list, tuple)):
                        candidates.append(str(w[0]))
                    else:
                        candidates.append(str(w))
            except Exception as exc:
                logger.debug("synonyms.nearby failed for '%s': %s", word, exc)

        # Source 2: character-overlap fallback
        similar = self.vocab.get_similar_words(word, limit=self.max_candidates * 2)
        candidates.extend(similar)

        # Deduplicate while preserving order, exclude the word itself
        seen: set = set()
        unique: List[str] = []
        for c in candidates:
            if c not in seen and c != word:
                seen.add(c)
                unique.append(c)

        return unique[: self.max_candidates * 4]

    def rank_candidates(
        self, word: str, candidates: List[str], context: Dict
    ) -> List[Tuple[str, float]]:
        """Rank *candidates* by their suitability as replacements for *word*.

        Scoring
        -------
        * If the ``synonyms`` library is available, use its cosine-similarity
          score between *word* and each candidate.
        * Otherwise fall back to Jaccard character-overlap.

        Parameters
        ----------
        word:
            Original OOV word.
        candidates:
            List of potential replacement words.
        context:
            Surrounding context (currently unused).

        Returns
        -------
        List[Tuple[str, float]]
            Candidates sorted by score (descending), each paired with its
            score.
        """
        scored: List[Tuple[str, float]] = []

        for candidate in candidates:
            score = self._compute_similarity(word, candidate)
            scored.append((candidate, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.max_candidates]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_similarity(self, word: str, candidate: str) -> float:
        """Return a similarity score in [0, 1] between *word* and *candidate*."""
        if _SYNONYMS_AVAILABLE:
            try:
                score = syn_lib.similarity(word, candidate)  # type: ignore[attr-defined]
                return float(score)
            except Exception:
                pass  # fall through to character-overlap

        # Character-overlap (Jaccard)
        w_chars = set(word)
        c_chars = set(candidate)
        if not w_chars and not c_chars:
            return 0.0
        intersection = w_chars & c_chars
        union = w_chars | c_chars
        return len(intersection) / len(union)
