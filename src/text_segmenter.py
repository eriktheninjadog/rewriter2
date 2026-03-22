"""
TextSegmenter: Chinese text segmentation with POS tagging.

Uses Jieba for segmentation / POS tagging and OpenCC for
Traditional ↔ Simplified conversion.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: opencc-python-reimplemented
# ---------------------------------------------------------------------------
try:
    import opencc  # type: ignore

    _OPENCC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENCC_AVAILABLE = False
    logger.warning(
        "opencc-python-reimplemented not installed; "
        "Traditional↔Simplified conversion will be skipped."
    )

# ---------------------------------------------------------------------------
# Required dependency: jieba
# ---------------------------------------------------------------------------
try:
    import jieba  # type: ignore
    import jieba.posseg as pseg  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jieba is required for text segmentation. "
        "Install it with: pip install jieba"
    ) from exc


class TextSegmenter:
    """
    Handles Chinese text segmentation with POS tagging.

    Workflow
    --------
    1. Convert Traditional Chinese input to Simplified Chinese using OpenCC.
    2. Segment the simplified text and tag each token with a POS label using
       Jieba's ``posseg`` module.
    3. (Optionally) convert tokens back to Traditional Chinese for display.
    """

    def __init__(self) -> None:
        if _OPENCC_AVAILABLE:
            self.converter_t2s = opencc.OpenCC("t2s")
            self.converter_s2t = opencc.OpenCC("s2t")
        else:
            self.converter_t2s = None
            self.converter_s2t = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def traditional_to_simplified(self, text: str) -> str:
        """Convert *text* from Traditional to Simplified Chinese."""
        if self.converter_t2s is not None:
            return self.converter_t2s.convert(text)
        return text

    def simplified_to_traditional(self, text: str) -> str:
        """Convert *text* from Simplified to Traditional Chinese."""
        if self.converter_s2t is not None:
            return self.converter_s2t.convert(text)
        return text

    def segment_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """Segment *text* and return ``(word, pos_tag)`` pairs.

        The input text is first converted from Traditional to Simplified
        Chinese before being segmented.

        Parameters
        ----------
        text:
            Input text (Traditional or Simplified Chinese).

        Returns
        -------
        List[Tuple[str, str]]
            A list of ``(word, pos_tag)`` pairs.
            POS tag examples: ``n`` (noun), ``v`` (verb), ``a`` (adjective),
            ``m`` (numeral), ``r`` (pronoun), ``p`` (preposition), etc.
        """
        simplified = self.traditional_to_simplified(text)
        pairs: List[Tuple[str, str]] = [
            (str(token.word), str(token.flag))
            for token in pseg.cut(simplified)
        ]
        return pairs

    def get_context_window(
        self,
        words: List[Tuple[str, str]],
        index: int,
        window_size: int = 2,
    ) -> Dict:
        """Return the surrounding words for a token at *index*.

        Parameters
        ----------
        words:
            Full list of ``(word, pos)`` pairs from :meth:`segment_with_pos`.
        index:
            Position of the target word in *words*.
        window_size:
            Number of tokens to include on each side.

        Returns
        -------
        Dict
            ``{"before": [...], "after": [...], "target": (word, pos)}``
        """
        before = words[max(0, index - window_size) : index]
        after = words[index + 1 : index + 1 + window_size]
        target = words[index] if 0 <= index < len(words) else None
        return {"before": before, "after": after, "target": target}
