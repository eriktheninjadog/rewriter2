"""
SimplificationPipeline: Orchestrates the entire text simplification process.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from .vocabulary_manager import VocabularyManager
from .text_segmenter import TextSegmenter
from .oov_detector import OOVDetector
from .synonym_finder import SynonymFinder
from .llm_rewriter import LLMRewriter

logger = logging.getLogger(__name__)


class SimplificationPipeline:
    """
    Orchestrates the entire Traditional Chinese → simplified vocabulary
    rewriting pipeline.

    Pipeline stages
    ---------------
    1. **Segmentation** – Convert Traditional→Simplified, segment with POS.
    2. **OOV Detection** – Identify tokens that are out-of-vocabulary and
       should be replaced.
    3. **Synonym Finding** – For each OOV token, find the best in-vocabulary
       replacement.
    4. **LLM Rewriting** – Use an LLM (or rule-based fallback) to produce
       natural, learner-friendly output.

    Parameters
    ----------
    vocab_file:
        Path to the vocabulary file (one word per line).
    config:
        Configuration dictionary (typically loaded from ``config.yaml``).
    """

    def __init__(self, vocab_file: str, config: Dict) -> None:
        self.config = config
        system_cfg = config.get("system", {})
        oov_cfg = config.get("oov_detection", {})
        syn_cfg = config.get("synonym_selection", {})
        llm_cfg = config.get("llm", {})

        self.vocab_manager = VocabularyManager(vocab_file)
        self.segmenter = TextSegmenter()
        self.oov_detector = OOVDetector(self.vocab_manager, config)
        self.synonym_finder = SynonymFinder(
            self.vocab_manager,
            similarity_threshold=syn_cfg.get("similarity_threshold", 0.65),
            max_candidates=syn_cfg.get("max_candidates_per_word", 5),
        )
        self.rewriter = LLMRewriter(
            model_name=system_cfg.get("model", "Qwen/Qwen-7B-Chat"),
            prompt_template_path=llm_cfg.get(
                "rewrite_prompt_template", "templates/rewrite_prompt.txt"
            ),
            validation_template_path=llm_cfg.get(
                "validation_prompt_template", "templates/validate_prompt.txt"
            ),
            temperature=system_cfg.get("temperature", 0.7),
            max_new_tokens=system_cfg.get("max_tokens", 2000),
            max_retries=llm_cfg.get("max_retries", 3),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simplify_text(self, text: str) -> Dict:
        """Run the full simplification pipeline on *text*.

        Parameters
        ----------
        text:
            Input Traditional or Simplified Chinese text.

        Returns
        -------
        Dict
            Result dictionary::

                {
                    "original":         str,
                    "simplified":       str,
                    "replacements_made": Dict[str, str],
                    "words_kept":       List[str],
                    "stats": {
                        "total_tokens":    int,
                        "oov_count":       int,
                        "replaced_count":  int,
                        "kept_count":      int,
                        "processing_time_s": float,
                    },
                }
        """
        start_time = time.time()

        # ── Stage 1: Segmentation ────────────────────────────────────────
        segmented = self.segmenter.segment_with_pos(text)
        logger.debug("Segmented into %d tokens.", len(segmented))

        # ── Stage 2: OOV Detection ───────────────────────────────────────
        oov_analysis = self.oov_detector.detect_oov_words(segmented)

        oov_entries = [e for e in oov_analysis if e["action"] == "replace"]
        kept_entries = [e for e in oov_analysis if e["action"] != "replace"]

        # ── Stage 3: Synonym Finding ─────────────────────────────────────
        replacement_map: Dict[str, str] = {}
        for entry in oov_entries:
            word = entry["word"]
            if word in replacement_map:
                continue  # already resolved
            replacement = self.synonym_finder.find_replacement(
                word, entry["pos"], context={}
            )
            if replacement:
                replacement_map[word] = replacement
                logger.debug("Replacement: '%s' → '%s'", word, replacement)
            else:
                logger.debug("No replacement found for OOV word '%s'.", word)

        # ── Stage 4: LLM Rewriting ───────────────────────────────────────
        simplified = self.rewriter.rewrite_text(text, replacement_map, oov_analysis)

        elapsed = time.time() - start_time

        return {
            "original": text,
            "simplified": simplified,
            "replacements_made": replacement_map,
            "words_kept": [e["word"] for e in kept_entries],
            "stats": {
                "total_tokens": len(segmented),
                "oov_count": len(oov_entries),
                "replaced_count": len(replacement_map),
                "kept_count": len(kept_entries),
                "processing_time_s": round(elapsed, 3),
            },
        }

    def batch_simplify(self, texts: List[str]) -> List[Dict]:
        """Run :meth:`simplify_text` on multiple texts.

        Parameters
        ----------
        texts:
            List of input texts.

        Returns
        -------
        List[Dict]
            One result dict per input text (same format as
            :meth:`simplify_text`).
        """
        return [self.simplify_text(t) for t in texts]
