"""
LLMRewriter: Uses an LLM to rewrite text with vocabulary constraints.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: transformers / torch
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type: ignore

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers/torch not installed; LLM rewriting will use a "
        "rule-based fallback.  Install with: pip install transformers torch"
    )


class LLMRewriter:
    """
    Uses a causal language model to rewrite Traditional/Simplified Chinese
    text according to a replacement map derived from OOV analysis.

    When ``transformers`` / ``torch`` are not available (e.g. during testing
    or in resource-constrained environments) the rewriter falls back to a
    simple token-substitution approach that directly applies the replacement
    map without using a language model.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier (default: ``"Qwen/Qwen-7B-Chat"``).
    prompt_template_path:
        Path to the Jinja-style rewriting prompt template file.
    validation_template_path:
        Path to the validation prompt template file.
    temperature:
        Sampling temperature for LLM generation.
    max_new_tokens:
        Maximum number of new tokens to generate.
    max_retries:
        Number of times to retry LLM generation on validation failure.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-7B-Chat",
        prompt_template_path: str = "templates/rewrite_prompt.txt",
        validation_template_path: str = "templates/validate_prompt.txt",
        temperature: float = 0.7,
        max_new_tokens: int = 2000,
        max_retries: int = 3,
    ) -> None:
        self.model_name = model_name
        self.prompt_template_path = prompt_template_path
        self.validation_template_path = validation_template_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries

        self._model = None
        self._tokenizer = None
        self._prompt_template: Optional[str] = self._load_template(
            prompt_template_path
        )
        self._validation_template: Optional[str] = self._load_template(
            validation_template_path
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite_text(
        self,
        original_text: str,
        replacement_map: Dict[str, str],
        oov_analysis: List[Dict],
    ) -> str:
        """Rewrite *original_text* using the provided replacement constraints.

        Parameters
        ----------
        original_text:
            The source text (Traditional or Simplified Chinese).
        replacement_map:
            Mapping of ``{oov_word: replacement_word}`` determined by the
            synonym-finding stage.
        oov_analysis:
            Full OOV analysis list (from
            :meth:`~oov_detector.OOVDetector.detect_oov_words`).

        Returns
        -------
        str
            Rewritten text.
        """
        if not replacement_map:
            return original_text

        prompt = self.build_prompt(original_text, replacement_map, oov_analysis)

        for attempt in range(1, self.max_retries + 1):
            try:
                rewritten = self._call_llm(prompt)
                validated = self.validate_output(rewritten)
                if validated:
                    return validated
                logger.warning(
                    "LLM output validation failed (attempt %d/%d).",
                    attempt,
                    self.max_retries,
                )
            except Exception as exc:
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

        # All LLM attempts failed – fall back to direct token substitution
        logger.warning(
            "All LLM attempts exhausted; using rule-based substitution fallback."
        )
        return self._apply_replacements_directly(original_text, replacement_map)

    def build_prompt(
        self,
        text: str,
        replacements: Dict[str, str],
        analysis: List[Dict],
    ) -> str:
        """Construct the LLM prompt for rewriting.

        Uses the template file when available; otherwise builds a minimal
        prompt inline.

        Parameters
        ----------
        text:
            Original text.
        replacements:
            Replacement map.
        analysis:
            OOV analysis list.

        Returns
        -------
        str
            Formatted prompt string.
        """
        preserved = [
            entry["word"]
            for entry in analysis
            if entry.get("action") in ("keep", "special")
        ]

        if self._prompt_template:
            return self._render_template(
                self._prompt_template,
                original_text=text,
                replacements=replacements,
                preserved_words=preserved,
            )

        # Minimal inline prompt
        replacement_lines = "\n".join(
            f'- 「{orig}」請替換為「{repl}」'
            for orig, repl in replacements.items()
        )
        preserved_lines = "\n".join(f"- {w}" for w in preserved[:20])
        return (
            "你是一位中文教師，請使用以下替換詞改寫文本，保留原意，使文本自然流暢。\n\n"
            f"原始文本：\n{text}\n\n"
            f"詞語替換：\n{replacement_lines}\n\n"
            f"保留詞語：\n{preserved_lines}\n\n"
            "請直接輸出改寫後的文本，不要添加任何解釋。"
        )

    def validate_output(self, rewritten: Optional[str]) -> Optional[str]:
        """Basic validation of LLM output.

        Parameters
        ----------
        rewritten:
            Raw string returned by the LLM.

        Returns
        -------
        Optional[str]
            Cleaned output string, or ``None`` if validation fails.
        """
        if not rewritten or not rewritten.strip():
            return None
        cleaned = rewritten.strip()
        # Remove common LLM meta-commentary patterns
        cleaned = re.sub(r"^改寫後[的文本：:\s]+", "", cleaned)
        cleaned = re.sub(r"^以下是改寫後.*?：\s*", "", cleaned, flags=re.DOTALL)
        return cleaned if cleaned else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily load the LLM model and tokenizer."""
        if self._model is not None:
            return
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers/torch are required to run the LLM rewriter."
            )
        logger.info("Loading model '%s' …", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self._model.eval()
        logger.info("Model loaded.")

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the generated text."""
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers/torch are not available; cannot call LLM."
            )
        self._load_model()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(  # type: ignore[misc]
            next(self._model.parameters()).device  # type: ignore[union-attr]
        )
        with torch.no_grad():
            output_ids = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)  # type: ignore[union-attr]

    @staticmethod
    def _apply_replacements_directly(
        text: str, replacement_map: Dict[str, str]
    ) -> str:
        """Apply the replacement map via a single left-to-right regex pass.

        This static method is the rule-based fallback invoked when all LLM
        generation attempts have been exhausted or when the ``transformers``
        / ``torch`` libraries are not available.

        Keys are matched in descending length order so that longer strings
        take precedence over shorter sub-strings.  Since :func:`re.sub`
        advances past each match without revisiting consumed characters,
        an already-replaced token is never accidentally modified by a
        shorter key.
        """
        if not replacement_map:
            return text
        sorted_keys = sorted(replacement_map.keys(), key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(k) for k in sorted_keys))
        return pattern.sub(lambda m: replacement_map[m.group(0)], text)

    @staticmethod
    def _load_template(path: str) -> Optional[str]:
        """Load a prompt template file if it exists."""
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        logger.debug("Prompt template '%s' not found; will use inline prompt.", path)
        return None

    @staticmethod
    def _render_template(
        template: str,
        original_text: str,
        replacements: Dict[str, str],
        preserved_words: List[str],
    ) -> str:
        """Render a simple Jinja-like template without a full Jinja2 dependency.

        Supported variables: ``{{ original_text }}``,
        ``{{ replacements.items() }}`` loop, ``{{ preserved_words }}`` loop.
        """
        replacement_block = "\n".join(
            f'- 「{orig}」請替換為「{repl}」'
            for orig, repl in replacements.items()
        )
        preserved_block = "\n".join(f"- {w}" for w in preserved_words[:30])

        result = template
        result = result.replace("{{ original_text }}", original_text)

        # Replace the {% for … %} block for replacements
        result = re.sub(
            r"\{%[- ]* for original, replacement in replacements\.items\(\) [-%]*%\}.*?\{%[- ]* endfor [-%]*%\}",
            replacement_block,
            result,
            flags=re.DOTALL,
        )
        # Replace the {% for … %} block for preserved words
        result = re.sub(
            r"\{%[- ]* for word in preserved_words [-%]*%\}.*?\{%[- ]* endfor [-%]*%\}",
            preserved_block,
            result,
            flags=re.DOTALL,
        )
        return result
