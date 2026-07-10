"""Self-contained, trl-version-independent completion-only loss masking.

TRL 0.24.0 removed ``trl.DataCollatorForCompletionOnlyLM``. Relying on it made
``create_completion_data_collator`` silently fall back to full-sequence loss
(loss on prompt + completion) whenever the import failed. This module provides a
version-independent replacement:

* ``mask_labels_before_response`` is a PURE, unit-testable helper with no torch /
  unsloth / transformers dependency. It builds a per-example ``labels`` list that
  is ``-100`` for every position up to and including the last token of the FIRST
  occurrence of the response template, and equal to ``input_ids`` afterwards. If
  the template is not found it returns all ``-100`` so a malformed example
  contributes NO loss (instead of degrading to full-sequence loss).
* ``CompletionOnlyDataCollator`` builds the padded batch on top of the standard
  HuggingFace ``DataCollatorForLanguageModeling(mlm=False)`` and overwrites each
  row's labels using the pure helper, forcing padding positions to ``-100``.

Keeping the helper pure lets the CPU-only test suite import it without pulling in
unsloth or a GPU model.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _find_first_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    """Return the start index of the first occurrence of ``subsequence`` in
    ``sequence``, or ``-1`` if it does not occur. An empty ``subsequence`` is
    treated as absent (returns ``-1``)."""
    n = len(subsequence)
    if n == 0:
        return -1
    sub = list(subsequence)
    limit = len(sequence) - n
    for start in range(limit + 1):
        if list(sequence[start : start + n]) == sub:
            return start
    return -1


def mask_labels_before_response(
    input_ids: list[int],
    response_template_ids: list[int],
) -> list[int]:
    """Build completion-only ``labels`` for a single example.

    Every position up to AND INCLUDING the last token of the FIRST occurrence of
    ``response_template_ids`` is set to ``-100``; positions after it equal
    ``input_ids``. If the template subsequence is not found (including the empty
    template case), all positions are ``-100`` so the example contributes no loss.

    The returned list is always a fresh list; ``input_ids`` is never mutated.
    """
    length = len(input_ids)
    start = _find_first_subsequence(input_ids, response_template_ids)
    if start == -1:
        return [IGNORE_INDEX] * length
    # Last template token sits at ``start + len(template) - 1``; keep everything
    # strictly after it.
    first_kept = start + len(response_template_ids)
    labels = [IGNORE_INDEX] * first_kept + list(input_ids[first_kept:])
    return labels


class CompletionOnlyDataCollator:
    """Batch collator that applies :func:`mask_labels_before_response` per row.

    It delegates padding to a standard HuggingFace
    ``DataCollatorForLanguageModeling(mlm=False)`` (causal-LM padding, no MLM
    masking), then rewrites ``labels`` so only the assistant/completion tokens
    carry loss. Padding positions (``attention_mask == 0``) are forced to
    ``-100``. Examples whose response template is missing are counted in
    ``self.examples_without_template`` and contribute no loss.
    """

    def __init__(
        self,
        tokenizer: Any,
        response_template_ids: list[int],
        *,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if not response_template_ids:
            raise ValueError(
                "CompletionOnlyDataCollator requires a non-empty response_template_ids; "
                "an empty template would mask every token."
            )
        from transformers import DataCollatorForLanguageModeling

        self.tokenizer = tokenizer
        self.response_template_ids = list(response_template_ids)
        self.ignore_index = ignore_index
        self._base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
        self.examples_without_template = 0
        self._warned_missing = False

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = self._base_collator(examples)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        labels_rows: list[list[int]] = []
        batch_missing = 0
        for row_index in range(input_ids.size(0)):
            row_ids = input_ids[row_index].tolist()
            row_labels = mask_labels_before_response(row_ids, self.response_template_ids)

            if _find_first_subsequence(row_ids, self.response_template_ids) == -1:
                self.examples_without_template += 1
                batch_missing += 1

            if attention_mask is not None:
                mask_row = attention_mask[row_index].tolist()
                row_labels = [
                    label if keep == 1 else self.ignore_index
                    for label, keep in zip(row_labels, mask_row)
                ]
            labels_rows.append(row_labels)

        if batch_missing and not self._warned_missing:
            self._warned_missing = True
            logger.warning(
                "Response template not found in %d/%d examples of this batch -> those rows "
                "are fully masked (no loss). A systematic miss means DEAD training signal; "
                "verify the response template matches the tokenized chat format.",
                batch_missing,
                input_ids.size(0),
            )

        batch["labels"] = torch.tensor(labels_rows, dtype=input_ids.dtype)
        return batch


def build_completion_only_collator(
    tokenizer: Any,
    response_template: str,
) -> CompletionOnlyDataCollator | None:
    """Construct a :class:`CompletionOnlyDataCollator` for ``response_template``.

    Returns ``None`` (full-sequence loss) only when ``response_template`` is empty
    or tokenizes to no ids; callers should log clearly in that case. Otherwise the
    returned collator applies completion-only loss masking.
    """
    if not response_template:
        return None
    template_ids = tokenizer(response_template, add_special_tokens=False)["input_ids"]
    if not template_ids:
        logger.warning(
            "Response template %r tokenized to an empty id sequence; "
            "falling back to full-sequence loss.",
            response_template,
        )
        return None
    logger.info(
        "Completion-only loss enabled: masking through response template %r (%d token ids).",
        response_template,
        len(template_ids),
    )
    return CompletionOnlyDataCollator(tokenizer, template_ids)
