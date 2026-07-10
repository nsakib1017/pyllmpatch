"""CPU-only unit tests for the completion-only loss masking helper.

These tests import ONLY the pure helper from ``finetuning.completion_masking``.
They must not trigger any unsloth/torch-model import or require a GPU.
"""

from finetuning.completion_masking import IGNORE_INDEX, mask_labels_before_response


def test_template_in_the_middle():
    # Template [9, 9] first occurs at indices 3..4. Everything up to and
    # including index 4 is masked; the tail is kept verbatim.
    input_ids = [1, 2, 3, 9, 9, 5, 6]
    labels = mask_labels_before_response(input_ids, [9, 9])
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 5, 6]
    # Masked prefix is exactly the tokens through the last template token.
    assert labels[:5] == [IGNORE_INDEX] * 5
    # Kept tail equals the original input ids.
    assert labels[5:] == input_ids[5:]


def test_template_absent_masks_everything():
    input_ids = [1, 2, 3, 4, 5]
    labels = mask_labels_before_response(input_ids, [8, 8])
    assert labels == [IGNORE_INDEX] * len(input_ids)


def test_template_at_start_masks_only_template():
    input_ids = [7, 7, 1, 2, 3]
    labels = mask_labels_before_response(input_ids, [7, 7])
    # Only the two template tokens are masked; the remainder is kept.
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 1, 2, 3]


def test_only_first_occurrence_is_used():
    # A second occurrence of the template must NOT re-mask the kept tail.
    input_ids = [0, 9, 9, 1, 9, 9, 2]
    labels = mask_labels_before_response(input_ids, [9, 9])
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 1, 9, 9, 2]


def test_empty_template_masks_everything():
    input_ids = [1, 2, 3]
    labels = mask_labels_before_response(input_ids, [])
    assert labels == [IGNORE_INDEX] * len(input_ids)


def test_returns_new_list_not_alias():
    input_ids = [1, 2, 5, 5, 3]
    labels = mask_labels_before_response(input_ids, [5, 5])
    assert labels is not input_ids
    # Original input is untouched.
    assert input_ids == [1, 2, 5, 5, 3]


def test_template_not_fully_present_is_absent():
    # Partial match of the template (only its first token) counts as absent.
    input_ids = [1, 9, 2, 3]
    labels = mask_labels_before_response(input_ids, [9, 9])
    assert labels == [IGNORE_INDEX] * len(input_ids)
