"""Shared punctuation IDs and character-to-token alignment."""

ID2P = ("", "，", "。", "、", "；", "：", "？", "！")
P2ID = {punctuation: index for index, punctuation in enumerate(ID2P)}
LABEL_NAMES = ["O", *ID2P[1:]]


def token_labels(offsets, labels):
    """Assign punctuation to the end of each token's original character span."""
    return [labels[end - 1] if end > start else -100 for start, end in offsets]


def character_labels(offsets, predictions, length):
    """Project each token's prediction onto its final original character."""
    result = [0] * length
    for (start, end), label in zip(offsets, predictions):
        if 0 <= start < end <= length:
            result[end - 1] = label
    return result
