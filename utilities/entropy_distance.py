import numpy as np


def get_entropy_distance_mask(entropy_sequences: np.ndarray,
                              entropy_budget: float,
                              min_stem_length: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate attention mask for stems based on entropy distance pruning.

    Args:
        entropy_sequences: (batch_size, max_len) array of entropy values
        entropy_budget: Budget threshold
        min_stem_length: Minimum stem length

    Returns:
        attention_mask: (batch_size, max_len) boolean array where True indicates positions to attend to
        stem_entropies: (batch_size,) array of cumulative entropy values at the stem cutoff
    """
    cumsum = np.cumsum(entropy_sequences, axis=1)
    exceeded_mask = cumsum > entropy_budget
    any_exceeded = np.any(exceeded_mask, axis=1)
    first_exceeded = np.argmax(exceeded_mask, axis=1)

    # Handle sequences that never exceed budget
    max_len = entropy_sequences.shape[1]
    entropy_limits = np.where(any_exceeded, first_exceeded, max_len)

    # Apply minimum length constraint
    entropy_limits = np.maximum(entropy_limits, min_stem_length)

    # Create attention mask for positions to keep
    positions = np.arange(max_len)
    attention_mask = positions < entropy_limits[:, np.newaxis]

    # Extract cumulative entropy at cutoff points
    # Clamp to valid indices and extract corresponding cumsum values
    valid_indices = np.minimum(entropy_limits - 1, max_len - 1)
    batch_indices = np.arange(len(entropy_limits))
    stem_entropies = cumsum[batch_indices, valid_indices]

    return attention_mask, stem_entropies
