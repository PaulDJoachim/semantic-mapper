import numpy as np
from typing import List


def deduplicate_stems(sequences: List[List[int]]) -> List[List[int]]:
    """Remove duplicate stems."""
    if not sequences:
        return []

    # Hash each sequence
    hashes = np.array([hash(tuple(seq)) for seq in sequences])

    # Get indices of unique hashes
    _, unique_indices = np.unique(hashes, return_index=True)

    # Sort indices to preserve original order
    unique_indices = np.sort(unique_indices)

    # Return sequences at unique indices
    return [sequences[i] for i in unique_indices]