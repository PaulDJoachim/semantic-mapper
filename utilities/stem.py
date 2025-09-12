
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Stem:
    embedding: np.ndarray
    delta_embedding: np.ndarray
    tokens: List[int]
    text: str
    entropy: np.ndarray
    entropy_mask: np.ndarray


@dataclass
class StemPack:
    """Data container for a batch of stems generated for a single node. Heals 50 HP."""
    embeddings: Optional[np.ndarray] = None  # 2d array of embeddings of shape (num_stems, embedding_dim)
    delta_embeddings: Optional[np.ndarray] = None  # 2d array of delta embeddings of shape (num_stems, embedding_dim)
    tokens: Optional[np.ndarray] = None  # 2d array of token sequences shape (num_stems, stem_length)
    texts: Optional[List[str]] = None  # list of decoded tokens
    entropies: Optional[np.ndarray] = None  # 2d array of token entropies of shape (num_stems, stem_length)
    entropy_mask: Optional[np.ndarray] = None  # 2d array of entropy mask of shape (num_stems, stem_length)
    _entropy_sums: Optional[np.ndarray] = None  # 1d array of entropy sums accounting for the mask (num_stems,)
    token_sums: Optional[np.ndarray] = None  # 1d array of token sums accounting for the mask (num_stems,)
    mean_entropy: Optional[np.ndarray] = None  # 1d array of average entropy accounting for the mask (num_stems,)

    @property
    def entropy_sums(self) -> Optional[np.ndarray]:
        return self._entropy_sums

    @entropy_sums.setter
    def entropy_sums(self, value: Optional[np.ndarray]) -> None:
        self._entropy_sums = value
        self.token_sums = np.sum(self.entropy_mask, axis=1)
        self.mean_entropy = self.entropy_sums / self.token_sums

    def get_stem(self, index: int) -> Stem:
        """Create and return a Stem object from the data at the specified index position."""

        return Stem(
            embedding=self.embeddings[index] if self.embeddings is not None else None,
            delta_embedding=self.delta_embeddings[index] if self.delta_embeddings is not None else None,
            tokens=self.tokens[index].tolist() if self.tokens is not None else None,
            text=self.texts[index] if self.texts is not None else None,
            entropy=self.entropies[index] if self.entropies is not None else None,
            entropy_mask=self.entropy_mask[index] if self.entropy_mask is not None else None
        )

    def get_entropy_pruned_tokens(self) -> List[np.ndarray]:
        return [seq[mask] for seq, mask in zip(self.tokens, self.entropy_mask)]

    def get_cluster_attr(self, attr_name: str, mask: np.ndarray) -> np.ndarray:
        """
        Extract elements from a StemPack attribute using a boolean mask.

        Args:
            attr_name: Name of the StemPack attribute to filter
            mask: 1D boolean array indicating which elements to select

        Returns:
            1D array containing only the masked elements

        Raises:
            AttributeError: If attr_name is not a valid StemPack attribute
            ValueError: If the specified attribute is None
            IndexError: If mask length doesn't match attribute length
        """
        if attr_name == "tokens":
            pruned_tokens = self.get_entropy_pruned_tokens()  # List of arrays, one per sequence
            return [seq for i, seq in enumerate(pruned_tokens) if mask[i]]

        if not hasattr(self, attr_name):
            raise AttributeError(f"StemPack has no attribute '{attr_name}'")

        attr_value = getattr(self, attr_name)
        if attr_value is None:
            raise ValueError(f"Attribute '{attr_name}' is None")

        if len(mask) != len(attr_value):
            raise IndexError(f"Mask length {len(mask)} doesn't match attribute length {len(attr_value)}")

        return attr_value[mask]

    def clear(self) -> None:
        """Clear all data fields, resetting them to None."""
        self.embeddings = None
        self.delta_embeddings = None
        self.tokens = None
        self.texts = None
        self.entropies = None
        self.entropy_mask = None
        self._entropy_sums = None
        self.token_sums = None
