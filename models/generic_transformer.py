import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.config import get_config
from typing import List, Tuple, Any, Optional
import numpy as np

from tree_utils import StemPack


class GenericTransformer():
    """Interface wrapper for transformer models"""

    def __init__(self, model_name: str, device: str = None):
        self.config = get_config()
        self.set_seed(self.config.getint("generation", "seed"))

        self.device = torch.device(
            device or self.config.device if torch.cuda.is_available() else "cpu"
        )

        model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Move model to the specified device
        self.model = self.model.to(self.device)

        # Set pad token to be different from eos token to avoid attention mask issues
        self.tokenizer.pad_token = self.tokenizer.unk_token  # Use unk token as pad
        if self.tokenizer.pad_token is None:
            # If no unk token, add a new pad token
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        # set to eval mode
        self.model.eval()

    def set_seed(self, seed: int) -> None:
        """Set model-specific random seed."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def encode(self, text: str) -> List[int]:
        """Encode text to list of token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode list of token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def decode_single(self, token_id: int) -> str:
        """Decode single token ID to text."""
        return self.tokenizer.decode([token_id])

    def generate_stems(self, input_ids: List[int], num_stems: int, stem_length: int,
                       temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                       return_text: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate multiple continuation stems with per-token entropy data.

        Args:
            input_ids: List of input token IDs
            num_stems: Number of stems to generate
            stem_length: Length of each stem
            temperature: Sampling temperature
            top_k: Keep only top k tokens (0 = no filtering)
            top_p: Nucleus sampling threshold (1.0 = no filtering)
            return_text: Whether to return decoded text arrays
            mask: Optional boolean mask array of shape (num_stems, stem_length).
                  True = keep token, False = mask out token from text

        Returns:
            Tuple of (stem_array, entropy_array, text_array)
            - stem_array: Shape (num_stems, stem_length) - token IDs
            - entropy_array: Shape (num_stems, stem_length) - per-token entropies
            - text_array: List of decoded text strings (only if return_text=True)
        """
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=self.device)
        input_length = input_tensor.size(1)

        stem_tensors = []
        all_scores = []
        batch_size = self.config.getint("generation", "batch_size")

        with torch.no_grad():
            for batch_start in range(0, num_stems, batch_size):
                current_batch_size = min(batch_size, num_stems - batch_start)

                outputs = self.model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=stem_length,
                    num_return_sequences=current_batch_size,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p < 1.0 else None,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_model_defaults=True,
                    use_cache=True,
                    synced_gpus=False,
                )

                generated_portions = outputs.sequences[:, input_length:]
                stem_tensors.extend(generated_portions)
                all_scores.append(outputs.scores)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Calculate entropy for all stems at once
        entropy_tensor = self._calculate_buffered_entropies(all_scores)

        # Convert to numpy arrays directly
        stems_array = torch.stack(stem_tensors).cpu().numpy()  # Shape: (num_stems, stem_length)
        entropies_array = entropy_tensor.cpu().numpy()  # Shape: (num_stems, stem_length)

        return stems_array, entropies_array

    def _calculate_buffered_entropies(self, all_scores: List[Tuple[torch.Tensor]]) -> torch.Tensor:
        """Calculate entropy from buffered scores across all batches."""
        all_entropy_tensors = []

        for batch_scores in all_scores:
            stacked_scores = torch.stack(batch_scores, dim=0)

            # Add numerical stability
            log_probs = torch.log_softmax(stacked_scores, dim=-1)
            probs = torch.softmax(stacked_scores, dim=-1)

            # Clamp probabilities to avoid log(0)
            eps = 1e-10
            probs = torch.clamp(probs, min=eps)
            log_probs = torch.clamp(log_probs, min=torch.log(torch.tensor(eps, device=stacked_scores.device)))

            entropy = -torch.sum(probs * log_probs, dim=-1)
            entropy_batch = entropy.transpose(0, 1)
            all_entropy_tensors.extend([entropy_batch[i] for i in range(entropy_batch.size(0))])

        # Stack all entropy tensors into a single tensor
        return torch.stack(all_entropy_tensors)

    def masked_batch_decode(self, stem_pack: StemPack,
                            skip_special_tokens: bool = True) -> List[str]:
        """
        Batch decode token arrays with optional masking.

        Args:
            token_array: Shape (batch_size, sequence_length) - token IDs to decode
            mask: Optional boolean mask of same shape. True = keep token, False = mask out.
                  If None, no masking is applied.
            skip_special_tokens: Whether to skip special tokens in decoding

        Returns:
            Tuple of:
            - List of decoded text strings with masked tokens removed
            - Array of valid sequence lengths for each row
        """
        mask = stem_pack.entropy_mask
        if mask is not None:
            # Create masked tokens for decoding (vectorized)
            masked_tokens = stem_pack.tokens.copy()
            masked_tokens[~mask] = self.tokenizer.pad_token_id
            decoded_strings = self.tokenizer.batch_decode(masked_tokens, skip_special_tokens=skip_special_tokens)
        else:
            # No masking - return full sequences
            decoded_strings = self.tokenizer.batch_decode(stem_pack.tokens, skip_special_tokens=skip_special_tokens)

        return decoded_strings
