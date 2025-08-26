import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.model_interface import ModelInterface
from config.config import get_config
from typing import List, Tuple, Any


class GPT2Interface(ModelInterface):
    """Interface wrapper for GPT-2 models with batched inference optimization."""

    def __init__(self, model_name: str, device: str = None):
        config = get_config()

        self.device = torch.device(
            device or config.device if torch.cuda.is_available() else "cpu"
        )

        model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Set pad token
        pad_token_mode = config.get("model", "pad_token_mode", "eos")
        if pad_token_mode == "eos":
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

    def apply_sampling_filters(self, logits: torch.Tensor, temperature: float = 1.0,
                              top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        """Apply temperature, top_k, and top_p filtering to logits."""
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top_k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Apply top_p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Convert back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_stems(self, input_ids: List[int], num_stems: int, stem_length: int,
                      temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> List[Tuple[List[int], torch.Tensor]]:
        """
        Generate multiple continuation stems using batched inference.

        Args:
            input_ids: List of input token IDs
            num_stems: Number of stems to generate
            stem_length: Length of each stem
            temperature: Sampling temperature
            top_k: Keep only top k tokens (0 = no filtering)
            top_p: Nucleus sampling threshold (1.0 = no filtering)

        Returns:
            List of (generated_token_ids, final_hidden_state) tuples
        """
        config = get_config()
        batch_size = config.getint("generation", "batch_size", 50)

        # Convert input list to tensor for model
        input_tensor = torch.tensor([input_ids], device=self.device)
        stems = []

        with torch.no_grad():
            for batch_start in range(0, num_stems, batch_size):
                batch_end = min(batch_start + batch_size, num_stems)
                current_batch_size = batch_end - batch_start

                # Create batch by repeating input_tensor
                batch_input = input_tensor.repeat(current_batch_size, 1)

                # Generate stems for this batch
                for step in range(stem_length):
                    outputs = self.model(batch_input, output_hidden_states=True)
                    logits = outputs.logits[:, -1, :]

                    # Apply sampling filters
                    filtered_logits = []
                    for i in range(current_batch_size):
                        filtered = self.apply_sampling_filters(
                            logits[i:i+1], temperature, top_k, top_p
                        )
                        filtered_logits.append(filtered)

                    filtered_logits = torch.cat(filtered_logits, dim=0)
                    probs = torch.softmax(filtered_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, 1)
                    batch_input = torch.cat([batch_input, next_tokens], dim=1)

                # Extract generated portions as lists and final hidden states
                for i in range(current_batch_size):
                    generated_tokens = batch_input[i, input_tensor.size(1):].tolist()
                    final_hidden = outputs.hidden_states[-1][i, -1, :]
                    stems.append((generated_tokens, final_hidden))

        return stems

    def get_sampling_info(self, temperature: float, top_k: int, top_p: float) -> str:
        """Get a human-readable description of current sampling settings."""
        parts = []
        if temperature != 1.0:
            if temperature < 0.5:
                parts.append(f"very conservative (temp={temperature:.2f})")
            elif temperature < 0.8:
                parts.append(f"conservative (temp={temperature:.2f})")
            elif temperature > 1.5:
                parts.append(f"very creative (temp={temperature:.2f})")
            elif temperature > 1.2:
                parts.append(f"creative (temp={temperature:.2f})")
            else:
                parts.append(f"temp={temperature:.2f}")

        if top_k > 0:
            parts.append(f"top-{top_k}")

        if top_p < 1.0:
            parts.append(f"nucleus-{top_p:.2f}")

        return "Sampling: " + (", ".join(parts) if parts else "default")