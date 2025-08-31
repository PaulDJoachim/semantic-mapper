import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.model_interface import ModelInterface
from config.config import get_config
from typing import List, Tuple, Any


class GenericTransformer(ModelInterface):
    """Interface wrapper for transformer models"""

    def __init__(self, model_name: str, device: str = None):
        config = get_config()
        self.set_seed(config.getint("generation", "seed"))

        self.device = torch.device(
            device or config.device if torch.cuda.is_available() else "cpu"
        )

        model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.unk_token  # Use unk token as pad

        # Move model to the specified device
        self.model = self.model.to(self.device)
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
                      temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Generate multiple continuation stems with per-token entropy data.

        Args:
            input_ids: List of input token IDs
            num_stems: Number of stems to generate
            stem_length: Length of each stem
            temperature: Sampling temperature
            top_k: Keep only top k tokens (0 = no filtering)
            top_p: Nucleus sampling threshold (1.0 = no filtering)

        Returns:
            Tuple of (generated_token_lists, entropy_lists)
        """
        config = get_config()
        stems = []
        entropies = []
        batch_size = config.getint("generation", "batch_size")

        with torch.no_grad():
            for batch_start in range(0, num_stems, batch_size):
                # Ensure we don't exceed num_stems
                current_batch_size = min(batch_size, num_stems - batch_start)

                # Convert input to tensor and create attention mask
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=self.device)

                # Generate batch of stems with scores
                outputs = self.model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=stem_length,
                    min_new_tokens=stem_length,  # Force exact length
                    num_return_sequences=current_batch_size,  # Use actual batch size
                    temperature=temperature if temperature > 0 else 1e-7,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p < 1.0 else None,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True,
                )

                # Extract the generated portions
                input_length = input_tensor.size(1)
                generated_portions = outputs.sequences[:, input_length:]

                # Calculate entropy from scores
                batch_entropies = self._calculate_entropies(outputs.scores)

                # Convert to lists and add to results
                for i in range(current_batch_size):
                    generated_tokens = generated_portions[i].tolist()
                    token_entropies = batch_entropies[i].tolist()
                    stems.append(generated_tokens)
                    entropies.append(token_entropies)

                # Memory cleanup
                del outputs
                del generated_portions
                del input_tensor
                del attention_mask

            # Force GPU memory cleanup after all batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return stems, entropies

    def _calculate_entropies(self, scores: Tuple[torch.Tensor]) -> torch.Tensor:
        """Calculate per-token entropy from generation scores."""
        # scores is a tuple of tensors, one for each generation step
        # Each tensor has shape (batch_size, vocab_size)
        entropies_per_step = []

        for step_scores in scores:
            # Convert logits to probabilities
            probs = torch.softmax(step_scores, dim=-1)
            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            entropies_per_step.append(entropy)

        # Stack to get shape (num_steps, batch_size)
        entropies = torch.stack(entropies_per_step, dim=0)
        # Transpose to get shape (batch_size, num_steps)
        return entropies.transpose(0, 1)