import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from models.model_interface import ModelInterface
from config.config import get_config
from typing import List, Tuple, Any


class GPT2Interface(ModelInterface):
    """Interface wrapper for GPT-2 models with batched inference optimization."""

    def __init__(self, model_name: str, device: str = None):
        config = get_config()
        self.set_seed(config.getint("generation", "seed"))

        self.device = torch.device(
            device or config.device if torch.cuda.is_available() else "cpu"
        )

        model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Set pad token to be different from eos token to avoid attention mask issues
        self.tokenizer.pad_token = self.tokenizer.unk_token  # Use unk token as pad
        if self.tokenizer.pad_token is None:
            # If no unk token, add a new pad token
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

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
                      temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> List[Tuple[List[int], Any]]:
        """
        Generate multiple continuation stems using HuggingFace's optimized generate.

        Args:
            input_ids: List of input token IDs
            num_stems: Number of stems to generate
            stem_length: Length of each stem
            temperature: Sampling temperature
            top_k: Keep only top k tokens (0 = no filtering)
            top_p: Nucleus sampling threshold (1.0 = no filtering)

        Returns:
            List of generated token IDs (without hidden states since they're not used)
        """
        # CRITICAL: Generate in smaller batches to avoid memory explosion
        # HuggingFace's generate with num_return_sequences creates a huge attention tensor
        config = get_config()
        stems = []
        batch_size = config.getint("generation", "batch_size")

        with torch.no_grad():
            for batch_start in range(0, num_stems, batch_size):

                # Convert input to tensor and create attention mask
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=self.device)

                # Generate batch of stems
                outputs = self.model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=stem_length,
                    min_new_tokens=stem_length,  # Force exact length
                    num_return_sequences=batch_size,
                    temperature=temperature if temperature > 0 else 1e-7,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p < 1.0 else None,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=False,
                    use_cache=True,
                )

                # Extract the generated portions
                input_length = input_tensor.size(1)
                generated_portions = outputs[:, input_length:]

                # Convert to list and add to results
                for i in range(batch_size):
                    generated_tokens = generated_portions[i].tolist()
                    stems.append(generated_tokens)

                # memory cleanup
                del outputs
                del generated_portions
                del input_tensor
                del attention_mask

            # Force GPU memory cleanup after all batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return stems
