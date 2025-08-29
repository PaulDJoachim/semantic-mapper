import random
import math
from typing import List, Tuple, Any
from models.model_interface import ModelInterface


class MockTensor:
    """Simple tensor-like object to replace torch.Tensor."""

    def __init__(self, data: List[int]):
        self.data = data

    def tolist(self) -> List[int]:
        return self.data.copy()

    def size(self, dim: int = None) -> int:
        if dim is None:
            return len(self.data)
        return len(self.data) if dim == 1 else 1

    def unsqueeze(self, dim: int):
        return MockTensor([self.data])

    def squeeze(self):
        return MockTensor(self.data)

    def item(self) -> int:
        return self.data[0] if self.data else 0

    def repeat(self, *sizes):
        return MockTensor(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index]
        return MockTensor(self.data[index])

    def __add__(self, other):
        """Support concatenation with other MockTensors or lists."""
        if isinstance(other, MockTensor):
            return MockTensor(self.data + other.data)
        elif isinstance(other, list):
            return MockTensor(self.data + other)
        else:
            raise TypeError(f"Cannot add MockTensor and {type(other)}")

    def __radd__(self, other):
        """Support left-side addition (list + MockTensor)."""
        if isinstance(other, list):
            return MockTensor(other + self.data)
        else:
            raise TypeError(f"Cannot add {type(other)} and MockTensor")


def mock_cat(tensors: List[MockTensor], dim: int = 1) -> MockTensor:
    """Concatenate mock tensors."""
    result = []
    for tensor in tensors:
        if isinstance(tensor.data[0], list):
            result.extend(tensor.data[0])
        else:
            result.extend(tensor.data)
    return MockTensor(result)


def mock_randn(*shape) -> List[float]:
    """Generate random floats to simulate hidden states."""
    size = shape[0] if shape else 768
    return [random.gauss(0, 1) for _ in range(size)]


class MockModel(ModelInterface):
    """Torch-free mock model for testing."""

    def __init__(self, mode: str = "semantic_clusters", vocab_size: int = 1000, seed: int = None):
        self.mode = mode
        self.vocab_size = vocab_size

        if seed is not None:
            random.seed(seed)

        self._build_vocabulary()
        self.custom_responses = {}
        if mode == "custom":
            self._load_custom_responses()

    def _build_vocabulary(self):
        """Build semantic vocabulary groups."""
        self.semantic_groups = {
            "positive": ["good", "great", "excellent", "wonderful", "amazing", "brilliant"],
            "negative": ["bad", "terrible", "awful", "horrible", "dreadful", "appalling"],
            "ethical_individual": ["freedom", "autonomy", "rights", "choice", "liberty", "independence"],
            "ethical_collective": ["society", "community", "collective", "group", "shared", "unity"],
            "academic": ["research", "study", "analysis", "evidence", "data", "methodology"],
            "creative": ["imagine", "creative", "artistic", "beautiful", "expression", "inspiration"],
            "technical": ["system", "process", "algorithm", "function", "interface", "implementation"],
            "neutral": ["the", "and", "or", "but", "however", "therefore", "because", "while"]
        }

        self.vocabulary = []
        self.token_to_group = {}

        token_id = 0
        for group, words in self.semantic_groups.items():
            for word in words:
                self.vocabulary.append(word)
                self.token_to_group[token_id] = group
                token_id += 1

        while len(self.vocabulary) < self.vocab_size:
            self.vocabulary.append(f"token_{len(self.vocabulary)}")
            self.token_to_group[len(self.vocabulary) - 1] = "generic"

    def _load_custom_responses(self):
        """Load predefined response patterns."""
        self.custom_responses = {
            "test_prompt": [
                ["This", "leads", "to", "option", "A"],
                ["This", "leads", "to", "option", "B"],
                ["However,", "we", "might", "consider", "C"]
            ]
        }

    def encode(self, text: str) -> List[int]:
        """Encode text to list of token IDs."""
        words = text.split()
        token_ids = []

        for word in words:
            if word.lower() in self.vocabulary:
                token_ids.append(self.vocabulary.index(word.lower()))
            else:
                token_ids.append(random.randint(0, len(self.vocabulary) - 1))

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        words = []
        for token_id in token_ids:
            if 0 <= token_id < len(self.vocabulary):
                words.append(self.vocabulary[token_id])
            else:
                words.append(f"<unk_{token_id}>")

        return " ".join(words)

    def decode_single(self, token_id: int) -> str:
        """Decode single token ID to text."""
        if 0 <= token_id < len(self.vocabulary):
            return self.vocabulary[token_id]
        else:
            return f"<unk_{token_id}>"

    def generate_stems(self, input_ids: List[int], num_stems: int, stem_length: int,
                      temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> Tuple[List[List[int]], List[List[float]]]:
        """Generate mock stems with entropy calculation."""
        stems = []
        entropies = []

        for i in range(num_stems):
            if self.mode == "semantic_clusters":
                tokens, token_entropies = self._generate_clustered_stem(i, num_stems, stem_length, temperature)
            elif self.mode == "linear":
                tokens, token_entropies = self._generate_linear_stem(stem_length, temperature)
            elif self.mode == "custom":
                tokens, token_entropies = self._generate_custom_stem(input_ids, stem_length, temperature)
            else:
                tokens, token_entropies = self._generate_random_stem(stem_length, temperature)

            stems.append(tokens)
            entropies.append(token_entropies)

        return stems, entropies

    def _calculate_mock_entropy(self, selected_token: int, context: str, temperature: float = 1.0) -> float:
        """Calculate mock entropy based on how 'certain' the selection should be."""
        # Base entropy varies by token type and context
        base_entropy = 2.5  # Reasonable default for language models

        # High-frequency words (like "the", "and") should have lower entropy
        if selected_token < len(self.vocabulary):
            token_word = self.vocabulary[selected_token]
            if token_word in self.semantic_groups.get("neutral", []):
                base_entropy = random.uniform(1.0, 2.0)  # More certain
            elif any(token_word in group for group in self.semantic_groups.values()):
                base_entropy = random.uniform(2.0, 3.5)  # Moderate certainty
            else:
                base_entropy = random.uniform(3.0, 5.0)  # Less certain

        # Temperature affects entropy - higher temp means more uncertainty
        temperature_factor = min(temperature, 2.0)  # Cap the effect
        adjusted_entropy = base_entropy * (0.5 + 0.5 * temperature_factor)

        # Add some random noise to make it feel more realistic
        noise = random.uniform(-0.3, 0.3)
        final_entropy = max(0.1, adjusted_entropy + noise)

        return final_entropy

    def _generate_clustered_stem(self, stem_index: int, total_stems: int, stem_length: int, temperature: float) -> Tuple[List[int], List[float]]:
        """Generate stems designed to form clusters with corresponding entropies."""
        num_clusters = max(2, min(4, total_stems // 10))
        cluster_id = stem_index % num_clusters

        group_names = list(self.semantic_groups.keys())
        target_group = group_names[cluster_id % len(group_names)]

        tokens = []
        token_entropies = []

        for pos in range(stem_length):
            if random.random() < 0.7 and target_group in self.semantic_groups:
                word = random.choice(self.semantic_groups[target_group])
                token_id = self.vocabulary.index(word)
                # Lower entropy for in-cluster tokens (more "certain")
                entropy = self._calculate_mock_entropy(token_id, f"cluster_{cluster_id}_pos_{pos}", temperature * 0.8)
            else:
                token_id = random.randint(0, len(self.vocabulary) - 1)
                # Higher entropy for random tokens
                entropy = self._calculate_mock_entropy(token_id, f"random_pos_{pos}", temperature * 1.2)

            tokens.append(token_id)
            token_entropies.append(entropy)

        return tokens, token_entropies

    def _generate_linear_stem(self, stem_length: int, temperature: float) -> Tuple[List[int], List[float]]:
        """Generate similar stems for no-branching scenario with low entropy."""
        neutral_words = self.semantic_groups["neutral"]
        tokens = []
        token_entropies = []

        for i in range(stem_length):
            if i == 0 and random.random() < 0.3:
                word = random.choice(neutral_words)
            else:
                word = neutral_words[0]  # Very predictable

            token_id = self.vocabulary.index(word)
            # Low entropy for predictable sequences
            entropy = self._calculate_mock_entropy(token_id, f"linear_pos_{i}", temperature * 0.5)

            tokens.append(token_id)
            token_entropies.append(entropy)

        return tokens, token_entropies

    def _generate_custom_stem(self, input_ids: List[int], stem_length: int, temperature: float) -> Tuple[List[int], List[float]]:
        """Generate from predefined patterns with contextual entropy."""
        tokens = []
        token_entropies = []

        if "test_prompt" in self.custom_responses:
            responses = self.custom_responses["test_prompt"]
            response = random.choice(responses)

            for i, word in enumerate(response[:stem_length]):
                if word.lower() in self.vocabulary:
                    token_id = self.vocabulary.index(word.lower())
                else:
                    token_id = random.randint(0, len(self.vocabulary) - 1)

                # First tokens in custom responses tend to be more certain
                position_factor = 0.8 if i < 2 else 1.0
                entropy = self._calculate_mock_entropy(token_id, f"custom_pos_{i}", temperature * position_factor)

                tokens.append(token_id)
                token_entropies.append(entropy)

        # Fill remaining positions if needed
        while len(tokens) < stem_length:
            token_id = random.randint(0, len(self.vocabulary) - 1)
            entropy = self._calculate_mock_entropy(token_id, f"custom_fill_{len(tokens)}", temperature)
            tokens.append(token_id)
            token_entropies.append(entropy)

        return tokens, token_entropies

    def _generate_random_stem(self, stem_length: int, temperature: float) -> Tuple[List[int], List[float]]:
        """Generate random token sequences with high entropy."""
        tokens = []
        token_entropies = []

        for i in range(stem_length):
            token_id = random.randint(0, len(self.vocabulary) - 1)
            # Random generation should have higher entropy
            entropy = self._calculate_mock_entropy(token_id, f"random_pos_{i}", temperature * 1.3)
            tokens.append(token_id)
            token_entropies.append(entropy)

        return tokens, token_entropies
    
    def get_mode_info(self) -> str:
        """Get current mode description."""
        descriptions = {
            "random": "Random token generation",
            "semantic_clusters": "Clusterable semantic stems",
            "linear": "Similar stems (no branching)",
            "custom": "Predefined response patterns"
        }
        return f"MockModel: {descriptions.get(self.mode, 'Unknown')}"