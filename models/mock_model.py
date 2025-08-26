import random
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
    
    def encode(self, text: str) -> MockTensor:
        """Encode text to mock token tensor."""
        words = text.split()
        token_ids = []
        
        for word in words:
            if word.lower() in self.vocabulary:
                token_ids.append(self.vocabulary.index(word.lower()))
            else:
                token_ids.append(random.randint(0, len(self.vocabulary) - 1))
        
        return MockTensor(token_ids)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        words = []
        for token_id in token_ids:
            if 0 <= token_id < len(self.vocabulary):
                words.append(self.vocabulary[token_id])
            else:
                words.append(f"<unk_{token_id}>")
        
        return " ".join(words)
    
    def generate_stems(self, input_ids: MockTensor, num_stems: int, stem_length: int,
                      temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> List[Tuple[MockTensor, List[float]]]:
        """Generate mock stems based on mode."""
        stems = []
        
        for i in range(num_stems):
            if self.mode == "semantic_clusters":
                tokens = self._generate_clustered_stem(i, num_stems, stem_length)
            elif self.mode == "linear":
                tokens = self._generate_linear_stem(stem_length)
            elif self.mode == "custom":
                tokens = self._generate_custom_stem(input_ids, stem_length)
            else:
                tokens = self._generate_random_stem(stem_length)
            
            hidden_state = mock_randn(768)
            stems.append((MockTensor(tokens), hidden_state))
        
        return stems

    def _generate_clustered_stem(self, stem_index: int, total_stems: int, stem_length: int) -> List[int]:
        """Generate stems designed to form clusters."""
        num_clusters = max(2, min(4, total_stems // 10))
        cluster_id = stem_index % num_clusters
        
        group_names = list(self.semantic_groups.keys())
        target_group = group_names[cluster_id % len(group_names)]
        
        tokens = []
        for _ in range(stem_length):
            if random.random() < 0.7 and target_group in self.semantic_groups:
                word = random.choice(self.semantic_groups[target_group])
                token_id = self.vocabulary.index(word)
            else:
                token_id = random.randint(0, len(self.vocabulary) - 1)
            
            tokens.append(token_id)
        
        return tokens
    
    def _generate_linear_stem(self, stem_length: int) -> List[int]:
        """Generate similar stems for no-branching scenario."""
        neutral_words = self.semantic_groups["neutral"]
        tokens = []
        
        for i in range(stem_length):
            if i == 0 and random.random() < 0.3:
                word = random.choice(neutral_words)
            else:
                word = neutral_words[0]
            
            token_id = self.vocabulary.index(word)
            tokens.append(token_id)
        
        return tokens
    
    def _generate_custom_stem(self, input_ids: MockTensor, stem_length: int) -> List[int]:
        """Generate from predefined patterns."""
        if "test_prompt" in self.custom_responses:
            responses = self.custom_responses["test_prompt"]
            response = random.choice(responses)
            tokens = []
            for word in response[:stem_length]:
                if word.lower() in self.vocabulary:
                    tokens.append(self.vocabulary.index(word.lower()))
                else:
                    tokens.append(random.randint(0, len(self.vocabulary) - 1))
        else:
            tokens = [random.randint(0, len(self.vocabulary) - 1) for _ in range(stem_length)]
        
        return tokens
    
    def _generate_random_stem(self, stem_length: int) -> List[int]:
        """Generate random token sequences."""
        return [random.randint(0, len(self.vocabulary) - 1) for _ in range(stem_length)]
    
    def get_mode_info(self) -> str:
        """Get current mode description."""
        descriptions = {
            "random": "Random token generation",
            "semantic_clusters": "Clusterable semantic stems",
            "linear": "Similar stems (no branching)",
            "custom": "Predefined response patterns"
        }
        return f"MockModel: {descriptions.get(self.mode, 'Unknown')}"