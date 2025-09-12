import configparser
from pathlib import Path
from typing import Optional


class Config:
    """Configuration loader with automatic path resolution."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = self._find_config_file()
        
        self.config_path = Path(config_path)
        self.parser = configparser.ConfigParser()
        self.load()

    def _find_config_file(self) -> str:
        """Find config.ini in the project structure."""
        # Start from this file's location
        current = Path(__file__).parent

        # Look for config.ini in this directory first
        config_file = current / "config.ini"
        if config_file.exists():
            return str(config_file)

        # Search upward in the directory tree
        for parent in current.parents:
            config_file = parent / "config" / "config.ini"
            if config_file.exists():
                return str(config_file)

            config_file = parent / "config.ini"
            if config_file.exists():
                return str(config_file)

        raise FileNotFoundError("config.ini not found in project structure")

    def load(self):
        """Load configuration from file."""
        self.parser.read(self.config_path)

    def get(self, section: str, key: str) -> str:
        return self.parser.get(section, key)

    def getint(self, section: str, key: str) -> int:
        return self.parser.getint(section, key)

    def getfloat(self, section: str, key: str) -> float:
        return self.parser.getfloat(section, key)

    def getboolean(self, section: str, key: str) -> bool:
        return self.parser.getboolean(section, key)

    @property
    def model_name(self) -> str:
        return self.get("model", "model_name")

    @property
    def device(self) -> str:
        return self.get("model", "device")

    @property
    def max_token_depth(self) -> int:
        return self.getint("generation", "max_token_depth")

    @property
    def threshold(self) -> float:
        return self.getfloat("generation", "threshold")

    @property
    def top_k(self) -> int:
        return self.getint("generation", "top_k")

    @property
    def top_p(self) -> float:
        return self.getfloat("generation", "top_p")

    @property
    def output_dir(self) -> str:
        return self.get("visualization", "output_dir")

    @property
    def normalize_pre_delta(self) -> bool:
        return self.getboolean("embeddings", "normalize_pre_delta")

    @property
    def normalize_post_delta(self) -> bool:
        return self.getboolean("embeddings", "normalize_post_delta")

    @property
    def batch_size(self) -> int:
        return self.getint("embeddings", "batch_size")

    @property
    def custom_pooling(self) -> str:
        return self.get("embeddings", "custom_pooling")

    @property
    def stem_length(self) -> int:
        return self.getint("generation", "stem_length")

    @property
    def max_proportion_gap(self) -> float:
        return self.getfloat("generation", "max_proportion_gap")

    @property
    def num_stems(self) -> int:
        return self.getint("generation", "num_stems")

    @property
    def temperature(self) -> float:
        return self.getfloat("generation", "temperature")

    @property
    def entropy_budget(self) -> float:
        return self.getfloat("generation", "entropy_budget")

    @property
    def max_entropy_depth(self) -> float:
        return self.getfloat("generation", "max_entropy_depth")


_config_instance = None

def get_config() -> Config:
    """Get global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance