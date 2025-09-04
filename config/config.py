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

    def get(self, section: str, key: str, fallback: Optional[str] = None) -> str:
        return self.parser.get(section, key, fallback=fallback)

    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        return self.parser.getint(section, key, fallback=fallback)

    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        return self.parser.getfloat(section, key, fallback=fallback)

    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        return self.parser.getboolean(section, key, fallback=fallback)

    @property
    def model_name(self) -> str:
        return self.get("model", "model_name", "gpt2")

    @property
    def device(self) -> str:
        return self.get("model", "device", "cuda")

    @property
    def max_depth(self) -> int:
        return self.getint("generation", "max_depth", 30)

    @property
    def threshold(self) -> float:
        return self.getfloat("generation", "threshold", 0.1)

    @property
    def top_k(self) -> int:
        return self.getint("generation", "top_k", 5)

    @property
    def top_p(self) -> float:
        return self.getfloat("generation", "top_p", 1.0)

    @property
    def output_dir(self) -> str:
        return self.get("visualization", "output_dir", "./output")

    @property
    def embed_full_branch(self) -> bool:
        return self.getboolean("embeddings", "embed_full_branch", False)


_config_instance = None

def get_config() -> Config:
    """Get global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance