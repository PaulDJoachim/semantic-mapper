import configparser
from pathlib import Path
from typing import Union, Optional


class Config:
    """Simple configuration loader for DIA settings."""

    def __init__(self, config_path: str = "config.ini"):
        self.config_path = Path(config_path)
        self.parser = configparser.ConfigParser()
        self.load()

    def load(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.parser.read(self.config_path)

    def get(self, section: str, key: str, fallback: Optional[str] = None) -> str:
        """Get string value from config."""
        return self.parser.get(section, key, fallback=fallback)

    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Get integer value from config."""
        return self.parser.getint(section, key, fallback=fallback)

    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Get float value from config."""
        return self.parser.getfloat(section, key, fallback=fallback)

    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get boolean value from config."""
        return self.parser.getboolean(section, key, fallback=fallback)

    # Convenience properties for common settings
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
    def template_path(self) -> str:
        return self.get("visualization", "template_path", "tree_template.html")

    @property
    def compress_linear(self) -> bool:
        return self.getboolean("visualization", "compress_linear", True)


# Global config instance
_config_instance = None


def get_config(config_path: str = "config.ini") -> Config:
    """Get global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance