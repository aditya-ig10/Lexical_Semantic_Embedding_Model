"""
Model Configuration for Lexical Semantic Embedding Model.

This module contains all model architecture and hyperparameter configurations.

Author: AI Assistant
Date: September 2025
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Configuration class for model architecture and hyperparameters.
    
    This class contains all the settings needed to initialize and configure
    the Lexical Semantic Embedding Model.
    """
    
    # Vocabulary and embedding settings
    vocab_size: int = 30000
    embed_dim: int = 300
    max_seq_length: int = 128
    pad_token_id: int = 0
    unk_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    
    # BiLSTM encoder settings
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    use_layer_norm: bool = True
    
    # Attention mechanism settings
    use_attention: bool = True
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    attention_hidden_dim: Optional[int] = None  # If None, uses embed_dim
    
    # Positional encoding settings
    use_positional_encoding: bool = True
    max_position_embeddings: int = 512
    
    # Similarity function settings
    similarity_function: str = "learned"  # Options: cosine, euclidean, manhattan, learned
    similarity_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    similarity_dropout: float = 0.2
    
    # Output settings
    output_dim: int = 1
    output_activation: str = "tanh"  # Options: tanh, sigmoid, relu, none
    
    # Embedding initialization settings
    use_pretrained_embeddings: bool = False
    pretrained_embeddings_path: Optional[str] = None
    freeze_embeddings: bool = False
    embedding_init_range: float = 0.1
    
    # Model regularization
    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0
    
    # Advanced settings
    use_residual_connections: bool = True
    use_highway_networks: bool = False
    highway_layers: int = 2
    
    # Activation functions
    lstm_activation: str = "tanh"
    attention_activation: str = "relu"
    
    # Numerical stability
    eps: float = 1e-12
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate vocabulary settings
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.embed_dim > 0, "embed_dim must be positive"
        assert self.max_seq_length > 0, "max_seq_length must be positive"
        
        # Validate token IDs
        assert 0 <= self.pad_token_id < self.vocab_size, "Invalid pad_token_id"
        assert 0 <= self.unk_token_id < self.vocab_size, "Invalid unk_token_id"
        assert 0 <= self.cls_token_id < self.vocab_size, "Invalid cls_token_id"
        assert 0 <= self.sep_token_id < self.vocab_size, "Invalid sep_token_id"
        
        # Validate LSTM settings
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        
        # Validate attention settings
        if self.use_attention:
            assert self.num_attention_heads > 0, "num_attention_heads must be positive"
            assert self.embed_dim % self.num_attention_heads == 0, \
                "embed_dim must be divisible by num_attention_heads"
            assert 0 <= self.attention_dropout <= 1, "attention_dropout must be between 0 and 1"
        
        # Validate similarity function
        valid_similarity_functions = ["cosine", "euclidean", "manhattan", "learned"]
        assert self.similarity_function in valid_similarity_functions, \
            f"similarity_function must be one of {valid_similarity_functions}"
        
        # Validate output settings
        assert self.output_dim > 0, "output_dim must be positive"
        valid_activations = ["tanh", "sigmoid", "relu", "none"]
        assert self.output_activation in valid_activations, \
            f"output_activation must be one of {valid_activations}"
        
        # Validate pretrained embeddings
        if self.use_pretrained_embeddings:
            assert self.pretrained_embeddings_path is not None, \
                "pretrained_embeddings_path must be provided when use_pretrained_embeddings=True"
            assert Path(self.pretrained_embeddings_path).exists(), \
                f"Pretrained embeddings file not found: {self.pretrained_embeddings_path}"
    
    def get_attention_hidden_dim(self) -> int:
        """Get attention hidden dimension."""
        return self.attention_hidden_dim or self.embed_dim
    
    def get_lstm_output_dim(self) -> int:
        """Get LSTM output dimension."""
        return self.hidden_dim * (2 if self.bidirectional else 1)
    
    def get_similarity_input_dim(self) -> int:
        """Get input dimension for similarity computation."""
        if self.similarity_function == "learned":
            # Concatenation of embeddings, absolute difference, element-wise product, and cosine similarity
            return self.embed_dim * 4
        else:
            return self.embed_dim
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different model sizes
@dataclass
class SmallModelConfig(ModelConfig):
    """Small model configuration for development and testing."""
    vocab_size: int = 10000
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    num_attention_heads: int = 4
    similarity_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])


@dataclass
class MediumModelConfig(ModelConfig):
    """Medium model configuration for standard tasks."""
    vocab_size: int = 30000
    embed_dim: int = 300
    hidden_dim: int = 512
    num_layers: int = 2
    num_attention_heads: int = 8
    similarity_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])


@dataclass
class LargeModelConfig(ModelConfig):
    """Large model configuration for complex tasks."""
    vocab_size: int = 50000
    embed_dim: int = 512
    hidden_dim: int = 768
    num_layers: int = 3
    num_attention_heads: int = 12
    similarity_hidden_dims: List[int] = field(default_factory=lambda: [768, 512, 256])
    dropout: float = 0.2
    attention_dropout: float = 0.05


@dataclass
class BERTLikeConfig(ModelConfig):
    """BERT-like configuration for transformer-style architecture."""
    vocab_size: int = 30522
    embed_dim: int = 768
    hidden_dim: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    max_seq_length: int = 512
    max_position_embeddings: int = 512
    use_positional_encoding: bool = True
    use_layer_norm: bool = True
    dropout: float = 0.1
    attention_dropout: float = 0.1


# Model configuration factory
def get_model_config(config_name: str = "medium") -> ModelConfig:
    """
    Get predefined model configuration.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Model configuration instance
    """
    configs = {
        "small": SmallModelConfig,
        "medium": MediumModelConfig,
        "large": LargeModelConfig,
        "bert": BERTLikeConfig
    }
    
    if config_name not in configs:
        available_configs = list(configs.keys())
        raise ValueError(f"Unknown config name: {config_name}. Available: {available_configs}")
    
    return configs[config_name]()


def create_custom_config(**kwargs) -> ModelConfig:
    """
    Create custom model configuration.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        Custom model configuration
    """
    base_config = MediumModelConfig()
    
    # Update configuration with provided parameters
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return base_config


# Configuration for different similarity functions
SIMILARITY_CONFIGS = {
    "cosine": {
        "similarity_function": "cosine",
        "similarity_hidden_dims": [],
        "output_activation": "tanh"
    },
    "euclidean": {
        "similarity_function": "euclidean",
        "similarity_hidden_dims": [],
        "output_activation": "none"
    },
    "manhattan": {
        "similarity_function": "manhattan",
        "similarity_hidden_dims": [],
        "output_activation": "none"
    },
    "learned": {
        "similarity_function": "learned",
        "similarity_hidden_dims": [512, 256],
        "similarity_dropout": 0.2,
        "output_activation": "tanh"
    }
}


def get_similarity_config(similarity_type: str, base_config: Optional[ModelConfig] = None) -> ModelConfig:
    """
    Get model configuration optimized for specific similarity function.
    
    Args:
        similarity_type: Type of similarity function
        base_config: Base configuration to modify (if None, uses medium config)
        
    Returns:
        Model configuration optimized for similarity function
    """
    if similarity_type not in SIMILARITY_CONFIGS:
        available_types = list(SIMILARITY_CONFIGS.keys())
        raise ValueError(f"Unknown similarity type: {similarity_type}. Available: {available_types}")
    
    # Start with base configuration
    if base_config is None:
        config = MediumModelConfig()
    else:
        config = base_config
    
    # Apply similarity-specific settings
    similarity_settings = SIMILARITY_CONFIGS[similarity_type]
    for key, value in similarity_settings.items():
        setattr(config, key, value)
    
    return config


# Export commonly used configurations
def get_config() -> ModelConfig:
    """Get default model configuration."""
    return MediumModelConfig()


if __name__ == "__main__":
    # Example usage and testing
    print("Model Configuration Examples:")
    print("=" * 50)
    
    # Default medium configuration
    config = get_model_config("medium")
    print(f"Medium config - Embed dim: {config.embed_dim}, Hidden dim: {config.hidden_dim}")
    
    # Small configuration for testing
    small_config = get_model_config("small")
    print(f"Small config - Embed dim: {small_config.embed_dim}, Hidden dim: {small_config.hidden_dim}")
    
    # Large configuration for production
    large_config = get_model_config("large")
    print(f"Large config - Embed dim: {large_config.embed_dim}, Hidden dim: {large_config.hidden_dim}")
    
    # Custom configuration
    custom_config = create_custom_config(
        embed_dim=256,
        hidden_dim=512,
        similarity_function="learned"
    )
    print(f"Custom config - Embed dim: {custom_config.embed_dim}, Similarity: {custom_config.similarity_function}")
    
    # Similarity-specific configurations
    for sim_type in ["cosine", "learned"]:
        sim_config = get_similarity_config(sim_type)
        print(f"{sim_type.capitalize()} similarity config - Function: {sim_config.similarity_function}")
    
    # Test configuration validation
    try:
        invalid_config = ModelConfig(vocab_size=-1)  # This should raise an error
    except AssertionError as e:
        print(f"Validation works: {e}")
    
    print("\n✅ All configuration examples completed successfully!")
