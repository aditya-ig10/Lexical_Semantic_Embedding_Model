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
