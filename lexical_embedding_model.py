"""
Lexical Semantic Embedding Model with BiLSTM and Attention Mechanism

This module implements a production-ready semantic embedding model using PyTorch
with BiLSTM layers and multi-head attention for capturing lexical semantics.

Author: AI Assistant
Date: September 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
import math
from pathlib import Path
import json
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism for capturing different types of relationships
    in the sequence data.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize Multi-Head Attention layer.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Store residual connection
        residual = x
        
        # Generate Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Apply output projection
        output = self.output(context)
        
        # Add residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding to inject sequence order information.
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, embed_dim)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder with optional attention mechanism.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
        num_attention_heads: int = 8,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False
    ):
        """
        Initialize BiLSTM encoder.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
            num_attention_heads: Number of attention heads
            pretrained_embeddings: Pretrained embedding matrix
            freeze_embeddings: Whether to freeze embedding weights
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
