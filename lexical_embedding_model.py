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
            
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = MultiHeadAttention(
                lstm_output_dim, 
                num_attention_heads, 
                dropout
            )
        
        # Output projection
        self.output_projection = nn.Linear(lstm_output_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the encoder.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Dictionary containing:
                - last_hidden_state: Final hidden states
                - pooler_output: Pooled representation
                - attention_weights: Attention weights (if attention is used)
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        
        # Add positional encoding
        embeddings = embeddings.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        embeddings = self.pos_encoding(embeddings)
        embeddings = embeddings.transpose(0, 1)  # (batch_size, seq_len, embed_dim)
        
        # Pack padded sequences if attention mask is provided
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            embeddings = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )
        
        # LSTM
        lstm_output, (hidden, cell) = self.lstm(embeddings)
        
        # Unpack if we packed
        if attention_mask is not None:
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_output, batch_first=True
            )
        
        # Apply attention if enabled
        attention_weights = None
        if self.use_attention:
            # Create attention mask for padded positions
            if attention_mask is not None:
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(-1, lstm_output.size(1), -1, -1)
            else:
                attn_mask = None
                
            lstm_output = self.attention(lstm_output, attn_mask)
        
        # Output projection
        last_hidden_state = self.output_projection(lstm_output)
        last_hidden_state = self.layer_norm(last_hidden_state)
        
        # Pooling: mean pooling over non-padded positions
        if attention_mask is not None:
            # Expand attention mask to match hidden state dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            # Apply mask and compute mean
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooler_output = sum_embeddings / sum_mask
        else:
            pooler_output = torch.mean(last_hidden_state, dim=1)
        
        return {
            'last_hidden_state': last_hidden_state,
            'pooler_output': pooler_output,
            'attention_weights': attention_weights
        }


class LexicalSemanticEmbeddingModel(nn.Module):
    """
    Complete Lexical Semantic Embedding Model with siamese architecture
    for semantic similarity tasks.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_attention_heads: int = 8,
        similarity_function: str = 'cosine',
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        output_dim: int = 1
    ):
        """
        Initialize the complete model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_attention_heads: Number of attention heads
            similarity_function: Similarity function ('cosine', 'euclidean', 'manhattan')
            pretrained_embeddings: Pretrained embedding matrix
            freeze_embeddings: Whether to freeze embedding weights
            output_dim: Output dimension (1 for similarity score)
        """
        super().__init__()
        
        self.similarity_function = similarity_function
