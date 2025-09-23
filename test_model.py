"""
Unit tests for the Lexical Semantic Embedding Model.

This module contains comprehensive tests for all model components
including the BiLSTM encoder, attention mechanism, and similarity functions.

Author: AI Assistant
Date: September 2025
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import tempfile
import shutil
from pathlib import Path
import json

# Import model components
from lexical_embedding_model import (
    LexicalSemanticEmbeddingModel,
    BiLSTMEncoder,
    MultiHeadAttention,
    PositionalEncoding,
    create_model_from_config
)


class TestMultiHeadAttention(unittest.TestCase):
    """Test cases for MultiHeadAttention module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embed_dim = 256
        self.num_heads = 8
        self.seq_len = 32
        self.batch_size = 4
        
        self.attention = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.1
        )
    
    def test_attention_initialization(self):
        """Test attention layer initialization."""
        self.assertEqual(self.attention.embed_dim, self.embed_dim)
        self.assertEqual(self.attention.num_heads, self.num_heads)
        self.assertEqual(self.attention.head_dim, self.embed_dim // self.num_heads)
    
    def test_attention_forward_pass(self):
        """Test attention forward pass."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        # Forward pass without mask
        output = self.attention(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_attention_with_mask(self):
        """Test attention with padding mask."""
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        mask = torch.ones(self.batch_size, 1, 1, self.seq_len)
        # Mask out last 10 positions
        mask[:, :, :, -10:] = 0
        
        output = self.attention(x, mask)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
    
    def test_attention_dimension_mismatch(self):
        """Test attention with invalid dimensions."""
        with self.assertRaises(AssertionError):
            MultiHeadAttention(embed_dim=255, num_heads=8)  # Not divisible

