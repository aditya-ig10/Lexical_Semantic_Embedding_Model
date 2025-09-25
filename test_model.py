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


class TestPositionalEncoding(unittest.TestCase):
    """Test cases for PositionalEncoding module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embed_dim = 256
        self.max_len = 1000
        self.pos_encoding = PositionalEncoding(self.embed_dim, self.max_len)
    
    def test_positional_encoding_shape(self):
        """Test positional encoding output shape."""
        seq_len = 50
        batch_size = 4
        
        x = torch.randn(seq_len, batch_size, self.embed_dim)
        output = self.pos_encoding(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic."""
        seq_len = 30
        batch_size = 2
        
        x = torch.randn(seq_len, batch_size, self.embed_dim)
        output1 = self.pos_encoding(x)
        output2 = self.pos_encoding(x)
        
        torch.testing.assert_close(output1, output2)


class TestBiLSTMEncoder(unittest.TestCase):
    """Test cases for BiLSTMEncoder module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.embed_dim = 128
        self.hidden_dim = 256
        self.num_layers = 2
        self.batch_size = 4
        self.seq_len = 32
        
        self.encoder = BiLSTMEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.1,
            bidirectional=True,
            use_attention=True
        )
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        self.assertEqual(self.encoder.vocab_size, self.vocab_size)
        self.assertEqual(self.encoder.embed_dim, self.embed_dim)
        self.assertEqual(self.encoder.hidden_dim, self.hidden_dim)
        self.assertTrue(self.encoder.bidirectional)
        self.assertTrue(self.encoder.use_attention)
    
    def test_encoder_forward_pass(self):
        """Test encoder forward pass."""
        input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        
        outputs = self.encoder(input_ids, attention_mask)
        
        self.assertIn('last_hidden_state', outputs)
        self.assertIn('pooler_output', outputs)
        
        # Check shapes
        self.assertEqual(
            outputs['last_hidden_state'].shape,
            (self.batch_size, self.seq_len, self.embed_dim)
        )
        self.assertEqual(
            outputs['pooler_output'].shape,
            (self.batch_size, self.embed_dim)
        )
    
    def test_encoder_without_attention(self):
        """Test encoder without attention mechanism."""
        encoder = BiLSTMEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            use_attention=False
        )
        
        input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        outputs = encoder(input_ids)
        
        self.assertIn('last_hidden_state', outputs)
        self.assertIn('pooler_output', outputs)
    
    def test_encoder_with_pretrained_embeddings(self):
        """Test encoder with pretrained embeddings."""
        pretrained_embeddings = torch.randn(self.vocab_size, self.embed_dim)
        
        encoder = BiLSTMEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=True
        )
        
        # Check that embeddings are frozen
        self.assertFalse(encoder.embedding.weight.requires_grad)
        
        # Check that embeddings match
        torch.testing.assert_close(
            encoder.embedding.weight, 
            pretrained_embeddings
        )

