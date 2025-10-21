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


class TestLexicalSemanticEmbeddingModel(unittest.TestCase):
    """Test cases for the complete LexicalSemanticEmbeddingModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.embed_dim = 128
        self.hidden_dim = 256
        self.batch_size = 4
        self.seq_len = 32
        
        self.model = LexicalSemanticEmbeddingModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            similarity_function='cosine'
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.similarity_function, 'cosine')
        self.assertEqual(self.model.output_dim, 1)
        self.assertIsInstance(self.model.encoder, BiLSTMEncoder)
    
    def test_model_forward_pass(self):
        """Test complete model forward pass."""
        input_ids_1 = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        input_ids_2 = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        attention_mask_1 = torch.ones(self.batch_size, self.seq_len)
        attention_mask_2 = torch.ones(self.batch_size, self.seq_len)
        
        outputs = self.model(
            input_ids_1=input_ids_1,
            input_ids_2=input_ids_2,
            attention_mask_1=attention_mask_1,
            attention_mask_2=attention_mask_2
        )
        
        self.assertIn('similarity', outputs)
        self.assertIn('embedding1', outputs)
        self.assertIn('embedding2', outputs)
        
        # Check shapes
        self.assertEqual(outputs['similarity'].shape, (self.batch_size,))
        self.assertEqual(outputs['embedding1'].shape, (self.batch_size, self.embed_dim))
        self.assertEqual(outputs['embedding2'].shape, (self.batch_size, self.embed_dim))
        
        # Check similarity scores are reasonable
        similarities = outputs['similarity']
        self.assertFalse(torch.isnan(similarities).any())
        self.assertFalse(torch.isinf(similarities).any())
    
    def test_different_similarity_functions(self):
        """Test different similarity functions."""
        similarity_functions = ['cosine', 'euclidean', 'manhattan', 'learned']
        
        for sim_func in similarity_functions:
            with self.subTest(similarity_function=sim_func):
                model = LexicalSemanticEmbeddingModel(
                    vocab_size=self.vocab_size,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    similarity_function=sim_func
                )
                
                input_ids_1 = torch.randint(1, self.vocab_size, (2, 16))
                input_ids_2 = torch.randint(1, self.vocab_size, (2, 16))
                
                outputs = model(input_ids_1, input_ids_2)
                
                self.assertEqual(outputs['similarity'].shape, (2,))
                self.assertFalse(torch.isnan(outputs['similarity']).any())
    
    def test_sentence_embedding(self):
        """Test single sentence embedding."""
        input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        
        embedding = self.model.get_sentence_embedding(input_ids, attention_mask)
        
        self.assertEqual(embedding.shape, (self.batch_size, self.embed_dim))
        self.assertFalse(torch.isnan(embedding).any())
    
    def test_model_save_and_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model"
            
            # Save model
            self.model.save_model(save_path)
            
            # Check files exist
            self.assertTrue((save_path / "model.pt").exists())
            self.assertTrue((save_path / "config.json").exists())
            
            # Load model
            loaded_model = LexicalSemanticEmbeddingModel.load_model(save_path)
            
            # Test that loaded model works
            input_ids = torch.randint(1, self.vocab_size, (2, 16))
            
            original_output = self.model.get_sentence_embedding(input_ids)
            loaded_output = loaded_model.get_sentence_embedding(input_ids)
            
            torch.testing.assert_close(original_output, loaded_output)
    
    def test_model_export_onnx(self):
        """Test ONNX export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "model.onnx"
            
            example_input = {
                'input_ids_1': torch.randint(1, self.vocab_size, (1, 16)),
                'input_ids_2': torch.randint(1, self.vocab_size, (1, 16)),
                'attention_mask_1': torch.ones(1, 16),
                'attention_mask_2': torch.ones(1, 16)
            }
            
            # Export to ONNX
            self.model.export_to_onnx(export_path, example_input)
            
            # Check file exists
            self.assertTrue(export_path.exists())
    
    def test_compute_similarity_edge_cases(self):
        """Test similarity computation edge cases."""
        # Test identical embeddings
        embedding = torch.randn(2, self.embed_dim)
        similarity = self.model.compute_similarity(embedding, embedding)
        
        # For cosine similarity with identical vectors, should be high
        self.assertGreater(similarity.mean().item(), 0.5)
        
        # Test orthogonal embeddings
        embedding1 = torch.zeros(2, self.embed_dim)
        embedding1[:, :self.embed_dim//2] = 1.0
        
        embedding2 = torch.zeros(2, self.embed_dim)
        embedding2[:, self.embed_dim//2:] = 1.0
        
        similarity = self.model.compute_similarity(embedding1, embedding2)
        
        # Should be close to 0 for orthogonal vectors
        self.assertLess(abs(similarity.mean().item()), 0.5)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_model_from_config(self):
        """Test model creation from configuration."""
        config = {
            'vocab_size': 1000,
            'embed_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'similarity_function': 'learned'
        }
        
        model = create_model_from_config(config)
        
        self.assertIsInstance(model, LexicalSemanticEmbeddingModel)
        self.assertEqual(model.encoder.vocab_size, 1000)
        self.assertEqual(model.similarity_function, 'learned')


class TestModelTraining(unittest.TestCase):
    """Test model training capabilities."""
    
    def setUp(self):
        """Set up training test fixtures."""
        self.model = LexicalSemanticEmbeddingModel(
            vocab_size=1000,
            embed_dim=64,  # Smaller for faster testing
            hidden_dim=128,
            num_layers=1
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def test_model_training_step(self):
        """Test a single training step."""
        # Create dummy data
        batch_size = 4
        seq_len = 16
        
        input_ids_1 = torch.randint(1, 1000, (batch_size, seq_len))
        input_ids_2 = torch.randint(1, 1000, (batch_size, seq_len))
        targets = torch.rand(batch_size)  # Random similarity scores
        
        # Forward pass
        outputs = self.model(input_ids_1, input_ids_2)
        loss = self.criterion(outputs['similarity'], targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Check that loss is computed correctly
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        input_ids_1 = torch.randint(1, 1000, (2, 8))
        input_ids_2 = torch.randint(1, 1000, (2, 8))
        targets = torch.rand(2)
        
        outputs = self.model(input_ids_1, input_ids_2)
        loss = self.criterion(outputs['similarity'], targets)
        loss.backward()
        
        # Check that gradients exist
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")



# fix: tmp_path for tokenizer cache, was hardcoded /data
