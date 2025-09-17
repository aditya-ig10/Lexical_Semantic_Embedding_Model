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
        self.output_dim = output_dim
        
        # Shared encoder
        self.encoder = BiLSTMEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            use_attention=True,
            num_attention_heads=num_attention_heads,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings
        )
        
        # Similarity computation layers
        if similarity_function == 'learned':
            # Learned similarity with element-wise operations
            self.similarity_net = nn.Sequential(
                nn.Linear(embed_dim * 4, embed_dim * 2),  # concat, abs_diff, hadamard, cosine
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, output_dim)
            )
        else:
            # Simple projection for traditional similarity functions
            self.similarity_net = nn.Linear(embed_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def encode_sequence(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a sequence to its embedding representation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Sequence embedding
        """
        outputs = self.encoder(input_ids, attention_mask)
        return outputs['pooler_output']
    
    def compute_similarity(
        self, 
        embedding1: torch.Tensor, 
        embedding2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        if self.similarity_function == 'cosine':
            # Cosine similarity
            similarity = F.cosine_similarity(embedding1, embedding2, dim=-1)
            return self.similarity_net(embedding1).squeeze(-1) * similarity
            
        elif self.similarity_function == 'euclidean':
            # Negative euclidean distance (higher = more similar)
            distance = torch.norm(embedding1 - embedding2, p=2, dim=-1)
            return -distance
            
        elif self.similarity_function == 'manhattan':
            # Negative manhattan distance
            distance = torch.norm(embedding1 - embedding2, p=1, dim=-1)
            return -distance
            
        elif self.similarity_function == 'learned':
            # Learned similarity function
            # Concatenate different interaction features
            concat_features = torch.cat([embedding1, embedding2], dim=-1)
            abs_diff = torch.abs(embedding1 - embedding2)
            hadamard = embedding1 * embedding2
            cosine_sim = F.cosine_similarity(embedding1, embedding2, dim=-1, keepdim=True)
            
            features = torch.cat([concat_features, abs_diff, hadamard, cosine_sim], dim=-1)
            return self.similarity_net(features).squeeze(-1)
        
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")
    
    def forward(
        self,
        input_ids_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_1: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for similarity computation.
        
        Args:
            input_ids_1: First sequence token IDs
            input_ids_2: Second sequence token IDs
            attention_mask_1: First sequence attention mask
            attention_mask_2: Second sequence attention mask
            
        Returns:
            Dictionary containing similarity scores and embeddings
        """
        # Encode both sequences
        embedding1 = self.encode_sequence(input_ids_1, attention_mask_1)
        embedding2 = self.encode_sequence(input_ids_2, attention_mask_2)
        
        # Compute similarity
        similarity = self.compute_similarity(embedding1, embedding2)
        
        return {
            'similarity': similarity,
            'embedding1': embedding1,
            'embedding2': embedding2
        }
    
    def get_sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get sentence embedding for a single sequence.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Sentence embedding
        """
        return self.encode_sequence(input_ids, attention_mask)
    
    def save_model(self, path: Union[str, Path], include_config: bool = True):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            include_config: Whether to save model configuration
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), path / "model.pt")
        
        if include_config:
            config = {
                'vocab_size': self.encoder.vocab_size,
                'embed_dim': self.encoder.embed_dim,
                'hidden_dim': self.encoder.hidden_dim,
                'num_layers': self.encoder.num_layers,
                'similarity_function': self.similarity_function,
                'output_dim': self.output_dim
            }
            
            with open(path / "config.json", "w") as f:
                json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(
        cls, 
        path: Union[str, Path], 
        device: Optional[torch.device] = None
    ) -> 'LexicalSemanticEmbeddingModel':
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            device: Device to load the model on
            
        Returns:
            Loaded model
        """
        path = Path(path)
        
        # Load configuration
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(**config)
        
        # Load state dict
        state_dict = torch.load(
            path / "model.pt", 
            map_location=device or torch.device('cpu')
        )
        model.load_state_dict(state_dict)
        
        if device:
            model = model.to(device)
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def export_to_onnx(self, path: Union[str, Path], example_input: Dict[str, torch.Tensor]):
        """
        Export model to ONNX format.
        
        Args:
            path: Path to save ONNX model
            example_input: Example input for tracing
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            torch.onnx.export(
                self,
                (
                    example_input['input_ids_1'],
                    example_input['input_ids_2'],
                    example_input.get('attention_mask_1'),
                    example_input.get('attention_mask_2')
                ),
                str(path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids_1', 'input_ids_2', 'attention_mask_1', 'attention_mask_2'],
                output_names=['similarity', 'embedding1', 'embedding2'],
                dynamic_axes={
                    'input_ids_1': {0: 'batch_size', 1: 'sequence'},
                    'input_ids_2': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask_1': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask_2': {0: 'batch_size', 1: 'sequence'},
                    'similarity': {0: 'batch_size'},
                    'embedding1': {0: 'batch_size'},
                    'embedding2': {0: 'batch_size'}
                }
            )
        
        logger.info(f"Model exported to ONNX format at {path}")


def create_model_from_config(config: Dict) -> LexicalSemanticEmbeddingModel:
    """
    Create model from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    return LexicalSemanticEmbeddingModel(**config)


if __name__ == "__main__":
    # Example usage
    vocab_size = 30000
    model = LexicalSemanticEmbeddingModel(
        vocab_size=vocab_size,
        embed_dim=300,
        hidden_dim=512,
        num_layers=2,
        similarity_function='learned'
    )
    
    # Example input
    batch_size = 4
    seq_len = 32
    
    input_ids_1 = torch.randint(1, vocab_size, (batch_size, seq_len))
    input_ids_2 = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask_1 = torch.ones(batch_size, seq_len)
    attention_mask_2 = torch.ones(batch_size, seq_len)
    
    # Forward pass
    outputs = model(input_ids_1, input_ids_2, attention_mask_1, attention_mask_2)
    
    print(f"Similarity scores shape: {outputs['similarity'].shape}")
    print(f"Embedding 1 shape: {outputs['embedding1'].shape}")
    print(f"Embedding 2 shape: {outputs['embedding2'].shape}")
    
    # Get single sentence embedding
    sentence_embedding = model.get_sentence_embedding(input_ids_1, attention_mask_1)
    print(f"Sentence embedding shape: {sentence_embedding.shape}")

# fix: correct attention scale by sqrt(head_dim), was sqrt(embed_dim)
