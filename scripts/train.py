"""
Training Module for Lexical Semantic Embedding Model.

This module handles the complete training pipeline including data loading,
model training, validation, and checkpointing.

Author: AI Assistant
Date: September 2025
"""

import os
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Import project modules
from lexical_embedding_model import LexicalSemanticEmbeddingModel
from config.model_config import ModelConfig
from config.train_config import TrainingConfig
from config.data_config import DataConfig
from scripts.utils import (
    setup_logging, set_seed, get_device, MetricsTracker, 
    Timer, create_experiment_dir, save_config
)

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class SimilarityDataset(Dataset):
    """
    Dataset class for semantic similarity tasks.
    """
    
    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        scores: List[float],
        tokenizer: Any,
        max_length: int = 128
    ):
        """
        Initialize dataset.
        
        Args:
            sentences1: First sentences
            sentences2: Second sentences
            scores: Similarity scores
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(sentences1) == len(sentences2) == len(scores), \
            "All inputs must have the same length"
    
    def __len__(self) -> int:
        return len(self.sentences1)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with tokenized inputs and target score
        """
        # Tokenize sentences
        tokens1 = self.tokenizer.encode(
            self.sentences1[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        tokens2 = self.tokenizer.encode(
            self.sentences2[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids_1': tokens1['input_ids'].squeeze(0),
            'attention_mask_1': tokens1['attention_mask'].squeeze(0),
            'input_ids_2': tokens2['input_ids'].squeeze(0),
            'attention_mask_2': tokens2['attention_mask'].squeeze(0),
            'score': torch.tensor(self.scores[idx], dtype=torch.float)
        }


class ModelTrainer:
    """
    Trainer class for the Lexical Semantic Embedding Model.
    """
    
    def __init__(
        self,
        model: LexicalSemanticEmbeddingModel,
        train_config: TrainingConfig,
        data_config: DataConfig,
        device: torch.device,
        output_dir: Path,
        use_wandb: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_config: Training configuration
            data_config: Data configuration
            device: Device to use for training
            output_dir: Output directory for checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.train_config = train_config
        self.data_config = data_config

# fix: move batch to device before forward, fixes CUDA/CPU mismatch
