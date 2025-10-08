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
        self.device = device
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        if self.use_wandb:
            self._init_wandb()
        
        if train_config.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tensorboard_writer = SummaryWriter(
                log_dir=self.output_dir / "tensorboard"
            )
        else:
            self.tensorboard_writer = None
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None  # For mixed precision training
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf') if train_config.best_metric_mode == 'max' else float('inf')
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        
        if train_config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb_config = self.train_config.get_wandb_config()
        
        wandb.init(
            project=wandb_config["project"],
            entity=wandb_config.get("entity"),
            name=wandb_config.get("name"),
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes"),
            config={
                **self.train_config.to_dict(),
                **self.data_config.to_dict()
            }
        )
        
        # Watch model
        wandb.watch(self.model, log="all", log_freq=self.train_config.logging_steps)
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        optimizer_config = self.train_config.get_optimizer_config()
        
        if self.train_config.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_config)
        elif self.train_config.optimizer == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_config)
        elif self.train_config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                momentum=0.9,
                weight_decay=optimizer_config["weight_decay"]
            )
        elif self.train_config.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), **optimizer_config)
        else:
            raise ValueError(f"Unknown optimizer: {self.train_config.optimizer}")
        
        logger.info(f"Optimizer: {self.train_config.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.train_config.lr_scheduler == "none":
            self.scheduler = None
            return
        
        # Estimate total steps (will be updated when data loaders are available)
        total_steps = 1000  # Placeholder
        scheduler_config = self.train_config.get_scheduler_config(total_steps)
        
        if self.train_config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config["T_max"],
                eta_min=scheduler_config["eta_min"]
            )
        elif self.train_config.lr_scheduler == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=scheduler_config["min_lr"] / self.train_config.learning_rate,
                total_iters=total_steps
            )
        elif self.train_config.lr_scheduler == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config["gamma"]
            )
        elif self.train_config.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"]
            )
        elif self.train_config.lr_scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config["patience"],
                factor=scheduler_config["factor"],
                min_lr=scheduler_config["min_lr"]
            )
        
        logger.info(f"Learning rate scheduler: {self.train_config.lr_scheduler}")
    
    def _setup_loss_function(self):
        """Setup loss function."""
        loss_config = self.train_config.get_loss_config()
        
        if loss_config["loss_type"] == "mse":
            self.criterion = nn.MSELoss()
        elif loss_config["loss_type"] == "mae":
            self.criterion = nn.L1Loss()
        elif loss_config["loss_type"] == "huber":
            self.criterion = nn.HuberLoss(delta=loss_config["delta"])
        elif loss_config["loss_type"] == "cosine_embedding":
            self.criterion = nn.CosineEmbeddingLoss(margin=loss_config["margin"])
        else:
            raise ValueError(f"Unknown loss function: {loss_config['loss_type']}")
        
        logger.info(f"Loss function: {loss_config['loss_type']}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.train_config.use_amp):
                outputs = self.model(
                    input_ids_1=batch['input_ids_1'],
                    input_ids_2=batch['input_ids_2'],
                    attention_mask_1=batch['attention_mask_1'],
                    attention_mask_2=batch['attention_mask_2']
                )
                
                loss = self.criterion(outputs['similarity'], batch['score'])
            
            # Backward pass
            if self.train_config.use_amp:
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.train_config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.train_config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            self.train_metrics.update(
                outputs['similarity'].detach().cpu().numpy(),
                batch['score'].detach().cpu().numpy(),
                loss.item()
            )
            
            # Logging
            if self.global_step % self.train_config.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                
                log_metrics = {
                    "train/loss": loss.item(),
                    "train/learning_rate": current_lr,
                    "train/epoch": self.current_epoch,
                    "train/step": self.global_step
                }
                
                if self.use_wandb:
                    wandb.log(log_metrics, step=self.global_step)
                
                if self.tensorboard_writer:
                    for key, value in log_metrics.items():
                        self.tensorboard_writer.add_scalar(key, value, self.global_step)
                
                logger.info(
                    f"Epoch {self.current_epoch}, Step {self.global_step}: "
                    f"Loss = {loss.item():.4f}, LR = {current_lr:.6f}"
                )
            
            self.global_step += 1

# fix: checkpoint save uses state_dict, load with map_location
