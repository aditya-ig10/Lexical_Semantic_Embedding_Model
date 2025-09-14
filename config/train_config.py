"""
Training Configuration for Lexical Semantic Embedding Model.

This module contains all training-related configurations including
optimization, scheduling, checkpointing, and monitoring settings.

Author: AI Assistant
Date: September 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import math


@dataclass
class TrainingConfig:
    """
    Configuration class for training settings and hyperparameters.
    
    This class contains all the settings needed for training the
    Lexical Semantic Embedding Model.
    """
    
    # Basic training settings
    num_epochs: int = 100
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimization settings
    optimizer: str = "adamw"  # Options: adam, adamw, sgd, rmsprop, adagrad
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    amsgrad: bool = False
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # Options: cosine, linear, exponential, step, plateau, none
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    min_lr: float = 1e-6
    lr_decay_rate: float = 0.95
    lr_decay_steps: int = 1000
    patience: int = 5  # For plateau scheduler
    
    # Loss function settings
    loss_function: str = "mse"  # Options: mse, mae, huber, cosine_embedding, ranking
    margin: float = 0.5  # For ranking loss
    huber_delta: float = 1.0  # For Huber loss
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"similarity": 1.0})
    
    # Regularization
    dropout: float = 0.3
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"  # Options: val_loss, val_accuracy, val_pearson
    early_stopping_mode: str = "min"  # Options: min, max
    early_stopping_delta: float = 1e-4
    
    # Checkpointing
    save_strategy: str = "epoch"  # Options: epoch, steps, best
    save_steps: int = 500
    save_total_limit: int = 3
    save_best_only: bool = False
    best_metric: str = "val_pearson"
    best_metric_mode: str = "max"
    
    # Evaluation settings
    eval_strategy: str = "epoch"  # Options: epoch, steps, no
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None
    eval_delay: int = 0
