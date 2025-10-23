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
    
    # Logging and monitoring
    logging_steps: int = 100
    log_level: str = "INFO"
    log_on_each_node: bool = False
    
    # Weights & Biases (wandb) settings
    use_wandb: bool = False
    wandb_project: str = "lexical-embedding-model"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    
    # TensorBoard settings
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "./logs"
    
    # Mixed precision training
    use_amp: bool = False  # Automatic Mixed Precision
    amp_opt_level: str = "O1"  # NVIDIA Apex optimization level
    
    # Multi-GPU settings
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Advanced training strategies
    gradient_checkpointing: bool = False
    find_unused_parameters: bool = False
    
    # Curriculum learning
    use_curriculum_learning: bool = False
    curriculum_strategy: str = "length"  # Options: length, difficulty, random
    curriculum_epochs: int = 20
    
    # Data-related training settings
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    remove_unused_columns: bool = True
    
    # Hyperparameter search settings
    use_hyperparameter_search: bool = False
    search_space: Dict[str, Any] = field(default_factory=dict)
    num_trials: int = 20
    search_metric: str = "val_pearson"
    search_direction: str = "maximize"
    
    # Model averaging
    use_model_averaging: bool = False
    averaging_start_epoch: int = 50
    averaging_decay: float = 0.999
    
    # Custom training settings
    custom_metrics: List[str] = field(default_factory=lambda: ["pearson", "spearman"])
    compute_metrics_during_training: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_dependent_values()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate basic training settings
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        
        # Validate optimization settings
        valid_optimizers = ["adam", "adamw", "sgd", "rmsprop", "adagrad"]
        assert self.optimizer in valid_optimizers, f"optimizer must be one of {valid_optimizers}"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.weight_decay <= 1, "weight_decay must be between 0 and 1"
        assert 0 <= self.beta1 < 1, "beta1 must be between 0 and 1"
        assert 0 <= self.beta2 < 1, "beta2 must be between 0 and 1"
        assert self.epsilon > 0, "epsilon must be positive"
        
        # Validate learning rate scheduling
        valid_schedulers = ["cosine", "linear", "exponential", "step", "plateau", "none"]
        assert self.lr_scheduler in valid_schedulers, f"lr_scheduler must be one of {valid_schedulers}"
        assert 0 <= self.warmup_ratio <= 1, "warmup_ratio must be between 0 and 1"
        assert self.min_lr >= 0, "min_lr must be non-negative"
        assert self.min_lr < self.learning_rate, "min_lr must be less than learning_rate"
        
        # Validate loss function
        valid_loss_functions = ["mse", "mae", "huber", "cosine_embedding", "ranking"]
        assert self.loss_function in valid_loss_functions, f"loss_function must be one of {valid_loss_functions}"
        
        # Validate dropout rates
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert 0 <= self.attention_dropout <= 1, "attention_dropout must be between 0 and 1"
        assert 0 <= self.hidden_dropout <= 1, "hidden_dropout must be between 0 and 1"
        assert 0 <= self.label_smoothing <= 1, "label_smoothing must be between 0 and 1"
        
        # Validate early stopping
        if self.use_early_stopping:
            assert self.early_stopping_patience > 0, "early_stopping_patience must be positive"
            valid_modes = ["min", "max"]
            assert self.early_stopping_mode in valid_modes, f"early_stopping_mode must be one of {valid_modes}"
        
        # Validate strategies
        valid_save_strategies = ["epoch", "steps", "best"]
        assert self.save_strategy in valid_save_strategies, f"save_strategy must be one of {valid_save_strategies}"
        
        valid_eval_strategies = ["epoch", "steps", "no"]
        assert self.eval_strategy in valid_eval_strategies, f"eval_strategy must be one of {valid_eval_strategies}"
        
        # Validate steps
        assert self.logging_steps > 0, "logging_steps must be positive"
        if self.save_strategy == "steps":
            assert self.save_steps > 0, "save_steps must be positive when save_strategy is 'steps'"
        if self.eval_strategy == "steps":
            assert self.eval_steps > 0, "eval_steps must be positive when eval_strategy is 'steps'"
    
    def _setup_dependent_values(self):
        """Setup values that depend on other configuration parameters."""
        # Setup warmup steps based on ratio if not explicitly set
        if hasattr(self, '_total_steps') and self._total_steps:
            if self.warmup_steps == 0 and self.warmup_ratio > 0:
                self.warmup_steps = int(self._total_steps * self.warmup_ratio)
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """
        Get optimizer configuration.
        
        Returns:
            Optimizer configuration dictionary
        """
        base_config = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay
        }
        
        if self.optimizer in ["adam", "adamw"]:
            base_config.update({
                "betas": (self.beta1, self.beta2),
                "eps": self.epsilon,
                "amsgrad": self.amsgrad
            })
        
        return base_config
    
    def get_scheduler_config(self, total_steps: int) -> Dict[str, Any]:
        """
        Get learning rate scheduler configuration.
        
        Args:
            total_steps: Total number of training steps
            
        Returns:
            Scheduler configuration dictionary
        """
        self._total_steps = total_steps
        
        if self.warmup_steps == 0 and self.warmup_ratio > 0:
            self.warmup_steps = int(total_steps * self.warmup_ratio)
        
        config = {
            "scheduler_type": self.lr_scheduler,
            "warmup_steps": self.warmup_steps,
            "total_steps": total_steps,
            "min_lr": self.min_lr
        }
        
        if self.lr_scheduler == "cosine":
            config.update({
                "eta_min": self.min_lr,
                "T_max": total_steps - self.warmup_steps
            })
        elif self.lr_scheduler == "exponential":
            config["gamma"] = self.lr_decay_rate
        elif self.lr_scheduler == "step":
            config.update({
                "step_size": self.lr_decay_steps,
                "gamma": self.lr_decay_rate
            })
        elif self.lr_scheduler == "plateau":
            config.update({
                "patience": self.patience,
                "factor": self.lr_decay_rate,
                "min_lr": self.min_lr
            })
        
        return config
    
    def get_loss_config(self) -> Dict[str, Any]:
        """
        Get loss function configuration.
        
        Returns:
            Loss function configuration dictionary
        """
        config = {"loss_type": self.loss_function}
        
        if self.loss_function == "huber":
            config["delta"] = self.huber_delta
        elif self.loss_function == "ranking":
            config["margin"] = self.margin
        elif self.loss_function == "cosine_embedding":
            config["margin"] = self.margin
        
        config["weights"] = self.loss_weights
        config["label_smoothing"] = self.label_smoothing
        
        return config
    
    def get_early_stopping_config(self) -> Dict[str, Any]:
        """
        Get early stopping configuration.
        
        Returns:
            Early stopping configuration dictionary
        """
        return {
            "patience": self.early_stopping_patience,
