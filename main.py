"""
Main entry point for the Lexical Semantic Embedding Model project.

This script provides a command-line interface for training, evaluating,
and using the semantic embedding model.

Author: AI Assistant
Date: September 2025
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
import json
import warnings
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from lexical_embedding_model import LexicalSemanticEmbeddingModel
from config.model_config import ModelConfig
from config.train_config import TrainingConfig
from config.data_config import DataConfig
from scripts.train import ModelTrainer
from scripts.evaluate import ModelEvaluator
from scripts.data_download import DataDownloader
from scripts.preprocess import DataPreprocessor
from scripts.utils import setup_logging, set_seed, get_device, save_config

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Lexical Semantic Embedding Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--config', 
        type=str, 
        default='config/train_config.py',
        help='Path to training configuration file'
    )
    train_parser.add_argument(
        '--model-config',
        type=str,
        default='config/model_config.py',
        help='Path to model configuration file'
    )
    train_parser.add_argument(
        '--data-config',
        type=str,
        default='config/data_config.py',
        help='Path to data configuration file'
    )
    train_parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Directory to save trained models'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    train_parser.add_argument(
        '--wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    train_parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use'
    )
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    eval_parser.add_argument(
        '--data-config',
        type=str,
        default='config/data_config.py',
        help='Path to data configuration file'
    )
    eval_parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation results'
    )
    eval_parser.add_argument(
        '--datasets',
        nargs='+',
        default=['sts-benchmark', 'sick', 'quora'],
        help='Datasets to evaluate on'
    )
    eval_parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Evaluation batch size'
    )
    
    # Data preparation command
    data_parser = subparsers.add_parser('prepare-data', help='Download and prepare datasets')
    data_parser.add_argument(
        '--config',
        type=str,
        default='config/data_config.py',
        help='Path to data configuration file'
    )
    data_parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Directory to save prepared data'
    )
    data_parser.add_argument(
        '--datasets',
        nargs='+',
        default=['sts-benchmark', 'sick', 'quora', 'mrpc'],
        help='Datasets to download and prepare'
    )
    data_parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download and re-process data'
    )
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on text pairs')
    infer_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    infer_parser.add_argument(
        '--text1',
        type=str,
        required=True,
        help='First text for similarity comparison'
    )
    infer_parser.add_argument(
        '--text2',
        type=str,
        required=True,
        help='Second text for similarity comparison'
    )
    infer_parser.add_argument(
        '--tokenizer-path',
        type=str,
        default='./data/tokenizer.json',
        help='Path to tokenizer'
    )
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model to different formats')
    export_parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    export_parser.add_argument(
        '--format',
        choices=['onnx', 'torchscript', 'tflite'],
        default='onnx',
        help='Export format'
    )
    export_parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Output path for exported model'
    )
    export_parser.add_argument(
        '--seq-length',
        type=int,
        default=128,
        help='Maximum sequence length for export'
    )
    
    # General arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file'
    )
    
    return parser.parse_args()


def load_configuration(config_path: str) -> Dict:
    """
    Load configuration from Python file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Import configuration module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract configuration
    if hasattr(config_module, 'get_config'):
        return config_module.get_config()
    else:
        # Look for config classes
        for attr_name in dir(config_module):
            attr = getattr(config_module, attr_name)
            if isinstance(attr, type) and attr_name.endswith('Config'):
                return attr().__dict__
    
    raise ValueError(f"No configuration found in {config_path}")


def train_model(args: argparse.Namespace) -> None:
    """
    Train the semantic embedding model.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting model training...")
    
    # Load configurations
    try:
        model_config = ModelConfig()
        train_config = TrainingConfig()
        data_config = DataConfig()
    except Exception as e:
        logger.error(f"Error loading configurations: {e}")
        sys.exit(1)
    
    # Set device
    device = get_device(args.gpu)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations
    save_config(model_config.__dict__, output_dir / "model_config.json")
    save_config(train_config.__dict__, output_dir / "train_config.json")
    save_config(data_config.__dict__, output_dir / "data_config.json")
    
    # Initialize model
    model = LexicalSemanticEmbeddingModel(
        vocab_size=model_config.vocab_size,
        embed_dim=model_config.embed_dim,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        num_attention_heads=model_config.num_attention_heads,
        similarity_function=model_config.similarity_function
    )
    
    # Move model to device
    model = model.to(device)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_config=train_config,
        data_config=data_config,
        device=device,
        output_dir=output_dir,
        use_wandb=args.wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from {args.resume}")
    
    # Train model
    trainer.train()
    
    logger.info("Training completed successfully!")


def evaluate_model(args: argparse.Namespace) -> None:
    """
    Evaluate the trained model.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting model evaluation...")
    
    # Load data configuration
    try:
        data_config = DataConfig()
    except Exception as e:
        logger.error(f"Error loading data configuration: {e}")
        sys.exit(1)
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        model = LexicalSemanticEmbeddingModel.load_model(args.model_path, device)
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=model,
        data_config=data_config,
        device=device,
        batch_size=args.batch_size
    )
    
    # Evaluate on specified datasets
    results = {}
    for dataset in args.datasets:
        logger.info(f"Evaluating on {dataset}...")
        try:
            result = evaluator.evaluate_dataset(dataset)
            results[dataset] = result
            logger.info(f"{dataset} evaluation completed: {result}")
        except Exception as e:
            logger.error(f"Error evaluating {dataset}: {e}")
    
    # Save results
    results_file = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_file}")


def prepare_data(args: argparse.Namespace) -> None:
