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
    """
    Download and prepare datasets.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting data preparation...")
    
    # Load data configuration
    try:
        data_config = DataConfig()
    except Exception as e:
        logger.error(f"Error loading data configuration: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data downloader
    downloader = DataDownloader(data_config, output_dir)
    
    # Download datasets
    for dataset in args.datasets:
        logger.info(f"Downloading {dataset}...")
        try:
            downloader.download_dataset(dataset, force=args.force)
        except Exception as e:
            logger.error(f"Error downloading {dataset}: {e}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_config, output_dir)
    
    # Preprocess datasets
    for dataset in args.datasets:
        logger.info(f"Preprocessing {dataset}...")
        try:
            preprocessor.preprocess_dataset(dataset, force=args.force)
        except Exception as e:
            logger.error(f"Error preprocessing {dataset}: {e}")
    
    logger.info("Data preparation completed!")


def run_inference(args: argparse.Namespace) -> None:
    """
    Run inference on text pairs.
    
    Args:
        args: Command line arguments
    """
    logger.info("Running inference...")
    
    # Set device
    device = get_device()
    
    # Load model
    try:
        model = LexicalSemanticEmbeddingModel.load_model(args.model_path, device)
        model.eval()
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load tokenizer (implement based on your tokenizer)
    # For now, using simple whitespace tokenization
    def simple_tokenize(text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Simple tokenization for demonstration."""
        tokens = text.lower().split()[:max_length-2]  # Reserve space for special tokens
        # Add special tokens (assuming 1=CLS, 2=SEP, 0=PAD)
        token_ids = [1] + [hash(token) % 30000 + 3 for token in tokens] + [2]
        
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(0)
        
        attention_mask = [1 if tid != 0 else 0 for tid in token_ids]
        
        return {
            'input_ids': torch.tensor([token_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }
    
    # Tokenize input texts
    tokens1 = simple_tokenize(args.text1)
    tokens2 = simple_tokenize(args.text2)
    
    # Move to device
    for key in tokens1:
        tokens1[key] = tokens1[key].to(device)
        tokens2[key] = tokens2[key].to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(
            input_ids_1=tokens1['input_ids'],
            input_ids_2=tokens2['input_ids'],
            attention_mask_1=tokens1['attention_mask'],
            attention_mask_2=tokens2['attention_mask']
        )
    
    similarity_score = outputs['similarity'].item()
    
    print(f"\nText 1: {args.text1}")
    print(f"Text 2: {args.text2}")
    print(f"Similarity Score: {similarity_score:.4f}")
    
    # Get individual embeddings
    embedding1 = outputs['embedding1'].cpu().numpy()
    embedding2 = outputs['embedding2'].cpu().numpy()
    
    print(f"Embedding 1 shape: {embedding1.shape}")
    print(f"Embedding 2 shape: {embedding2.shape}")


def export_model(args: argparse.Namespace) -> None:
    """
    Export model to different formats.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Exporting model to {args.format} format...")
    
    # Set device
    device = get_device()
    
    # Load model
    try:
        model = LexicalSemanticEmbeddingModel.load_model(args.model_path, device)
        model.eval()
        logger.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create example input
    batch_size = 1
    seq_len = args.seq_length
    
    example_input = {
        'input_ids_1': torch.randint(1, 1000, (batch_size, seq_len), device=device),
        'input_ids_2': torch.randint(1, 1000, (batch_size, seq_len), device=device),
        'attention_mask_1': torch.ones(batch_size, seq_len, device=device),
        'attention_mask_2': torch.ones(batch_size, seq_len, device=device)
    }
    
    # Export based on format
    if args.format == 'onnx':
        model.export_to_onnx(args.output_path, example_input)
    elif args.format == 'torchscript':
        # Export to TorchScript
        traced_model = torch.jit.trace(model, (
            example_input['input_ids_1'],
            example_input['input_ids_2'],
            example_input['attention_mask_1'],
            example_input['attention_mask_2']
        ))
        traced_model.save(args.output_path)
        logger.info(f"Model exported to TorchScript format at {args.output_path}")
    else:
        logger.error(f"Export format {args.format} not yet implemented")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logger.info(f"Starting Lexical Semantic Embedding Model - Command: {args.command}")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        if args.command == 'train':
            train_model(args)
        elif args.command == 'evaluate':
            evaluate_model(args)
        elif args.command == 'prepare-data':
            prepare_data(args)
        elif args.command == 'infer':
            run_inference(args)
        elif args.command == 'export':
            export_model(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Import modules that might not be available during argument parsing
    import importlib.util
    main()
