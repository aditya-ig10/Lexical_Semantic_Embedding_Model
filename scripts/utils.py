"""
Utility functions for the Lexical Semantic Embedding Model project.

This module contains helper functions for logging, device management,
data processing, metrics computation, and other common utilities.

Author: AI Assistant
Date: September 2025
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib

# Import torch with error handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import scipy for statistical functions
try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import sklearn for metrics
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Setup module logger
logger = logging.getLogger(__name__)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (if None, logs to console only)
        format_string: Custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Setup logging configuration
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    logger.info(f"Logging setup complete. Level: {level}, File: {log_file}")


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")


def get_device(gpu_id: Optional[int] = None) -> 'torch.device':
    """
    Get the appropriate device for computation.
    
    Args:
        gpu_id: Specific GPU ID to use (if None, uses first available)
        
    Returns:
        PyTorch device object
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available")
    
    if torch.cuda.is_available():
        if gpu_id is not None:
            if gpu_id >= torch.cuda.device_count():
                logger.warning(f"GPU {gpu_id} not available. Using GPU 0 instead.")
                gpu_id = 0
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda")
        
        # Log GPU information
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def get_model_size(model: 'nn.Module') -> Dict[str, int]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": model_size_mb
    }


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)
    
    with open(path, 'w') as f:
        json.dump(serializable_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Configuration saved to {path}")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Configuration loaded from {path}")
    return config


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        path: Path to save the object
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.info(f"Object saved to {path}")


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        path: Path to the pickle file
        
    Returns:
        Loaded object
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    
    logger.info(f"Object loaded from {path}")
    return obj


def compute_correlation_metrics(
    predictions: List[float],
    targets: List[float]
) -> Dict[str, float]:
    """
    Compute correlation metrics between predictions and targets.
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        Dictionary of correlation metrics
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = {}
    
    # Pearson correlation
    if SCIPY_AVAILABLE:
        pearson_corr, pearson_p = pearsonr(predictions, targets)
        spearman_corr, spearman_p = spearmanr(predictions, targets)
        
        metrics.update({
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p
        })
    else:
        # Simple correlation calculation
        def simple_pearson(x, y):
            x_mean, y_mean = np.mean(x), np.mean(y)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
            return numerator / denominator if denominator != 0 else 0
        
        metrics["pearson_correlation"] = simple_pearson(predictions, targets)
    
    # Mean squared error and mean absolute error
    if SKLEARN_AVAILABLE:
        metrics.update({
            "mse": mean_squared_error(targets, predictions),
            "mae": mean_absolute_error(targets, predictions)
        })
    else:
        metrics.update({
            "mse": np.mean((predictions - targets) ** 2),
            "mae": np.mean(np.abs(predictions - targets))
        })
    
    # Root mean squared error
    metrics["rmse"] = np.sqrt(metrics["mse"])
    
    return metrics


def normalize_scores(
    scores: List[float],
    source_range: Tuple[float, float],
    target_range: Tuple[float, float] = (0.0, 1.0)
) -> List[float]:
    """
    Normalize scores from source range to target range.
    
    Args:
        scores: List of scores to normalize
        source_range: Original range (min, max)
        target_range: Target range (min, max)
        
    Returns:
        Normalized scores
    """
    scores = np.array(scores)
    source_min, source_max = source_range
    target_min, target_max = target_range
    
    # Normalize to [0, 1]
    normalized = (scores - source_min) / (source_max - source_min)
    
    # Scale to target range
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled.tolist()


def create_progress_bar(
    total: int,
    desc: str = "Progress",
    unit: str = "it"
) -> 'tqdm.tqdm':
    """
    Create a progress bar.
    
    Args:
        total: Total number of iterations
        desc: Description
        unit: Unit name
        
    Returns:
        Progress bar object
    """
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit=unit)
    except ImportError:
        logger.warning("tqdm not available. Progress bar disabled.")
        return None


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dir(base_dir: str, experiment_name: str = None) -> Path:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to created experiment directory
    """
    timestamp = get_timestamp()
    
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = f"experiment_{timestamp}"
    
    exp_dir = Path(base_dir) / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def hash_file(file_path: Union[str, Path]) -> str:
    """
    Compute hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA256 hash of the file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def load_text_file(file_path: Union[str, Path], encoding: str = "utf-8") -> List[str]:
    """
    Load text file and return lines.
    
    Args:
        file_path: Path to the text file
        encoding: File encoding
        
    Returns:
        List of lines
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding=encoding) as f:
        lines = [line.strip() for line in f.readlines()]
    
    return lines


def save_text_file(
    lines: List[str],
    file_path: Union[str, Path],
    encoding: str = "utf-8"
) -> None:
    """
    Save lines to text file.
    
    Args:
        lines: List of lines to save
        file_path: Path to save the file
        encoding: File encoding
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding=encoding) as f:
        for line in lines:
            f.write(line + '\n')
    
    logger.info(f"Text file saved to {file_path}")


def download_file(url: str, file_path: Union[str, Path], chunk_size: int = 8192) -> None:
    """
    Download file from URL.
    
    Args:
        url: URL to download from
        file_path: Path to save the file
        chunk_size: Size of chunks to download
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests library is required for downloading files")
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {url} to {file_path}")
    
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}%", end="", flush=True)
    
    print()  # New line after progress
    logger.info(f"Download completed: {file_path}")


def extract_archive(
    archive_path: Union[str, Path],
    extract_dir: Union[str, Path],
    remove_archive: bool = False
) -> None:
    """
    Extract archive file.
    
    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
        remove_archive: Whether to remove archive after extraction
    """
    import tarfile
    import zipfile
    import gzip
    import shutil
    
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {archive_path} to {extract_dir}")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.suffix == '.gz':
        with gzip.open(archive_path, 'rb') as gz_file:
            with open(extract_dir / archive_path.stem, 'wb') as out_file:
                shutil.copyfileobj(gz_file, out_file)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    if remove_archive:
        archive_path.unlink()
        logger.info(f"Removed archive: {archive_path}")
    
    logger.info("Extraction completed")


def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage information
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent()
        }
    except ImportError:
        logger.warning("psutil not available. Memory usage unavailable.")
        return {}


def gpu_memory_usage() -> Dict[str, float]:
    """
    Get GPU memory usage information.
    
    Returns:
        Dictionary with GPU memory usage information
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    return {
        "allocated_mb": allocated,
        "cached_mb": cached,
        "max_allocated_mb": max_allocated
    }


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {format_time(elapsed)}")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


class MetricsTracker:
    """Track and compute metrics during training/evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(
        self,
        predictions: Union[List[float], np.ndarray],
        targets: Union[List[float], np.ndarray],
        loss: Optional[float] = None
    ):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Predicted values
            targets: Target values
            loss: Loss value
        """
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()
        
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions or not self.targets:
            return {}
        
        metrics = compute_correlation_metrics(self.predictions, self.targets)
        
        if self.losses:
            metrics["avg_loss"] = np.mean(self.losses)
        
        return metrics


if __name__ == "__main__":
    # Example usage and testing
    print("Utility Functions Examples:")
    print("=" * 50)
    
    # Setup logging
    setup_logging("INFO")
    
    # Set random seed
    set_seed(42)
    
    # Test device detection
    if TORCH_AVAILABLE:
        device = get_device()
        print(f"Device: {device}")
    
    # Test metrics computation
    predictions = [0.1, 0.5, 0.8, 0.3, 0.9]
    targets = [0.2, 0.4, 0.7, 0.3, 0.8]
    metrics = compute_correlation_metrics(predictions, targets)
    print(f"Metrics: {metrics}")
    
    # Test normalization
    scores = [1, 2, 3, 4, 5]
    normalized = normalize_scores(scores, (1, 5), (0, 1))
    print(f"Normalized scores: {normalized}")
    
    # Test timer
    with Timer("Test operation"):
        time.sleep(0.1)
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update([0.1, 0.2], [0.15, 0.25], 0.05)
    tracker.update([0.3, 0.4], [0.35, 0.45], 0.03)
    final_metrics = tracker.compute()
    print(f"Tracked metrics: {final_metrics}")
    
    # Test timestamp
    timestamp = get_timestamp()
    print(f"Timestamp: {timestamp}")
    
    print("\n✅ All utility function examples completed successfully!")
