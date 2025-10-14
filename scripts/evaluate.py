"""
Evaluation Module for Lexical Semantic Embedding Model.

This module provides comprehensive evaluation tools for semantic similarity models
including metrics computation, visualization, and benchmarking.

Author: AI Assistant
Date: September 2025
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import math

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Import project modules
from lexical_embedding_model import LexicalSemanticEmbeddingModel
from config.data_config import DataConfig
from scripts.utils import (
    setup_logging, get_device, compute_correlation_metrics,
    MetricsTracker, Timer, create_experiment_dir
)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class for semantic similarity tasks.
    """
    
    def __init__(
        self,
        model: LexicalSemanticEmbeddingModel,
        data_config: DataConfig,
        device: torch.device,
        batch_size: int = 32
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained model to evaluate
            data_config: Data configuration
            device: Device for computation
            batch_size: Batch size for evaluation
        """
        self.model = model.to(device)
        self.data_config = data_config
        self.device = device
        self.batch_size = batch_size
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluation metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Results storage
        self.evaluation_results = {}
    
    def evaluate_pairs(
        self,
        sentences1: List[str],
        sentences2: List[str],
        true_scores: List[float],
        dataset_name: str = "unknown"
    ) -> Dict[str, float]:
        """
        Evaluate model on sentence pairs.
        
        Args:
            sentences1: First sentences
            sentences2: Second sentences
            true_scores: True similarity scores
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {len(sentences1)} pairs from {dataset_name}")
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(sentences1), self.batch_size):
                batch_end = min(i + self.batch_size, len(sentences1))
                
                batch_sentences1 = sentences1[i:batch_end]
                batch_sentences2 = sentences2[i:batch_end]
                
                # Tokenize sentences (simplified - should use proper tokenizer)
                batch_inputs1 = self._tokenize_batch(batch_sentences1)
                batch_inputs2 = self._tokenize_batch(batch_sentences2)
                
                # Forward pass
                outputs = self.model(
                    input_ids_1=batch_inputs1['input_ids'],
                    input_ids_2=batch_inputs2['input_ids'],
                    attention_mask_1=batch_inputs1['attention_mask'],
                    attention_mask_2=batch_inputs2['attention_mask']
                )
                
                batch_predictions = outputs['similarity'].cpu().numpy()
                predictions.extend(batch_predictions)
        
        # Compute metrics
        metrics = compute_correlation_metrics(predictions, true_scores)
        
        # Add dataset-specific metrics
        metrics['dataset'] = dataset_name
        metrics['num_samples'] = len(predictions)
        
        logger.info(f"Evaluation completed for {dataset_name}")
        logger.info(f"Pearson correlation: {metrics.get('pearson_correlation', 0):.4f}")
        logger.info(f"Spearman correlation: {metrics.get('spearman_correlation', 0):.4f}")
        logger.info(f"MSE: {metrics.get('mse', 0):.4f}")
        
        return metrics
    
    def _tokenize_batch(self, sentences: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Simple tokenization (should be replaced with proper tokenizer)
        max_length = self.data_config.max_seq_length
        
        input_ids = []
        attention_masks = []
        
        for sentence in sentences:
            # Simple word tokenization and ID assignment
            tokens = sentence.lower().split()[:max_length-2]  # Reserve space for special tokens
            
            # Add special tokens
            token_ids = [2] + [hash(token) % 30000 + 3 for token in tokens] + [3]  # CLS + tokens + SEP
            
            # Pad to max_length
            while len(token_ids) < max_length:
                token_ids.append(0)  # PAD token
            
            attention_mask = [1 if tid != 0 else 0 for tid in token_ids]
            
            input_ids.append(token_ids)
            attention_masks.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(input_ids, device=self.device),
            'attention_mask': torch.tensor(attention_masks, device=self.device)
        }
    
    def evaluate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate model on a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Evaluation results
        """
        # Load dataset (simplified - should load from preprocessed data)
        try:
            dataset_path = self.data_config.get_processed_data_dir() / f"{dataset_name}_processed.pkl"
            
            if not dataset_path.exists():
                logger.error(f"Processed dataset not found: {dataset_path}")
                return {}
            
            # Load processed data (simplified loading)
            import pickle
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            
            results = {}
            
            # Evaluate each split
            for split_name, split_data in data['datasets'].items():
                sentences1 = [item['sentence1'] for item in split_data]
                sentences2 = [item['sentence2'] for item in split_data]
                scores = [item['score'] for item in split_data]
                
                split_metrics = self.evaluate_pairs(
                    sentences1, sentences2, scores, f"{dataset_name}_{split_name}"
                )
                
                results[split_name] = split_metrics
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating dataset {dataset_name}: {e}")
            return {}
    
    def compute_embedding_similarities(
        self,
        sentences: List[str],
        query_sentence: str
    ) -> List[Tuple[str, float]]:
        """
        Compute similarities between a query sentence and a list of sentences.
        
        Args:
            sentences: List of candidate sentences
            query_sentence: Query sentence
            
        Returns:
            List of (sentence, similarity) tuples sorted by similarity
        """
        # Get query embedding
        query_embedding = self._get_sentence_embedding(query_sentence)
        
        # Get embeddings for all sentences
        sentence_embeddings = []
        for sentence in sentences:
            embedding = self._get_sentence_embedding(sentence)
            sentence_embeddings.append(embedding)
        
        # Compute similarities
        similarities = []
        for i, sentence in enumerate(sentences):
            similarity = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                sentence_embeddings[i].unsqueeze(0)
            ).item()
            similarities.append((sentence, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def _get_sentence_embedding(self, sentence: str) -> torch.Tensor:
        """
        Get embedding for a single sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Sentence embedding tensor
        """
        with torch.no_grad():
            tokens = self._tokenize_batch([sentence])
            embedding = self.model.get_sentence_embedding(
                tokens['input_ids'],
                tokens['attention_mask']
            )
            return embedding.squeeze(0)
    
    def analyze_embeddings(
        self,
        sentences: List[str],
        labels: Optional[List[str]] = None,
        method: str = "pca"
    ) -> Dict[str, Any]:
        """
        Analyze sentence embeddings using dimensionality reduction.
        
        Args:
            sentences: List of sentences
            labels: Optional labels for sentences
            method: Dimensionality reduction method ('pca' or 'tsne')
            
        Returns:
            Analysis results including coordinates and plot data
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available for embedding analysis")
            return {}
        
        logger.info(f"Analyzing embeddings for {len(sentences)} sentences using {method}")
        
        # Get embeddings
        embeddings = []
        for sentence in sentences:
            embedding = self._get_sentence_embedding(sentence)
            embeddings.append(embedding.cpu().numpy())
        
        embeddings = np.array(embeddings)
        
        # Apply dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sentences)-1))
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        coordinates = reducer.fit_transform(embeddings)
        
        # Prepare results
        results = {
            'coordinates': coordinates,
            'sentences': sentences,
            'labels': labels or [f"sent_{i}" for i in range(len(sentences))],
            'method': method,
            'variance_explained': getattr(reducer, 'explained_variance_ratio_', None)
        }
        
        return results
    
    def create_evaluation_plots(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """
        Create evaluation plots and visualizations.
        
        Args:
            results: Evaluation results

# fix: pad-batch collate for variable lengths
