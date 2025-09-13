"""
Data Configuration for Lexical Semantic Embedding Model.

This module contains all data processing and dataset configurations.

Author: AI Assistant
Date: September 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import os


@dataclass
class DataConfig:
    """
    Configuration class for data processing and dataset settings.
    
    This class contains all the settings needed for data downloading,
    preprocessing, and loading for the Lexical Semantic Embedding Model.
    """
    
    # Data paths
    data_root: str = "./data"
    raw_data_dir: str = "raw"
    processed_data_dir: str = "processed"
    embeddings_dir: str = "embeddings"
    cache_dir: str = "cache"
    
    # Dataset settings
    datasets: List[str] = field(default_factory=lambda: [
        "sts-benchmark", "sick", "quora", "mrpc"
    ])
    
    # Text preprocessing
    max_seq_length: int = 128
    min_seq_length: int = 3
    max_vocab_size: int = 30000
    min_word_frequency: int = 2
    lowercase: bool = True
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    remove_special_chars: bool = True
    
    # Tokenization settings
    tokenizer_type: str = "word"  # Options: word, subword, char, bert
    tokenizer_vocab_file: Optional[str] = None
    special_tokens: Dict[str, str] = field(default_factory=lambda: {
        "pad": "<PAD>",
        "unk": "<UNK>",
        "cls": "<CLS>",
        "sep": "<SEP>",
        "mask": "<MASK>"
    })
    
    # Subword tokenization settings (for BPE/SentencePiece)
    subword_vocab_size: int = 30000
    subword_model_type: str = "bpe"  # Options: bpe, unigram, char, word
    subword_coverage: float = 0.9995
    
    # Data splitting
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    stratify: bool = True
    
    # Data loading
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    drop_last: bool = True
    
    # Data augmentation
    use_data_augmentation: bool = False
    augmentation_prob: float = 0.1
