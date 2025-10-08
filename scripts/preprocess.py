"""
Data Preprocessing Module for Lexical Semantic Embedding Model.

This module handles text preprocessing, tokenization, and dataset preparation
for training the semantic embedding model.

Author: AI Assistant
Date: September 2025
"""

import os
import logging
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import string
from collections import Counter

import numpy as np
import pandas as pd

# Import project modules
from config.data_config import DataConfig
from scripts.utils import setup_logging, Timer, save_pickle, load_pickle

# Optional imports for advanced preprocessing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing utilities for semantic similarity tasks.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize text preprocessor.
        
        Args:
            config: Data configuration
        """
        self.config = config
        
        # Initialize NLTK components
        if NLTK_AVAILABLE:
            self._init_nltk()
        
        # Initialize spaCy
        if SPACY_AVAILABLE:
            self._init_spacy()
        
        # Preprocessing settings
        self.punctuation = set(string.punctuation)
        
        if NLTK_AVAILABLE and config.remove_stopwords:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                logger.warning("NLTK stopwords not available")
                self.stop_words = set()
        else:
            self.stop_words = set()
    
    def _init_nltk(self):
        """Initialize NLTK components."""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            
            logger.info("NLTK initialized successfully")
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")
    
    def _init_spacy(self):
        """Initialize spaCy model."""
        try:
            # Try to load English model
            self.spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.spacy_nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Remove special characters
        if self.config.remove_special_chars:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.config.tokenizer_type == "word":
            if NLTK_AVAILABLE:
                tokens = word_tokenize(text)
            else:
                tokens = text.split()
        elif self.config.tokenizer_type == "char":
            tokens = list(text)
        else:
            # Simple whitespace tokenization as fallback
            tokens = text.split()
        
        # Remove punctuation
        if self.config.remove_punctuation:
            tokens = [token for token in tokens if token not in self.punctuation]
        
        # Remove stopwords
        if self.config.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Filter by length
        tokens = [token for token in tokens if len(token) > 0]
        
        return tokens
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Apply length constraints
        if len(tokens) < self.config.min_seq_length:
            # Pad with special tokens if too short
            tokens.extend(['<PAD>'] * (self.config.min_seq_length - len(tokens)))
        elif len(tokens) > self.config.max_seq_length:
            # Truncate if too long
            tokens = tokens[:self.config.max_seq_length]
        
        return tokens


class Vocabulary:
    """
    Vocabulary management for text data.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize vocabulary.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.word_to_id = {}
        self.id_to_word = {}
        self.word_counts = Counter()
        
        # Add special tokens
        self.special_tokens = config.special_tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        for token_name, token in self.special_tokens.items():
            if token not in self.word_to_id:
                token_id = len(self.word_to_id)
                self.word_to_id[token] = token_id
                self.id_to_word[token_id] = token
    
    def build_from_texts(self, texts: List[List[str]]) -> None:
        """
        Build vocabulary from list of tokenized texts.
        
        Args:
            texts: List of tokenized texts
        """
        logger.info("Building vocabulary from texts...")
        
        # Count word frequencies
        for tokens in texts:
            self.word_counts.update(tokens)
        
        logger.info(f"Found {len(self.word_counts)} unique tokens")
        
        # Filter by frequency and add to vocabulary
        filtered_words = [
            word for word, count in self.word_counts.most_common()
            if count >= self.config.min_word_frequency
        ]
        
        # Limit vocabulary size
        if len(filtered_words) > self.config.max_vocab_size - len(self.special_tokens):
            filtered_words = filtered_words[:self.config.max_vocab_size - len(self.special_tokens)]
        
        # Add words to vocabulary
        for word in filtered_words:
            if word not in self.word_to_id:
                word_id = len(self.word_to_id)
                self.word_to_id[word] = word_id
                self.id_to_word[word_id] = word
        
        logger.info(f"Final vocabulary size: {len(self.word_to_id)}")
    
    def encode(self, tokens: List[str]) -> List[int]:
        """
        Encode tokens to IDs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        unk_id = self.word_to_id.get(self.special_tokens["unk"], 1)
        return [self.word_to_id.get(token, unk_id) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs to tokens.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of tokens
        """
        unk_token = self.special_tokens["unk"]
        return [self.id_to_word.get(token_id, unk_token) for token_id in token_ids]
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save vocabulary to file.
        
        Args:
            path: Path to save vocabulary
        """
        vocab_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'word_counts': dict(self.word_counts),
            'special_tokens': self.special_tokens,
            'config': self.config.to_dict()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vocabulary saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], config: DataConfig) -> 'Vocabulary':
        """
        Load vocabulary from file.
        
        Args:
            path: Path to vocabulary file
            config: Data configuration
            
        Returns:
            Loaded vocabulary
        """
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab = cls(config)
        vocab.word_to_id = vocab_data['word_to_id']
        vocab.id_to_word = {int(k): v for k, v in vocab_data['id_to_word'].items()}
        vocab.word_counts = Counter(vocab_data['word_counts'])
        
        logger.info(f"Vocabulary loaded from {path}")
        return vocab
    
    def __len__(self) -> int:
        return len(self.word_to_id)


class DataPreprocessor:
    """
    Main data preprocessing class for semantic similarity datasets.
    """
    
    def __init__(self, config: DataConfig, output_dir: Path):
        """
        Initialize data preprocessor.
        
        Args:
            config: Data configuration
            output_dir: Output directory for processed data
        """
        self.config = config
