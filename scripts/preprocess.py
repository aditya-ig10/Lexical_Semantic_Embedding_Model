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
            

# fix: strip unicode punctuation, handle empty strings -> [UNK]
