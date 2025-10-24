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
        self.output_dir = Path(output_dir)
        self.processed_dir = self.output_dir / "processed"
        self.raw_dir = self.output_dir / "raw"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.text_preprocessor = TextPreprocessor(config)
        self.vocabulary = Vocabulary(config)
    
    def load_sts_benchmark(self) -> Dict[str, pd.DataFrame]:
        """
        Load STS Benchmark dataset.
        
        Returns:
            Dictionary with train/dev/test DataFrames
        """
        dataset_dir = self.raw_dir / "sts-benchmark"
        
        datasets = {}
        file_mapping = {
            "train": "sts-train.csv",
            "dev": "sts-dev.csv", 
            "test": "sts-test.csv"
        }
        
        for split, filename in file_mapping.items():
            file_path = None
            # Search for the file in subdirectories
            for path in dataset_dir.rglob(filename):
                file_path = path
                break
            
            if file_path and file_path.exists():
                # STS benchmark format: genre, filename, year, score, sentence1, sentence2
                df = pd.read_csv(
                    file_path, 
                    sep='\t', 
                    header=None,
                    names=['genre', 'filename', 'year', 'score', 'sentence1', 'sentence2'],
                    on_bad_lines='skip'
                )
                
                # Filter out rows with missing scores
                df = df.dropna(subset=['score', 'sentence1', 'sentence2'])
                
                # Normalize scores to [0, 1]
                if self.config.sts_benchmark_config.get('normalize_scores', True):
                    score_range = self.config.sts_benchmark_config.get('score_range', (0, 5))
                    df['score'] = (df['score'] - score_range[0]) / (score_range[1] - score_range[0])
                
                datasets[split] = df
                logger.info(f"Loaded STS-{split}: {len(df)} samples")
            else:
                logger.warning(f"STS file not found: {filename}")
        
        return datasets
    
    def load_sick_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Load SICK dataset.
        
        Returns:
            Dictionary with train/trial/test DataFrames
        """
        dataset_dir = self.raw_dir / "sick"
        
        datasets = {}
        file_mapping = {
            "train": "SICK_train.txt",
            "trial": "SICK_trial.txt",
            "test": "SICK_test_annotated.txt"
        }
        
        for split, filename in file_mapping.items():
            file_path = None
            # Search for the file in subdirectories
            for path in dataset_dir.rglob(filename):
                file_path = path
                break
            
            if file_path and file_path.exists():
                df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
                
                # SICK dataset columns: pair_ID, sentence_A, sentence_B, relatedness_score, entailment_judgment
                if 'sentence_A' in df.columns and 'sentence_B' in df.columns:
                    df = df.rename(columns={
                        'sentence_A': 'sentence1',
                        'sentence_B': 'sentence2',
                        'relatedness_score': 'score'
                    })
                    
                    # Filter out rows with missing data
                    df = df.dropna(subset=['score', 'sentence1', 'sentence2'])
                    
                    # Normalize scores to [0, 1]
                    if self.config.sick_config.get('normalize_scores', True):
                        score_range = self.config.sick_config.get('score_range', (1, 5))
                        df['score'] = (df['score'] - score_range[0]) / (score_range[1] - score_range[0])
                    
                    datasets[split] = df
                    logger.info(f"Loaded SICK-{split}: {len(df)} samples")
                else:
                    logger.warning(f"Unexpected format in SICK file: {filename}")
            else:
                logger.warning(f"SICK file not found: {filename}")
        
        return datasets
    
    def load_quora_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Load Quora Question Pairs dataset.
        
        Returns:
            Dictionary with train/dev/test DataFrames
        """
        dataset_dir = self.raw_dir / "quora"
        file_path = dataset_dir / "quora_duplicate_questions.tsv"
        
        if not file_path.exists():
            # Search in subdirectories
            for path in dataset_dir.rglob("*.tsv"):
                if "quora" in path.name.lower():
                    file_path = path
                    break
        
        if not file_path.exists():
            logger.warning("Quora dataset file not found")
            return {}
        
        # Load dataset
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
        
        # Expected columns: id, qid1, qid2, question1, question2, is_duplicate
        required_cols = ['question1', 'question2', 'is_duplicate']
        
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Quora dataset missing required columns: {required_cols}")
            return {}
        
        # Clean data
        df = df.dropna(subset=required_cols)
        df = df.rename(columns={
            'question1': 'sentence1',
            'question2': 'sentence2',
            'is_duplicate': 'score'
        })
        
        # Convert binary labels to float
        df['score'] = df['score'].astype(float)
        
        # Split dataset
        train_size = int(len(df) * self.config.train_split)
        val_size = int(len(df) * self.config.val_split)
        
        datasets = {
            'train': df[:train_size],
            'dev': df[train_size:train_size + val_size],
            'test': df[train_size + val_size:]
        }
        
        for split, data in datasets.items():
            logger.info(f"Loaded Quora-{split}: {len(data)} samples")
        
        return datasets
    
    def preprocess_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """
        Preprocess a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            force: Whether to force reprocessing
            
        Returns:
            True if successful, False otherwise
        """
        output_path = self.processed_dir / f"{dataset_name}_processed.pkl"
        
        if output_path.exists() and not force:
            logger.info(f"Processed dataset {dataset_name} already exists")
            return True
        
        logger.info(f"Preprocessing dataset: {dataset_name}")
        
        try:
            with Timer(f"Preprocess {dataset_name}"):
                # Load raw dataset
                if dataset_name == "sts-benchmark":
                    datasets = self.load_sts_benchmark()
                elif dataset_name == "sick":
                    datasets = self.load_sick_dataset()
                elif dataset_name == "quora":
                    datasets = self.load_quora_dataset()
                else:
                    logger.error(f"Unknown dataset: {dataset_name}")
                    return False
                
                if not datasets:
                    logger.error(f"Failed to load dataset: {dataset_name}")
                    return False
                
                # Preprocess each split
                processed_datasets = {}
                all_texts = []
                
                for split, df in datasets.items():
                    # Preprocess texts
                    processed_data = []
                    
                    for _, row in df.iterrows():
                        sentence1 = str(row['sentence1'])
                        sentence2 = str(row['sentence2'])
                        score = float(row['score'])
                        
                        # Preprocess sentences
                        tokens1 = self.text_preprocessor.preprocess_text(sentence1)
                        tokens2 = self.text_preprocessor.preprocess_text(sentence2)
                        
                        if len(tokens1) >= self.config.min_seq_length and len(tokens2) >= self.config.min_seq_length:
                            processed_data.append({
                                'sentence1': sentence1,
                                'sentence2': sentence2,
                                'tokens1': tokens1,
                                'tokens2': tokens2,
                                'score': score
                            })
                            
                            all_texts.extend([tokens1, tokens2])
                    
                    processed_datasets[split] = processed_data
                    logger.info(f"Processed {split}: {len(processed_data)} samples")
                
                # Build vocabulary from all texts
                self.vocabulary.build_from_texts(all_texts)
                
                # Encode tokens to IDs
                for split, data in processed_datasets.items():
                    for item in data:
                        item['token_ids1'] = self.vocabulary.encode(item['tokens1'])
                        item['token_ids2'] = self.vocabulary.encode(item['tokens2'])
                
                # Save processed data
                processed_output = {
                    'datasets': processed_datasets,
                    'vocabulary': {
                        'word_to_id': self.vocabulary.word_to_id,
                        'id_to_word': self.vocabulary.id_to_word,
                        'special_tokens': self.vocabulary.special_tokens
                    },
                    'config': self.config.to_dict()
                }
                
                save_pickle(processed_output, output_path)
                
                # Save vocabulary separately
                vocab_path = self.processed_dir / f"{dataset_name}_vocab.json"
                self.vocabulary.save(vocab_path)
                
                logger.info(f"Dataset {dataset_name} preprocessed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error preprocessing {dataset_name}: {e}")
            return False
    
    def preprocess_all_datasets(self, force: bool = False) -> Dict[str, bool]:
        """
        Preprocess all configured datasets.
        
        Args:
            force: Whether to force reprocessing
            
        Returns:
            Dictionary with preprocessing results
        """
        results = {}
        
        for dataset_name in self.config.datasets:
            results[dataset_name] = self.preprocess_dataset(dataset_name, force)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"Preprocessing summary: {successful}/{total} datasets successful")
        
        return results


def main():
    """Main function for data preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess semantic similarity datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sts-benchmark", "sick", "quora"],
        help="Datasets to preprocess"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO")
    
    # Create configuration
    config = DataConfig()
    config.datasets = args.datasets
    
    # Create preprocessor
    preprocessor = DataPreprocessor(config, Path(args.data_dir))
    
    # Preprocess datasets
    results = preprocessor.preprocess_all_datasets(args.force)
    
    # Print results
    print("\nPreprocessing Results:")
    print("=" * 30)
    
    for dataset_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{dataset_name}: {status}")


if __name__ == "__main__":
    main()
