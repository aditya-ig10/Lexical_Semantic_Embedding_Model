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
    augmentation_methods: List[str] = field(default_factory=lambda: [
        "synonym_replacement", "random_insertion", "random_swap", "random_deletion"
    ])
    
    # Pretrained embeddings
    use_pretrained_embeddings: bool = True
    embedding_name: str = "glove.6B.300d"  # Options: glove, word2vec, fasttext
    embedding_dim: int = 300
    embedding_cache_dir: str = "embeddings"
    freeze_embeddings: bool = False
    
    # Dataset-specific configurations
    sts_benchmark_config: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://ixa2.si.ehu.eus/stswiki/images/4/48/Stsbenchmark.tar.gz",
        "filename": "stsbenchmark.tar.gz",
        "score_range": (0.0, 5.0),
        "normalize_scores": True
    })
    
    sick_config: Dict[str, Any] = field(default_factory=lambda: {
        "url": "http://clic.cimec.unitn.it/composes/materials/SICK.zip",
        "filename": "SICK.zip",
        "score_range": (1.0, 5.0),
        "normalize_scores": True
    })
    
    quora_config: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://qim.fs.quoracdn.net/quora_duplicate_questions.tsv",
        "filename": "quora_duplicate_questions.tsv",
        "score_range": (0.0, 1.0),
        "normalize_scores": False
    })
    
    mrpc_config: Dict[str, Any] = field(default_factory=lambda: {
        "url": "https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi",
        "filename": "MSRParaphraseCorpus.msi",
        "score_range": (0.0, 1.0),
        "normalize_scores": False
    })
    
    # Custom dataset settings
    custom_dataset_paths: Dict[str, str] = field(default_factory=dict)
    custom_dataset_formats: Dict[str, str] = field(default_factory=dict)
    
    # Performance optimization
    cache_processed_data: bool = True
    use_fast_tokenizer: bool = True
    parallel_processing: bool = True
    max_parallel_jobs: int = 4
    
    # Data validation
    validate_data: bool = True
    skip_invalid_samples: bool = True
    max_invalid_samples: int = 1000
    
    # Logging and monitoring
    log_data_stats: bool = True
    save_vocab_stats: bool = True
    save_dataset_samples: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_paths()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate splits
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Data splits must sum to 1.0"
        assert all(0 <= split <= 1 for split in [self.train_split, self.val_split, self.test_split]), \
            "All splits must be between 0 and 1"
        
        # Validate sequence lengths
        assert self.max_seq_length > self.min_seq_length > 0, \
            "max_seq_length must be greater than min_seq_length, both positive"
        
        # Validate vocabulary settings
        assert self.max_vocab_size > 0, "max_vocab_size must be positive"
        assert self.min_word_frequency >= 1, "min_word_frequency must be at least 1"
        
        # Validate batch sizes
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.eval_batch_size > 0, "eval_batch_size must be positive"
        
        # Validate tokenizer settings
        valid_tokenizer_types = ["word", "subword", "char", "bert"]
        assert self.tokenizer_type in valid_tokenizer_types, \
            f"tokenizer_type must be one of {valid_tokenizer_types}"
        
        # Validate embedding settings
        if self.use_pretrained_embeddings:
            assert self.embedding_dim > 0, "embedding_dim must be positive"
        
        # Validate augmentation settings
        if self.use_data_augmentation:
            assert 0 <= self.augmentation_prob <= 1, "augmentation_prob must be between 0 and 1"
    
    def _setup_paths(self):
        """Setup and create necessary directories."""
        # Create base paths
        base_paths = [
            self.get_raw_data_dir(),
            self.get_processed_data_dir(),
            self.get_embeddings_dir(),
            self.get_cache_dir()
        ]
        
        for path in base_paths:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get_data_root(self) -> Path:
        """Get data root directory path."""
        return Path(self.data_root)
    
    def get_raw_data_dir(self) -> Path:
        """Get raw data directory path."""
        return self.get_data_root() / self.raw_data_dir
    
    def get_processed_data_dir(self) -> Path:
        """Get processed data directory path."""
        return self.get_data_root() / self.processed_data_dir
    
    def get_embeddings_dir(self) -> Path:
        """Get embeddings directory path."""
        return self.get_data_root() / self.embeddings_dir
    
    def get_cache_dir(self) -> Path:
        """Get cache directory path."""
        return self.get_data_root() / self.cache_dir
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get configuration for specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset configuration dictionary
        """
        config_map = {
            "sts-benchmark": self.sts_benchmark_config,
            "sick": self.sick_config,
            "quora": self.quora_config,
            "mrpc": self.mrpc_config
        }
        
        if dataset_name not in config_map:
            if dataset_name in self.custom_dataset_paths:
                return {
                    "path": self.custom_dataset_paths[dataset_name],
                    "format": self.custom_dataset_formats.get(dataset_name, "tsv")
                }
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return config_map[dataset_name]
    
    def get_dataset_path(self, dataset_name: str, split: str = None) -> Path:
        """
        Get path for dataset file.
        
        Args:
            dataset_name: Name of the dataset
            split: Data split (train/val/test), if None returns base path
            
        Returns:
            Path to dataset file
        """
        base_path = self.get_processed_data_dir() / dataset_name
        
        if split is None:
            return base_path
        else:
            return base_path / f"{split}.json"
    
    def get_tokenizer_path(self) -> Path:
        """Get tokenizer save/load path."""
        return self.get_processed_data_dir() / "tokenizer.json"
    
    def get_vocab_path(self) -> Path:
        """Get vocabulary save/load path."""
        return self.get_processed_data_dir() / "vocab.json"
    
    def get_embedding_path(self, embedding_name: Optional[str] = None) -> Path:
        """
        Get pretrained embedding path.
        
        Args:
            embedding_name: Name of embedding (uses default if None)
            
        Returns:
            Path to embedding file
        """
        name = embedding_name or self.embedding_name
        return self.get_embeddings_dir() / f"{name}.txt"
    
    def add_custom_dataset(
        self, 
        name: str, 
        path: str, 
        format: str = "tsv"
    ) -> None:
        """
        Add custom dataset configuration.
        
        Args:
            name: Dataset name
            path: Path to dataset file
            format: Dataset format (tsv, csv, json)
        """
        self.custom_dataset_paths[name] = path
        self.custom_dataset_formats[name] = format
        
        if name not in self.datasets:
            self.datasets.append(name)
    
    def get_data_loader_config(self, is_training: bool = True) -> Dict[str, Any]:
        """
        Get data loader configuration.
        
        Args:
            is_training: Whether this is for training
            
        Returns:
            Data loader configuration
        """
        return {
            "batch_size": self.batch_size if is_training else self.eval_batch_size,
            "shuffle": self.shuffle_train if is_training else False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last if is_training else False
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'DataConfig':
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined data configurations
@dataclass
class FastDataConfig(DataConfig):
    """Fast data configuration for development and testing."""
    max_seq_length: int = 64
    max_vocab_size: int = 10000
    batch_size: int = 16
    num_workers: int = 2
    use_data_augmentation: bool = False
    cache_processed_data: bool = True


@dataclass
class ProductionDataConfig(DataConfig):
    """Production data configuration for full training."""
    max_seq_length: int = 256
    max_vocab_size: int = 50000
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 8
    use_data_augmentation: bool = True
    cache_processed_data: bool = True
    parallel_processing: bool = True


@dataclass
class LargeScaleDataConfig(DataConfig):
    """Large-scale data configuration for massive datasets."""
    max_seq_length: int = 512
    max_vocab_size: int = 100000
    batch_size: int = 32  # Smaller batch size for large sequences
    eval_batch_size: int = 64
    num_workers: int = 16
    use_data_augmentation: bool = True
    cache_processed_data: bool = True
    parallel_processing: bool = True
    max_parallel_jobs: int = 8


# Data configuration factory
def get_data_config(config_name: str = "default") -> DataConfig:
    """
    Get predefined data configuration.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Data configuration instance
    """
    configs = {
        "default": DataConfig,
        "fast": FastDataConfig,
        "production": ProductionDataConfig,
        "large_scale": LargeScaleDataConfig
    }
    
    if config_name not in configs:
        available_configs = list(configs.keys())
        raise ValueError(f"Unknown config name: {config_name}. Available: {available_configs}")
    
    return configs[config_name]()


# Dataset URL configurations
DATASET_URLS = {
    "sts-benchmark": {
        "url": "https://ixa2.si.ehu.eus/stswiki/images/4/48/Stsbenchmark.tar.gz",
        "filename": "stsbenchmark.tar.gz",
        "description": "Semantic Textual Similarity Benchmark"
    },
    "sick": {
        "url": "http://clic.cimec.unitn.it/composes/materials/SICK.zip",
        "filename": "SICK.zip",
        "description": "Sentences Involving Compositional Knowledge"
    },
    "quora": {
        "url": "https://qim.fs.quoracdn.net/quora_duplicate_questions.tsv",
        "filename": "quora_duplicate_questions.tsv",
        "description": "Quora Question Pairs"
    },
    "mrpc": {
        "url": "https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi",
        "filename": "MSRParaphraseCorpus.msi",
        "description": "Microsoft Research Paraphrase Corpus"
    }
}

# Embedding configurations
EMBEDDING_CONFIGS = {
    "glove.6B.50d": {
        "url": "https://nlp.stanford.edu/data/glove.6B.zip",
        "filename": "glove.6B.50d.txt",
        "dim": 50
    },
    "glove.6B.100d": {
        "url": "https://nlp.stanford.edu/data/glove.6B.zip",
        "filename": "glove.6B.100d.txt",
        "dim": 100
    },
    "glove.6B.200d": {
        "url": "https://nlp.stanford.edu/data/glove.6B.zip",
        "filename": "glove.6B.200d.txt",
        "dim": 200
    },
    "glove.6B.300d": {
        "url": "https://nlp.stanford.edu/data/glove.6B.zip",
        "filename": "glove.6B.300d.txt",
        "dim": 300
    },
    "word2vec.300d": {
        "url": "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz",
        "filename": "GoogleNews-vectors-negative300.bin.gz",
        "dim": 300
    },
    "fasttext.300d": {
        "url": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz",
        "filename": "cc.en.300.vec.gz",
        "dim": 300
    }
}


def get_config() -> DataConfig:
    """Get default data configuration."""
    return DataConfig()


if __name__ == "__main__":
    # Example usage and testing
    print("Data Configuration Examples:")
    print("=" * 50)
    
    # Default configuration
    config = get_data_config("default")
    print(f"Default config - Max seq length: {config.max_seq_length}, Batch size: {config.batch_size}")
    
    # Fast configuration for development
    fast_config = get_data_config("fast")
    print(f"Fast config - Max seq length: {fast_config.max_seq_length}, Batch size: {fast_config.batch_size}")
    
    # Production configuration
    prod_config = get_data_config("production")
    print(f"Production config - Max seq length: {prod_config.max_seq_length}, Batch size: {prod_config.batch_size}")
    
    # Test dataset configuration
    for dataset in ["sts-benchmark", "sick", "quora"]:
        dataset_config = config.get_dataset_config(dataset)
        print(f"{dataset} config: {dataset_config.get('url', 'Custom dataset')[:50]}...")
    
    # Test path generation
    print(f"Raw data dir: {config.get_raw_data_dir()}")
    print(f"Processed data dir: {config.get_processed_data_dir()}")
    print(f"STS dataset path: {config.get_dataset_path('sts-benchmark', 'train')}")
    
    # Test custom dataset
    config.add_custom_dataset("my_dataset", "/path/to/my/data.tsv", "tsv")
    print(f"Custom dataset added: {config.custom_dataset_paths}")
    
    print("\n✅ All data configuration examples completed successfully!")
