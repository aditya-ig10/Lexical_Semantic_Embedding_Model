"""
Data Download Module for Lexical Semantic Embedding Model.

This module handles downloading and initial processing of datasets
including STS Benchmark, SICK, Quora Question Pairs, and MRPC.

Author: AI Assistant
Date: September 2025
"""

import os
import logging
import requests
import tarfile
import zipfile
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from urllib.parse import urlparse
import hashlib

# Import project modules
from config.data_config import DataConfig
from scripts.utils import setup_logging, download_file, extract_archive, Timer

# Setup logging
logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Data downloader for semantic similarity datasets.
    
    This class handles downloading, extraction, and basic validation
    of various semantic similarity datasets.
    """
    
    def __init__(self, config: DataConfig, output_dir: Path):
        """
        Initialize data downloader.
        
        Args:
            config: Data configuration
            output_dir: Output directory for downloaded data
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.raw_data_dir = self.output_dir / "raw"
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = {
            "sts-benchmark": {
                "url": "https://ixa2.si.ehu.eus/stswiki/images/4/48/Stsbenchmark.tar.gz",
                "filename": "stsbenchmark.tar.gz",
                "expected_files": ["sts-train.csv", "sts-dev.csv", "sts-test.csv"],
                "description": "Semantic Textual Similarity Benchmark"
            },
            "sick": {
                "url": "http://clic.cimec.unitn.it/composes/materials/SICK.zip",
                "filename": "SICK.zip",
                "expected_files": ["SICK_train.txt", "SICK_trial.txt", "SICK_test_annotated.txt"],
                "description": "Sentences Involving Compositional Knowledge"
            },
            "quora": {
                "url": "https://qim.fs.quoracdn.net/quora_duplicate_questions.tsv",
                "filename": "quora_duplicate_questions.tsv",
                "expected_files": ["quora_duplicate_questions.tsv"],
                "description": "Quora Question Pairs"
            },
            "mrpc": {
                "url": "https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi",
                "filename": "MSRParaphraseCorpus.msi",
                "expected_files": ["msr_paraphrase_train.txt", "msr_paraphrase_test.txt"],
                "description": "Microsoft Research Paraphrase Corpus",
                "note": "Requires manual download and extraction"
            }
        }
        
        # Alternative URLs for datasets that might be down
        self.alternative_urls = {
            "sts-benchmark": [
                "https://github.com/PhilipMay/stsb-multi-mt/raw/main/data/stsbenchmark.tar.gz",
            ],
            "sick": [
                "https://raw.githubusercontent.com/alvations/SICK/master/SICK.zip",
            ]
        }
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """
        Download a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to download
            force: Whether to force re-download if dataset exists
            
        Returns:
            True if download was successful, False otherwise
        """
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_config = self.dataset_configs[dataset_name]
        dataset_dir = self.raw_data_dir / dataset_name
        
        # Check if dataset already exists
        if dataset_dir.exists() and not force:
            if self._validate_dataset(dataset_name):
                logger.info(f"Dataset {dataset_name} already exists and is valid")
                return True
            else:
                logger.warning(f"Dataset {dataset_name} exists but is invalid. Re-downloading...")
        
