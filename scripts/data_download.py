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
        
        # Create dataset directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {dataset_name}: {dataset_config['description']}")
        
        try:
            with Timer(f"Download {dataset_name}"):
                success = self._download_and_extract(dataset_name, dataset_config, dataset_dir)
            
            if success:
                logger.info(f"Successfully downloaded {dataset_name}")
                return True
            else:
                logger.error(f"Failed to download {dataset_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return False
    
    def _download_and_extract(
        self, 
        dataset_name: str, 
        config: Dict[str, Any], 
        output_dir: Path
    ) -> bool:
        """
        Download and extract a dataset.
        
        Args:
            dataset_name: Name of the dataset
            config: Dataset configuration
            output_dir: Output directory
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_name == "mrpc":
            return self._download_mrpc(config, output_dir)
        
        filename = config["filename"]
        file_path = output_dir / filename
        
        # Try primary URL first
        urls_to_try = [config["url"]]
        
        # Add alternative URLs if available
        if dataset_name in self.alternative_urls:
            urls_to_try.extend(self.alternative_urls[dataset_name])
        
        # Try downloading from each URL
        for url in urls_to_try:
            try:
                logger.info(f"Attempting to download from: {url}")
                self._download_file_with_progress(url, file_path)
                break
            except Exception as e:
                logger.warning(f"Failed to download from {url}: {e}")
                if url == urls_to_try[-1]:  # Last URL
                    raise
        
        # Extract if it's an archive
        if filename.endswith(('.tar.gz', '.tgz', '.tar', '.zip', '.gz')):
            extract_archive(file_path, output_dir, remove_archive=True)
        
        # Validate download
        return self._validate_dataset(dataset_name)
    
    def _download_file_with_progress(self, url: str, file_path: Path) -> None:
        """
        Download file with progress indication.
        
        Args:
            url: URL to download from
            file_path: Path to save the file
        """
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f:
            downloaded = 0
            start_time = time.time()
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress every 1MB or at the end
                    if downloaded % (1024 * 1024) == 0 or downloaded == total_size:
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            elapsed = time.time() - start_time
                            speed = downloaded / elapsed / 1024 / 1024 if elapsed > 0 else 0
                            print(f"\rDownload progress: {progress:.1f}% ({speed:.1f} MB/s)", 
                                  end="", flush=True)
        
        print()  # New line after progress
        logger.info(f"Downloaded {file_path.name} ({downloaded / 1024 / 1024:.1f} MB)")
    
    def _download_mrpc(self, config: Dict[str, Any], output_dir: Path) -> bool:
        """
        Handle MRPC dataset download (requires special handling).
        
        Args:
            config: Dataset configuration
            output_dir: Output directory
            
        Returns:
            True if successful, False otherwise
        """
        logger.warning("MRPC dataset requires manual download from Microsoft.")
        logger.info("Please download the dataset manually from:")
        logger.info(config["url"])
        logger.info("And extract the files to:")
        logger.info(str(output_dir))
        
        # Check if files are already present
        expected_files = config["expected_files"]
        missing_files = []
        
        for expected_file in expected_files:
            file_path = output_dir / expected_file
            if not file_path.exists():
                missing_files.append(expected_file)
        
        if missing_files:
            logger.error(f"Missing MRPC files: {missing_files}")
            return False
        else:
            logger.info("MRPC dataset files found")
            return True
    
    def _validate_dataset(self, dataset_name: str) -> bool:
        """
        Validate that a dataset was downloaded correctly.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if valid, False otherwise
        """
        if dataset_name not in self.dataset_configs:
            return False
        
        dataset_config = self.dataset_configs[dataset_name]
        dataset_dir = self.raw_data_dir / dataset_name
        
        if not dataset_dir.exists():
            return False
        
        # Check for expected files
        expected_files = dataset_config["expected_files"]
        
        for expected_file in expected_files:
            # Look for the file in dataset directory and subdirectories
            file_found = False
            
            for file_path in dataset_dir.rglob(expected_file):
                if file_path.is_file() and file_path.stat().st_size > 0:
                    file_found = True
                    break
            
            if not file_found:
                logger.warning(f"Missing or empty file: {expected_file}")
                return False
        
        logger.info(f"Dataset {dataset_name} validation passed")
        return True
    
    def download_all_datasets(self, force: bool = False) -> Dict[str, bool]:
        """
        Download all configured datasets.
        
        Args:
            force: Whether to force re-download existing datasets
            
        Returns:
            Dictionary with download results for each dataset
        """
        results = {}
        
        for dataset_name in self.config.datasets:
            logger.info(f"Processing dataset: {dataset_name}")
            try:
                results[dataset_name] = self.download_dataset(dataset_name, force)
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                results[dataset_name] = False
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"Download summary: {successful}/{total} datasets successful")
        
        for dataset_name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"  {status} {dataset_name}")
        
        return results
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset information dictionary
        """
        if dataset_name not in self.dataset_configs:
            return {}
        
        config = self.dataset_configs[dataset_name]
        dataset_dir = self.raw_data_dir / dataset_name
        
        info = {
            "name": dataset_name,
            "description": config["description"],
            "url": config["url"],
            "downloaded": dataset_dir.exists(),
            "valid": self._validate_dataset(dataset_name) if dataset_dir.exists() else False,
            "files": []
        }
        
        if dataset_dir.exists():
            # Get file information
            for file_path in dataset_dir.rglob("*"):
                if file_path.is_file():
                    file_info = {
                        "name": file_path.name,
                        "relative_path": str(file_path.relative_to(dataset_dir)),
                        "size_mb": file_path.stat().st_size / 1024 / 1024
                    }
                    info["files"].append(file_info)
        
        return info
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets with their status.
        
        Returns:
            List of dataset information dictionaries
        """
        datasets_info = []
        
        for dataset_name in self.dataset_configs.keys():
            info = self.get_dataset_info(dataset_name)
            datasets_info.append(info)
        
        return datasets_info
    
    def cleanup_dataset(self, dataset_name: str) -> bool:
        """
        Remove a downloaded dataset.
        
        Args:
            dataset_name: Name of the dataset to remove
            
        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_dir = self.raw_data_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.info(f"Dataset {dataset_name} not found")
            return True
        
        try:
            shutil.rmtree(dataset_dir)
            logger.info(f"Removed dataset: {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Error removing dataset {dataset_name}: {e}")
            return False


def main():
    """Main function for running data download from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download semantic similarity datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sts-benchmark", "sick", "quora"],
        help="Datasets to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download existing datasets"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing datasets"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("INFO")
    
    # Create data configuration
    config = DataConfig()
    config.datasets = args.datasets
    
    # Create downloader
    downloader = DataDownloader(config, Path(args.output_dir))
    
    if args.list:
        # List datasets
        datasets_info = downloader.list_datasets()
        print("\nAvailable Datasets:")
        print("=" * 50)
        
        for info in datasets_info:
            status = "✓ Downloaded" if info["downloaded"] and info["valid"] else "✗ Not downloaded"
            print(f"{info['name']}: {status}")
            print(f"  Description: {info['description']}")
            if info["files"]:
                total_size = sum(f["size_mb"] for f in info["files"])
                print(f"  Files: {len(info['files'])} ({total_size:.1f} MB)")
            print()
    
    elif args.validate:
        # Validate datasets
        print("\nValidating Datasets:")
        print("=" * 50)
        
        for dataset_name in args.datasets:
            is_valid = downloader._validate_dataset(dataset_name)
            status = "✓ Valid" if is_valid else "✗ Invalid"
            print(f"{dataset_name}: {status}")
    
    else:
        # Download datasets
        print("\nDownloading Datasets:")
        print("=" * 50)
        
        results = {}
        for dataset_name in args.datasets:
            results[dataset_name] = downloader.download_dataset(dataset_name, args.force)
        
        # Print summary
        print("\nDownload Results:")
        print("=" * 20)
        
        for dataset_name, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"{dataset_name}: {status}")


if __name__ == "__main__":
    main()
