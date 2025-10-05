# Lexical Semantic Embedding Model

A production-ready implementation of a Lexical Semantic Embedding Model using BiLSTM with Multi-Head Attention for semantic similarity tasks. This project provides a complete machine learning pipeline for training, evaluating, and deploying semantic similarity models.

## 🌟 Features

- **Modern Architecture**: BiLSTM encoder with multi-head attention mechanism
- **Multiple Similarity Functions**: Cosine, Euclidean, Manhattan, and learned similarity
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Docker Support**: Full containerization with CPU and GPU support
- **Interactive Notebooks**: Data exploration, training, evaluation, and demo notebooks
- **Multi-Dataset Support**: STS Benchmark, SICK, Quora Question Pairs, MRPC
- **Comprehensive Evaluation**: Pearson correlation, Spearman correlation, MSE, MAE
- **Visualization Tools**: Embedding space analysis and performance visualizations
- **Flexible Configuration**: Configurable model architectures and training parameters

## 🏗️ Architecture

The model implements a BiLSTM encoder with multi-head attention:

```
Input Text → Embedding → BiLSTM → Multi-Head Attention → Pooling → Similarity Computation
```

### Key Components:
- **Embedding Layer**: Learnable word embeddings with optional positional encoding
- **BiLSTM Encoder**: Bidirectional LSTM for sequence encoding
- **Multi-Head Attention**: Self-attention mechanism for capturing dependencies
- **Similarity Functions**: Multiple similarity computation methods
- **Classification/Regression Head**: Task-specific output layers

## 📁 Project Structure

```
lexical_embedding_project/
├── lexical_embedding_model.py      # Main model implementation
├── main.py                         # CLI entry point
├── test_model.py                   # Unit tests
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Standard Docker Compose
├── docker-compose.gpu.yml          # GPU Docker Compose
├── README.md                       # This file
│
├── config/                         # Configuration files
│   ├── model_config.py            # Model architecture settings
│   ├── data_config.py             # Dataset configurations
│   └── train_config.py            # Training hyperparameters
│
├── scripts/                        # Utility scripts
│   ├── data_download.py           # Dataset downloading
│   ├── preprocess.py              # Data preprocessing
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Evaluation suite
│   └── utils.py                   # Helper functions
│
├── notebooks/                      # Jupyter notebooks
│   ├── exploration.ipynb          # Data exploration
│   ├── evaluation.ipynb           # Model evaluation
│   └── demo.ipynb                 # Interactive demo
│
├── data/                          # Data directory
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Preprocessed data
│   └── embeddings/               # Pre-trained embeddings
│
├── models/                        # Saved models
├── logs/                          # Training logs
└── evaluation_results/           # Evaluation outputs
```
