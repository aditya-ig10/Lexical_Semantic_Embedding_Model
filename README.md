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

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

1. **Clone the repository**:
```bash
git clone <repository-url>
cd lexical_embedding_project
```

2. **CPU Setup**:
```bash
# Build and run with Docker Compose
docker-compose up lexical-dev

# Access Jupyter Lab at http://localhost:8888
```

3. **GPU Setup** (requires NVIDIA Docker):
```bash
# Build and run with GPU support
docker-compose -f docker-compose.gpu.yml up lexical-dev-gpu

# Access Jupyter Lab at http://localhost:8888
```

### Option 2: Local Installation

1. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download datasets**:
```bash
python scripts/data_download.py --datasets all --output-dir data/raw
```

4. **Preprocess data**:
```bash
python scripts/preprocess.py --datasets sts-benchmark sick quora mrpc
```

5. **Train model**:
```bash
python scripts/train.py --config config/train_config.py --model-config medium
```

6. **Evaluate model**:
```bash
python scripts/evaluate.py --model-path models/best_model.pt --datasets sts-benchmark sick
```

## 📊 Usage Examples

### Command Line Interface

```bash
# Train a model
python main.py train --model-config large --epochs 50 --batch-size 64

# Evaluate a trained model
python main.py evaluate --model-path models/best_model.pt --datasets all

# Make predictions
python main.py infer --model-path models/best_model.pt \
  --sentence1 "The cat sat on the mat" \
  --sentence2 "A feline rested on the rug"

# Prepare data
python main.py prepare-data --datasets sts-benchmark sick --output-dir data/processed
```

### Python API

```python
from lexical_embedding_model import LexicalSemanticEmbeddingModel
from config.model_config import ModelConfig

# Load model
config = ModelConfig.get_config("medium")
model = LexicalSemanticEmbeddingModel(config)

