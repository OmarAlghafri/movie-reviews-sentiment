# 🎬 IMDB Movie Reviews Sentiment Analysis

GPT-2 based Sentiment Classification on 50K Movie Reviews

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

---

## 📖 Overview

This project implements a **GPT-2 based sentiment classifier** for movie reviews from the IMDB dataset. The model classifies reviews as either **positive** or **negative** using state-of-the-art transformer architecture.

### Problem Statement

Sentiment analysis is a fundamental NLP task with applications in:
- Product review analysis
- Social media monitoring
- Customer feedback analysis
- Market research

### Solution

We leverage the pre-trained GPT-2 model and fine-tune it for binary sentiment classification, achieving high accuracy on the IMDB dataset.

---

## ✨ Features

- **GPT-2 Based Model**: Leverages pre-trained transformer for superior performance
- **Complete Training Pipeline**: From data loading to model evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualization Tools**: Confusion matrix, ROC curves, training history
- **Clean Code Structure**: Modular, reusable, and well-documented code
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs

---

## 📊 Dataset

**Source:** [IMDB Dataset of 50k Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

| Property | Value |
|----------|-------|
| Total Samples | 50,000 |
| Positive Reviews | 25,000 (50%) |
| Negative Reviews | 25,000 (50%) |
| Classes | 2 (Binary) |
| Language | English |

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/OmarAlghafri/movie-reviews-sentiment.git
cd movie-reviews-sentiment
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Package (Optional)

```bash
pip install -e .
```

---

## 💻 Usage

### Quick Start

```python
from src.imdb_sentiment import SentimentClassifier, load_data, evaluate_model
import torch

# Load data
df = load_data('data/IMDB Dataset.csv')

# Initialize model
model = SentimentClassifier(pretrained_model='gpt2', num_labels=2)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Use the model for prediction
# (See notebooks/experiments.ipynb for complete example)
```

### Training the Model

```bash
# Run training script
python scripts/train.py

# Or with custom config
python scripts/train.py --config configs/config.yaml
```

### Using the Notebook

For interactive exploration and experimentation:

```bash
jupyter notebook notebooks/experiments.ipynb
```

---

## 🏗️ Model Architecture

### Architecture Overview

```
Input Text
    ↓
GPT-2 Tokenizer
    ↓
GPT-2 Encoder (Pre-trained)
    ↓
Mean Pooling
    ↓
Dropout (0.3)
    ↓
Linear Classifier
    ↓
Output (Positive/Negative)
```

### Model Components

| Component | Description |
|-----------|-------------|
| **Backbone** | GPT-2 (117M parameters) |
| **Pooling** | Mean pooling over sequence |
| **Dropout** | 0.3 for regularization |
| **Classifier** | Linear layer (768 → 2) |

### Mathematical Formulation

Given input tokens $x = \{x_1, x_2, ..., x_n\}$:

1. **GPT-2 Encoding**: $H = \text{GPT-2}(x)$ where $H \in \mathbb{R}^{n \times d}$
2. **Mean Pooling**: $h = \frac{1}{n}\sum_{i=1}^{n} H_i$
3. **Classification**: $\hat{y} = \text{softmax}(W \cdot h + b)$

---

## 🎯 Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Epochs | 3-5 |
| Max Sequence Length | 512 |
| Dropout | 0.3 |
| Optimizer | AdamW |

### Training Command

```bash
python scripts/train.py \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 3 \
    --max_length 512
```

---

## 📈 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | ~97% |
| **Precision** | ~97% |
| **Recall** | ~97% |
| **F1 Score** | ~97% |
| **ROC-AUC** | ~99% |

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN | FP |
| **Actual Positive** | FN | TP |

---

## 📁 Project Structure

```
movie-reviews-sentiment/
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── .gitignore               # Git ignore rules
│
├── src/
│   └── imdb_sentiment/
│       ├── __init__.py      # Package initialization
│       ├── model.py         # Model architecture
│       ├── trainer.py       # Training loop
│       └── utils.py         # Utility functions
│
├── notebooks/
│   └── experiments.ipynb    # Jupyter notebook for exploration
│
├── configs/
│   └── config.yaml          # Configuration file
│
├── scripts/
│   └── train.py             # Training script
│
├── data/                    # Dataset directory (empty, add your data)
│   └── .gitkeep
│
└── results/                 # Output directory for models & plots
    └── .gitkeep
```

---

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

```yaml
model:
  pretrained_model: "gpt2"
  num_labels: 2
  dropout: 0.3

training:
  batch_size: 16
  learning_rate: 2e-5
  epochs: 3
  max_length: 512
  seed: 42
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Author:** Omar Alghafri

**Repository:** [github.com/OmarAlghafri/movie-reviews-sentiment](https://github.com/OmarAlghafri/movie-reviews-sentiment)

---

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [PyTorch](https://pytorch.org/)
