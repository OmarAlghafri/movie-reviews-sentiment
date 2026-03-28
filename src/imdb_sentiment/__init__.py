"""
IMDB Sentiment Analysis Package
GPT-2 based Sentiment Classification on 50K Movie Reviews
"""

__version__ = "1.0.0"
__author__ = "Omar Alghafri"

from .model import SentimentClassifier
from .utils import load_data, preprocess_text, evaluate_model

__all__ = ["SentimentClassifier", "load_data", "preprocess_text", "evaluate_model"]
