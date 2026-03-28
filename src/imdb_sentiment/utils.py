"""
Utility functions for IMDB Sentiment Analysis
"""

import os
import re
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_path):
    """
    Load IMDB dataset from CSV file.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        df: pandas DataFrame with 'review' and 'sentiment' columns
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} samples")
    return df


def preprocess_text(text):
    """
    Basic text preprocessing.
    
    Args:
        text: Raw review text
        
    Returns:
        Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z\s.!?]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def encode_labels(df, label_column='sentiment'):
    """
    Encode sentiment labels to numeric values.
    
    Args:
        df: DataFrame with sentiment column
        label_column: Name of the sentiment column
        
    Returns:
        df: DataFrame with encoded labels
        label_map: Mapping from sentiment to numeric label
    """
    label_map = {'negative': 0, 'positive': 1}
    df['label'] = df[label_column].map(label_map)
    return df, label_map


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        threshold: Classification threshold
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs[1] if isinstance(outputs, tuple) else outputs
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'classification_report': classification_report(all_labels, all_preds, target_names=['negative', 'positive']),
        'predictions': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }
    
    return metrics


def plot_metrics(metrics, save_path=None):
    """
    Plot evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics from evaluate_model
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probs'])
    roc_auc_val = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('Receiver Operating Characteristic')
    axes[0, 1].legend(loc='lower right')
    
    # Accuracy bar chart
    acc_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    acc_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    axes[1, 0].bar(acc_metrics, acc_values, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].set_title('Performance Metrics')
    axes[1, 0].set_ylabel('Score')
    for i, v in enumerate(acc_values):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    # Prediction distribution
    axes[1, 1].hist(metrics['probs'], bins=50, alpha=0.7, label='Positive Probability')
    axes[1, 1].axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    axes[1, 1].set_xlabel('Predicted Probability (Positive)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Predicted Probabilities')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to: {save_path}")
    
    plt.show()


def print_metrics(metrics):
    """
    Print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics from evaluate_model
    """
    print("\n" + "="*50)
    print("📊 EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(metrics['classification_report'])
    print("="*50)
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("="*50)
