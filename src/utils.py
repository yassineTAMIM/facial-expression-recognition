import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
import json
from pathlib import Path


def plot_training_history(history, save_path='results/training_curves.png'):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path='results/confusion_matrix.png'):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def save_classification_report(y_true, y_pred, classes, save_path='results/classification_report.txt'):
    """Save detailed classification report."""
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"Classification report saved to {save_path}")
    return report


def save_metrics(metrics, save_path='results/metrics.json'):
    """Save metrics to JSON file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {save_path}")


def visualize_predictions(model, dataloader, classes, device, num_images=16, save_path='results/predictions.png'):
    """Visualize model predictions on sample images."""
    model.eval()
    
    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    images = images.cpu()
    predictions = predictions.cpu()
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx in range(num_images):
        img = images[idx].squeeze().numpy()
        img = (img * 0.5) + 0.5  # Denormalize
        
        true_label = classes[labels[idx]]
        pred_label = classes[predictions[idx]]
        
        color = 'green' if true_label == pred_label else 'red'
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', 
                           color=color, fontsize=9, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction visualization saved to {save_path}")


def plot_model_architecture(save_path='results/model_architecture.png'):
    """Create a simple model architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    layers = [
        "Input: 48x48x1",
        "Conv2D(32) + BN + ReLU + MaxPool → 24x24x32",
        "Conv2D(64) + BN + ReLU + MaxPool → 12x12x64",
        "Conv2D(128) + BN + ReLU + MaxPool → 6x6x128",
        "Flatten → 4608",
        "Dense(512) + ReLU + Dropout(0.5)",
        "Dense(7) + Softmax",
        "Output: 7 classes"
    ]
    
    y_pos = 0.9
    for i, layer in enumerate(layers):
        color = 'lightblue' if i % 2 == 0 else 'lightgreen'
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black', linewidth=2)
        ax.text(0.5, y_pos, layer, ha='center', va='center', 
                fontsize=11, fontweight='bold', bbox=bbox)
        y_pos -= 0.11
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title('Emotion CNN Architecture', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model architecture diagram saved to {save_path}")


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'max'
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop
