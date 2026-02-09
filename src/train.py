import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
from tqdm import tqdm
import sys

sys.path.append('src')
from model import EmotionCNN, count_parameters
from dataset import get_data_loaders
from utils import (plot_training_history, plot_confusion_matrix, 
                   save_classification_report, save_metrics, 
                   visualize_predictions, plot_model_architecture, EarlyStopping)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def test(model, dataloader, device, classes):
    """Test the model and return predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return all_labels, all_preds


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Configuration
    config = {
    'data_dir': 'data',
    'batch_size': 32,              # Changed from 64
    'num_epochs': 40,              # Changed from 25
    'learning_rate': 0.0005,       # Changed from 0.001
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
    
    print("=" * 60)
    print("FACIAL EXPRESSION RECOGNITION - TRAINING")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print("=" * 60)
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        config['data_dir'], 
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Create model
    print("\nInitializing model...")
    device = torch.device(config['device'])
    model = EmotionCNN(num_classes=7).to(device)
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.3)  # More aggressive
    
    # Early stopping
    early_stopping = EarlyStopping(patience=7, mode='min')  # Changed from 5
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'models/best_model.pth')
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETED in {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)
    
    # Plot training curves
    print("\nGenerating visualizations...")
    plot_training_history(history)
    plot_model_architecture()
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    print("Testing on test set...")
    y_true, y_pred = test(model, test_loader, device, classes)
    
    # Calculate test accuracy
    test_acc = 100 * sum([1 for true, pred in zip(y_true, y_pred) if true == pred]) / len(y_true)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Generate visualizations
    plot_confusion_matrix(y_true, y_pred, classes)
    report = save_classification_report(y_true, y_pred, classes)
    print("\n" + report)
    
    visualize_predictions(model, test_loader, classes, device)
    
    # Save final metrics
    metrics = {
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'training_time_minutes': training_time / 60,
        'total_epochs': len(history['train_loss']),
        'total_parameters': count_parameters(model),
        'config': config
    }
    save_metrics(metrics)
    
    print("\n" + "=" * 60)
    print("ALL DONE! Results saved in 'results/' and 'models/' directories")
    print("=" * 60)


if __name__ == "__main__":
    main()
