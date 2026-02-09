import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    """
    Custom CNN for facial expression recognition on 48x48 grayscale images.
    7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    """
    
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 48x48 -> 24x24
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 24x24 -> 12x12
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 12x12 -> 6x6
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout2 = nn.Dropout(0.3)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 6 * 6)
        
        # FC layers
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model instantiation
    model = EmotionCNN(num_classes=7)
    print(f"Model created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 48, 48)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([1, 7])")
