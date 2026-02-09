import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class FER2013Dataset(Dataset):
    """
    FER-2013 Dataset loader.
    Expects data structure:
    data/
      train/
        angry/
        disgust/
        ...
      val/
      test/
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all images
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train'):
    """Get data augmentation transforms."""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


def get_data_loaders(data_dir, batch_size=64, num_workers=2):
    """Create train, validation, and test data loaders."""
    
    train_dataset = FER2013Dataset(
        root_dir=data_dir,
        split='train',
        transform=get_transforms('train')
    )
    
    val_dataset = FER2013Dataset(
        root_dir=data_dir,
        split='val',
        transform=get_transforms('val')
    )
    
    test_dataset = FER2013Dataset(
        root_dir=data_dir,
        split='test',
        transform=get_transforms('test')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    data_dir = "../data"
    
    if os.path.exists(data_dir):
        train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=32)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test one batch
        images, labels = next(iter(train_loader))
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
    else:
        print(f"Data directory not found: {data_dir}")
