"""
Dataset Download and Organization Script for FER-2013

This script helps organize the FER-2013 dataset into the required structure.

IMPORTANT: You must first download the dataset from:
https://www.kaggle.com/datasets/msambare/fer2013

After downloading, extract the ZIP file and run this script.
"""

import os
import shutil
from pathlib import Path


def organize_dataset(source_dir, target_dir="data"):
    """
    Organize FER-2013 dataset into train/val/test splits.
    
    Expected source structure (after extracting from Kaggle):
    source_dir/
        train/
            angry/
            disgust/
            ...
        test/
            angry/
            ...
    
    Target structure:
    data/
        train/
            angry/
            ...
        val/
            angry/
            ...
        test/
            angry/
            ...
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory not found: {source_dir}")
        print("\nPlease download FER-2013 from Kaggle:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
        return False
    
    print("Organizing FER-2013 dataset...")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print("-" * 60)
    
    classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Create target directories
    for split in ['train', 'val', 'test']:
        for cls in classes:
            (target_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Copy training data (80% train, 20% val split)
    train_source = source_path / 'train'
    if train_source.exists():
        print("Processing training data...")
        
        for cls in classes:
            cls_dir = train_source / cls
            if not cls_dir.exists():
                print(f"Warning: {cls} not found in train")
                continue
            
            images = list(cls_dir.glob('*.png')) + list(cls_dir.glob('*.jpg'))
            n_images = len(images)
            n_val = int(n_images * 0.2)  # 20% for validation
            
            # Split into train and val
            val_images = images[:n_val]
            train_images = images[n_val:]
            
            # Copy to train
            for img in train_images:
                shutil.copy2(img, target_path / 'train' / cls / img.name)
            
            # Copy to val
            for img in val_images:
                shutil.copy2(img, target_path / 'val' / cls / img.name)
            
            print(f"  {cls}: {len(train_images)} train, {len(val_images)} val")
    
    # Copy test data
    test_source = source_path / 'test'
    if test_source.exists():
        print("\nProcessing test data...")
        
        for cls in classes:
            cls_dir = test_source / cls
            if not cls_dir.exists():
                print(f"Warning: {cls} not found in test")
                continue
            
            images = list(cls_dir.glob('*.png')) + list(cls_dir.glob('*.jpg'))
            
            for img in images:
                shutil.copy2(img, target_path / 'test' / cls / img.name)
            
            print(f"  {cls}: {len(images)} test")
    
    print("\n" + "=" * 60)
    print("Dataset organization complete!")
    print("=" * 60)
    
    # Print summary
    print("\nDataset summary:")
    for split in ['train', 'val', 'test']:
        total = sum(len(list((target_path / split / cls).glob('*'))) 
                   for cls in classes)
        print(f"  {split}: {total} images")
    
    return True


def verify_dataset(data_dir="data"):
    """Verify dataset structure and count images."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return False
    
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    
    classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        split_total = 0
        
        for cls in classes:
            cls_path = data_path / split / cls
            if cls_path.exists():
                n_images = len(list(cls_path.glob('*')))
                split_total += n_images
                print(f"  {cls:10s}: {n_images:5d} images")
            else:
                print(f"  {cls:10s}: MISSING")
        
        print(f"  {'TOTAL':10s}: {split_total:5d} images")
    
    print("\n" + "=" * 60)
    return True


def main():
    print("=" * 60)
    print("FER-2013 DATASET SETUP")
    print("=" * 60)
    print()
    
    # Check if data already exists
    if Path("data/train").exists():
        print("Dataset already organized in 'data/' directory")
        response = input("Do you want to reorganize? (y/n): ").lower()
        if response != 'y':
            verify_dataset()
            return
    
    # Ask for source directory
    print("\nPlease download FER-2013 from:")
    print("https://www.kaggle.com/datasets/msambare/fer2013")
    print()
    
    source = input("Enter path to extracted dataset (or press Enter to skip): ").strip()
    
    if source:
        if organize_dataset(source):
            verify_dataset()
    else:
        print("\nSkipping organization.")
        print("Manual setup instructions:")
        print("1. Download from: https://www.kaggle.com/datasets/msambare/fer2013")
        print("2. Extract the ZIP file")
        print("3. Run this script again with the path to extracted folder")
        print()
        print("Expected structure after extraction:")
        print("  fer2013/")
        print("    train/")
        print("      angry/")
        print("      disgust/")
        print("      ...")
        print("    test/")
        print("      angry/")
        print("      ...")


if __name__ == "__main__":
    main()
