# Facial Expression Recognition with Docker

A complete Deep Learning project for real-time facial expression recognition using PyTorch and Docker. Detects 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## 🎯 Project Overview

This project implements a CNN-based emotion classifier trained on the FER-2013 dataset. It demonstrates:
- Custom CNN architecture optimized for 48x48 grayscale images
- Complete training pipeline with validation and early stopping
- Model deployment via REST API
- Real-time webcam demo
- Full Docker containerization for reproducibility

## 📋 Requirements

### Hardware
- CPU: Intel i5 or equivalent (tested on i5-1145G7)
- RAM: 16GB recommended (minimum 8GB)
- Storage: 2GB free space
- Webcam (optional, for live demo)

### Software
- Windows 10/11 or Linux
- Docker Desktop (with WSL2 on Windows)
- Python 3.10+ (for local execution)
- Git

## 📁 Project Structure

```
facial-expression-recognition/
├── data/
│   ├── train/          # Training images (organized by class)
│   ├── val/            # Validation images
│   └── test/           # Test images
├── models/
│   └── best_model.pth  # Trained model checkpoint
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── predictions.png
│   ├── model_architecture.png
│   ├── classification_report.txt
│   └── metrics.json
├── src/
│   ├── model.py        # CNN architecture
│   ├── dataset.py      # Data loading and augmentation
│   ├── train.py        # Training script
│   └── utils.py        # Visualization and metrics
├── api.py              # Flask REST API
├── test_webcam.py      # Real-time webcam demo
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Step 1: Download Dataset

1. Download FER-2013 dataset from Kaggle:
   - https://www.kaggle.com/datasets/msambare/fer2013
   
2. Extract and organize into this structure:
```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Step 2: Local Training (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py
```

**Expected output:**
- Training time: ~35-40 minutes on CPU
- Validation accuracy: ~60-65%
- Model saved to `models/best_model.pth`
- Visualizations saved to `results/`

### Step 3: Docker Training

```bash
# Build Docker image
docker-compose build

# Run training
docker-compose up training

# Or run in detached mode
docker-compose up -d training

# Check logs
docker logs -f emotion-training
```

### Step 4: Deploy API

```bash
# Start API service
docker-compose up api

# Test the API
curl http://localhost:5000/health
```

**API Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict single image
- `POST /predict_batch` - Predict multiple images

**Example usage:**
```bash
# Predict emotion from image
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.8734,
  "all_probabilities": {
    "angry": 0.0123,
    "disgust": 0.0089,
    "fear": 0.0234,
    "happy": 0.8734,
    "sad": 0.0145,
    "surprise": 0.0421,
    "neutral": 0.0254
  }
}
```

### Step 5: Webcam Demo (Local Only)

```bash
# Run webcam demo (outside Docker)
python test_webcam.py
```

Press 'q' to quit the demo.

## 🐳 Docker Commands

```bash
# Build images
docker-compose build

# Run training
docker-compose up training

# Run API
docker-compose up api

# Run both
docker-compose up

# Stop all services
docker-compose down

# Remove all containers and volumes
docker-compose down -v

# View logs
docker logs emotion-training
docker logs emotion-api

# Access container shell
docker exec -it emotion-training bash
```

## 🔬 Model Architecture

```
Input: 48x48x1 grayscale image
├─ Conv2D(32) + BatchNorm + ReLU + MaxPool → 24x24x32
├─ Conv2D(64) + BatchNorm + ReLU + MaxPool → 12x12x64
├─ Conv2D(128) + BatchNorm + ReLU + MaxPool → 6x6x128
├─ Flatten → 4608
├─ Dense(512) + ReLU + Dropout(0.5)
└─ Dense(7) + Softmax
Output: 7 emotion classes

Total parameters: ~487,943
Model size: ~1.86 MB
```

## 📊 Performance Metrics

**Expected Results:**
- Training accuracy: ~70-75%
- Validation accuracy: ~60-65%
- Test accuracy: ~60-65%
- Training time (CPU): ~35-40 minutes
- Inference time: ~15ms per image

**Key Features:**
- Early stopping (patience=5)
- Learning rate scheduling
- Data augmentation (flip, rotation, brightness)
- Batch normalization
- Dropout regularization


## 📈 Results Visualization

After training, check the `results/` directory for:
- **training_curves.png** - Loss and accuracy over epochs
- **confusion_matrix.png** - 7x7 heatmap of predictions
- **predictions.png** - Sample predictions with labels
- **model_architecture.png** - Visual model diagram
- **classification_report.txt** - Detailed metrics per class
- **metrics.json** - All metrics in JSON format

## 🔄 Comparison: Local vs Docker

| Metric | Local | Docker |
|--------|-------|--------|
| Setup time | 5 min | 10 min |
| Training time | 35-40 min | 35-40 min |
| Reproducibility | Medium | High |
| Portability | Low | High |
| Dependency issues | Possible | None |

**Key advantage of Docker:**
- Guaranteed consistent environment
- Easy deployment to cloud/production
- No "works on my machine" issues
- Simple sharing with team members

## 🎓 MLOps Best Practices

This project demonstrates:
1. **Reproducibility**: Fixed random seeds, versioned dependencies
2. **Experiment tracking**: Training history saved to JSON
3. **Model versioning**: Checkpoints with metadata
4. **Containerization**: Full Docker support
5. **API deployment**: Production-ready Flask service
6. **Documentation**: Comprehensive README
7. **Code organization**: Modular structure
8. **Visualization**: Complete result analysis

## 📝 License

This project is for educational purposes as part of the Technologies IA course (3A-SDD 2025-2026).

## 👥 Authors

Yassine TAMIM - Zakaria LIMI