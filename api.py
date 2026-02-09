from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import sys
from pathlib import Path

sys.path.append('src')
from model import EmotionCNN

app = Flask(__name__)

# Configuration
MODEL_PATH = 'models/best_model.pth'
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = None

def load_model():
    """Load the trained model."""
    global model
    
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    model = EmotionCNN(num_classes=7).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully from {MODEL_PATH}")
    print(f"Using device: {DEVICE}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE),
        'classes': CLASSES
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotion from uploaded image.
    
    Expected: multipart/form-data with 'file' field
    Returns: JSON with predicted emotion and probabilities
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Prepare response
        predicted_class = CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        all_probabilities = {
            CLASSES[i]: float(probabilities[0][i])
            for i in range(len(CLASSES))
        }
        
        response = {
            'emotion': predicted_class,
            'confidence': round(confidence_score, 4),
            'all_probabilities': {k: round(v, 4) for k, v in all_probabilities.items()}
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict emotions for multiple images.
    
    Expected: multipart/form-data with multiple 'files' fields
    Returns: JSON array with predictions
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({'error': 'Empty file list'}), 400
    
    try:
        results = []
        
        for file in files:
            # Read and preprocess image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = CLASSES[predicted.item()]
            confidence_score = confidence.item()
            
            results.append({
                'filename': file.filename,
                'emotion': predicted_class,
                'confidence': round(confidence_score, 4)
            })
        
        return jsonify({'predictions': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information."""
    return jsonify({
        'name': 'Facial Expression Recognition API',
        'version': '1.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict single image',
            '/predict_batch': 'POST - Predict multiple images'
        },
        'model_info': {
            'classes': CLASSES,
            'input_size': '48x48 grayscale',
            'device': str(DEVICE)
        }
    })


if __name__ == '__main__':
    try:
        load_model()
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error starting API: {e}")
        sys.exit(1)
