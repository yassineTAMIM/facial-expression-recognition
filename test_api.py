import requests
import json
from pathlib import Path

# API configuration
API_URL = "http://localhost:5000"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 60)


def test_root():
    """Test root endpoint."""
    print("Testing / endpoint...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 60)


def test_predict(image_path):
    """Test single image prediction."""
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Testing /predict with {image_path}...")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print("\nAll probabilities:")
        for emotion, prob in result['all_probabilities'].items():
            print(f"  {emotion}: {prob*100:.2f}%")
    else:
        print(f"Error: {response.json()}")
    
    print("-" * 60)


def test_predict_batch(image_paths):
    """Test batch prediction."""
    print(f"Testing /predict_batch with {len(image_paths)} images...")
    
    files = []
    for img_path in image_paths:
        if Path(img_path).exists():
            files.append(('files', open(img_path, 'rb')))
        else:
            print(f"Warning: Skipping {img_path} (not found)")
    
    if not files:
        print("Error: No valid images found")
        return
    
    response = requests.post(f"{API_URL}/predict_batch", files=files)
    
    # Close file handles
    for _, f in files:
        f.close()
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPredictions:")
        for pred in result['predictions']:
            print(f"  {pred['filename']}: {pred['emotion']} ({pred['confidence']*100:.2f}%)")
    else:
        print(f"Error: {response.json()}")
    
    print("-" * 60)


def main():
    print("=" * 60)
    print("FACIAL EXPRESSION RECOGNITION - API TEST")
    print("=" * 60)
    print()
    
    # Test basic endpoints
    test_root()
    test_health()
    
    # Test prediction (you need to provide sample images)
    print("\nTo test prediction endpoints:")
    print("1. Place test images in the project directory")
    print("2. Update the image paths below:")
    print()
    
    # Example: Uncomment and update paths
    # test_predict("sample_happy.jpg")
    # test_predict_batch(["sample_happy.jpg", "sample_sad.jpg", "sample_angry.jpg"])
    
    print("\nAPI is ready for testing!")
    print("Use the functions above with your own images.")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API")
        print("Make sure the API is running: docker-compose up api")
    except Exception as e:
        print(f"Error: {e}")
