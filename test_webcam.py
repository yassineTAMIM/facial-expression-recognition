import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import sys
from pathlib import Path

sys.path.append('src')
from model import EmotionCNN


class EmotionDetector:
    """Real-time emotion detection using webcam."""
    
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 255, 255),  # Yellow
            'Fear': (255, 0, 255),     # Magenta
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (0, 165, 255), # Orange
            'Neutral': (128, 128, 128) # Gray
        }
        
        # Load model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = EmotionCNN(num_classes=7).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print(f"Model loaded on {self.device}")
        print("Press 'q' to quit")
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image."""
        # Convert to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Preprocess
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion = self.classes[predicted.item()]
        conf = confidence.item()
        
        return emotion, conf
    
    def run(self):
        """Run real-time emotion detection."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\nWebcam opened successfully!")
        print("Show your face to the camera...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence = self.predict_emotion(face_roi)
                
                # Get color for emotion
                color = self.colors.get(emotion, (255, 255, 255))
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw emotion label
                label = f"{emotion}: {confidence*100:.1f}%"
                
                # Background for text
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame,
                    (x, y - text_height - 10),
                    (x + text_width, y),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Display info
            cv2.putText(
                frame,
                "Press 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                frame,
                f"Faces detected: {len(faces)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show frame
            cv2.imshow('Facial Expression Recognition', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam closed. Goodbye!")


def main():
    try:
        detector = EmotionDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
