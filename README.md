## Immediate Emergency Response System

On any given regular day, the threat detected by a security personnel is very rare and thus his chances of detecting a threat quick and responding to it and very low. So we can use Convulational Neural Networks to detect a threat within nanoseconds and respond to it by alerting all the concerned authorities.
## Final Output

![image](https://github.com/user-attachments/assets/66a679d9-4ae0-454a-aef8-1a956660bdcf)
![Screenshot from 2024-11-28 08-10-28](https://github.com/user-attachments/assets/47ec69da-9689-4cef-bb01-c160d52e567a)

## Model for Object Classification and Dangerous Object Detection

```python
  import os
import cv2
import numpy as np
import torch
from io import BytesIO
from base64 import b64decode
from PIL import Image as PILImage
from IPython.display import display, Image as IPImage
from IPython.display import Javascript
from google.colab.output import eval_js
from transformers import (
    AutoModelForObjectDetection,
    AutoFeatureExtractor,
    pipeline
)
import firebase_admin
from firebase_admin import credentials, storage
from datetime import datetime
import pytz

local_tz = pytz.timezone('Asia/Kolkata')
people = ['balaje', 'lipin']

# Initialize Firebase Admin SDK
firebase_admin.delete_app(firebase_admin.get_app())  # Ensure Firebase is initialized fresh
cred = credentials.Certificate("firebase.json")  # Path to Firebase credentials
firebase_admin.initialize_app(cred, {
    'storageBucket': 'chumma-cafe.appspot.com'  # Your Firebase Storage bucket
})

n = 0

# Load pre-trained LBPH face recognizer model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the pre-trained model from the yml file
face_recognizer.read("face_trained.yml")  # Replace with the actual path to your .yml file

def recognize_faces(frame):
    """
    Recognize faces in the given frame using the trained LBPH recognizer.
    Args:
        frame (numpy.array): The input image/frame.
    Returns:
        list: A list of tuples (x, y, w, h, label) for each detected face.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    recognized_faces = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(face_roi)
        label = people[label]
        recognized_faces.append((x, y, w, h, label, confidence))
        
    return recognized_faces

def get_camera_frame():
    """
    Captures a frame from the webcam using JavaScript.
    Returns:
        bytes: Raw image data in bytes format.
    """
    display(Javascript(''' 
        async function capture() {
            const video = document.createElement('video');
            document.body.appendChild(video);

            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            await video.play();

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
            video.srcObject.getTracks().forEach(track => track.stop());
            document.body.removeChild(video);
            return dataUrl;
        }
        capture();
    ''')) 
    data = eval_js('capture()')
    _, encoded = data.split(",", 1)
    return b64decode(encoded)

class ImageObjectDetector:
    def __init__(self, model_name="facebook/detr-resnet-50", output_dir="image_detection_output"):
        """
        Initializes the object detector and sets up the model.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the model and feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name).to(self.device)

        # Create the detection pipeline
        self.detector = pipeline(
            "object-detection",
            model=self.model,
            feature_extractor=self.feature_extractor
        )

    def detect_objects(self, image_buffer):
        """
        Detects objects in the image.
        """
        timestamp = datetime.now(local_tz).strftime("%H:%M:%S")
        global n
        np_image = np.frombuffer(image_buffer, np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image buffer.")

        # Detect objects
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_frame)

        results = self.detector(pil_image, threshold=0.5)

        for detection in results:
            label = detection['label']
            score = detection['score']
            box = detection['box']

            # Draw bounding box
            x_min, y_min, x_max, y_max = (
                int(box['xmin']), int(box['ymin']),
                int(box['xmax']), int(box['ymax'])
            )
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Add label and confidence score
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Face recognition
        recognized_faces = recognize_faces(frame)
        for (x, y, w, h, label, confidence) in recognized_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box for faces
            cv2.putText(frame, f"ID: {label} Conf: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        output_image_path = os.path.join(self.output_dir, f"detect_objects_{timestamp}.jpg")
        n += 1
        cv2.imwrite(output_image_path, frame)
        print(f"Output saved to: {output_image_path}")

        # Upload the output image to Firebase Storage
        self.upload_to_firebase(output_image_path)

        return output_image_path

    def upload_to_firebase(self, image_path):
        """
        Uploads the image to Firebase Storage.
        """
        bucket = storage.bucket()
        blob = bucket.blob(f"images/{os.path.basename(image_path)}")
        blob.upload_from_filename(image_path)
        blob.make_public()
        print(f"Image uploaded to Firebase Storage: {blob.public_url}")

def main():
    # Capture webcam frame and detect objects
    while True:
        frame_data = get_camera_frame()

        # Decode and detect objects
        img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        # Encode image to send as buffer
        _, encoded_img = cv2.imencode('.jpg', img)
        img_buffer = BytesIO(encoded_img.tobytes()).getvalue()

        # Initialize object detection
        detector = ImageObjectDetector()

        # Perform object detection
        output_image_path = detector.detect_objects(img_buffer)

        # Display annotated image
        with open(output_image_path, "rb") as img_file:
            display(IPImage(data=img_file.read()))

if __name__ == "__main__":
    main()
