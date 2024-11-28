## Immediate Emergency Response System

On any given regular day, the threat detected by a security personnel is very rare and thus his chances of detecting a threat quick and responding to it and very low. So we can use Convulational Neural Networks to detect a threat within nanoseconds and respond to it by alerting all the concerned authorities.
## Final Output

![image](https://github.com/user-attachments/assets/66a679d9-4ae0-454a-aef8-1a956660bdcf)
![Screenshot from 2024-11-28 08-10-28](https://github.com/user-attachments/assets/47ec69da-9689-4cef-bb01-c160d52e567a)
![image](https://github.com/user-attachments/assets/4214973c-bb51-429b-8f3c-3b673a1080d7)


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
```
## Model for Dangerous Sound Detection(Gun shots)

```python
import os

# Define the directory name
directory_name = "audio_samples"

# Create the directory if it doesn't exist
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
    print(f"Directory '{directory_name}' created successfully.")
else:
    print(f"Directory '{directory_name}' already exists.")




# Directory containing the files
directory = "audio_samples"

# Create a dictionary to store the labels
labels = {}

# Iterate over the files in the directory
for filename in os.listdir(directory):
    # Assign label 1 to each file
    if(filename[0]=="3"):
    # print(type(filename[0]))
      labels[filename] = 1
    else:
      labels[filename] = 0

# Print the labels (optional)
print(labels)


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
!pip install tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class GunshotAudioCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GunshotAudioCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        features = self.features(x)
        features = features.squeeze(-1)
        output = self.classifier(features)
        return output

class GunshotAudioRecognizer:
    def __init__(self, audio_dir, sample_rate=22050, duration=2):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.label_encoder = LabelEncoder()
    def predict(self, audio_file_path):
       
        if self.model is None:
            raise ValueError("Model not trained yet. Call 'train' method first.")

        # Extract features from the audio file
        features = self.extract_features(audio_file_path)
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        # Make prediction
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = self.model(features)
            _, predicted_idx = torch.max(output, 1)
            predicted_label = self.label_encoder.inverse_transform([predicted_idx.item()])[0]

        return predicted_label



    def extract_features(self, file_path):
            waveform, sr = torchaudio.load(file_path)

            if sr != self.sample_rate:
              resampler = T.Resample(sr, self.sample_rate)
              waveform = resampler(waveform)

              target_length = self.sample_rate * self.duration
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            else:
              padding = target_length - waveform.shape[1]
              waveform = torch.nn.functional.pad(waveform, (0, padding))

            mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=13,
            melkwargs={'n_fft': 400, 'hop_length': 160}
        )
            mfccs = mfcc_transform(waveform)
            return mfccs.mean(dim=2).numpy().flatten()

    def prepare_dataset(self):
        features = []
        labels = []
        for audio_file in os.listdir(self.audio_dir):
          if os.path.isfile(os.path.join(self.audio_dir, audio_file)) and not audio_file.startswith('.'):
              file_path = os.path.join(self.audio_dir, audio_file)
              # print(audio_file[0])
                    
              feature = self.extract_features(file_path)
              features.append(feature)
              if(audio_file[0]=="3"):
                labels.append("Weapon")
              else:
                labels.append("Unknown")

        if not features or not labels:
            return None, None
        features=pad_sequences(features,dtype='float32',padding='post')
        features = np.array(features)
        labels = self.label_encoder.fit_transform(labels)

        return features, labels
    def train(self, test_size=0.2, batch_size=32, epochs=100, learning_rate=0.001):
        features, labels = self.prepare_dataset()

        if features is None or labels is None:
            print("Error: No data found in the specified directory.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            random_state=42
        )

        train_dataset = AudioDataset(X_train, y_train)
        test_dataset = AudioDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        num_classes = len(np.unique(labels))
        self.model = GunshotAudioCNN(input_size=features.shape[1], num_classes=num_classes)
        self.model = self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []

        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()

            self.model.eval()
            test_loss, test_correct, test_total = 0, 0, 0

            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)

                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_labels.size(0)
                    test_correct += (predicted == batch_labels).sum().item()

            train_losses.append(train_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))
            train_accuracies.append(100 * train_correct / train_total)
            test_accuracies.append(100 * test_correct / test_total)

            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%')
            print(f'Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%')

        return {
            'train_loss': train_losses,
            'test_loss': test_losses,
            'train_accuracy': train_accuracies,
            'test_accuracy': test_accuracies
        }
    def save_model(self, model_path):
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call 'train' method first.")

        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    def load_model(self, model_path):
        
        num_classes = len(self.label_encoder.classes_)  
        input_size = 13 #Assuming mfcc features of length 13
        self.model = GunshotAudioCNN(input_size=input_size, num_classes=num_classes)  # Create a new model instance
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print(f"Model loaded from: {model_path}")

# Example usage
recognizer = GunshotAudioRecognizer('/content/audio_samples')
history = recognizer.train()
new_audio_file = '/content/audio_samples/3 (90).wav' 
prediction = recognizer.predict(new_audio_file)
print(prediction)





```
