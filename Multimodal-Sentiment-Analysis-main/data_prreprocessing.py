import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import cv2
import librosa
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
dataset_path = "dataset/your-dataset-file.csv"
data = pd.read_csv(dataset_path)

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

data['cleaned_text'] = data['text'].apply(preprocess_text)

# Preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to 128x128
    return img

data['image_array'] = data['image_path'].apply(preprocess_image)

# Preprocess audio
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)  # Load audio at 16kHz
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.mean(axis=1)  # Take the mean for each MFCC coefficient

data['audio_features'] = data['audio_path'].apply(preprocess_audio)

# Split data
X_text = data['cleaned_text']
X_image = np.stack(data['image_array'])
X_audio = np.stack(data['audio_features'])
y = data['label']

X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
X_train_image, X_test_image = train_test_split(X_image, test_size=0.2, random_state=42)
X_train_audio, X_test_audio = train_test_split(X_audio, test_size=0.2, random_state=42)

# Save processed data
processed_data = {
    'X_train_text': X_train_text,
    'X_test_text': X_test_text,
    'X_train_image': X_train_image,
    'X_test_image': X_test_image,
    'X_train_audio': X_train_audio,
    'X_test_audio': X_test_audio,
    'y_train': y_train,
    'y_test': y_test,
}
np.save('processed_data.npy', processed_data)

print("Data preprocessing completed and saved.")
