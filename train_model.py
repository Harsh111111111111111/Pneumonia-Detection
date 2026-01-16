import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Path to dataset
dataset_path = "./datasets/chest_xray/"
# Image size for resizing
IMG_SIZE = 150

# Categories for classification
CATEGORIES = ['NORMAL', 'PNEUMONIA']

# Function to load and preprocess data
def load_data():
    data = []
    labels = []
    for category in CATEGORIES:
        path = os.path.join(dataset_path, 'train', category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append(img_resized)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    
    # Normalize data
    data = np.array(data) / 255.0
    labels = np.array(labels)
    return data, labels

# Load the data
data, labels = load_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape data for CNN input
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save('pneumonia_detection_model.h5')
print("Model trained and saved successfully!")
