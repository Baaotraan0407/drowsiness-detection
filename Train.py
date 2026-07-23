import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


DATASET_DIR = "datasets"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_models(eye_model_path='eye_detector.h5', mouth_model_path='mouth_detector.h5'):
    
    eye_train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        validation_split=0.2
    )

    mouth_train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.15,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    eye_train = eye_train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "eyes"),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        classes=['Close-Eyes', 'Open-Eyes'],
        subset='training',
        seed=42
    )

    eye_val = eye_train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "eyes"),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        classes=['Close-Eyes', 'Open-Eyes'],
        subset='validation',
        seed=42
    )

    mouth_train = mouth_train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "mouth"),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        classes=['no yawn', 'yawn'],
        subset='training',
        seed=42
    )

    mouth_val = mouth_train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "mouth"),
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        classes=['no yawn', 'yawn'],
        subset='validation',
        seed=42
    )

    #  Xây dựng model
    def build_eye_model():
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_mouth_model():
        model = Sequential([
            Conv2D(32, (5,5), activation='relu', input_shape=(64,64,3)),
            MaxPooling2D(2,2),
            
            Conv2D(64, (5,5), activation='relu'),
            MaxPooling2D(2,2),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Huấn luyện và lưu model
    print("Training Eye Model")
    eye_model = build_eye_model()
    eye_history = eye_model.fit(
        eye_train,
        validation_data=eye_val,
        epochs=50,
        callbacks=[ 
            EarlyStopping(patience=5, monitor='val_loss'),
            ModelCheckpoint(os.path.join(MODEL_DIR, eye_model_path), 
                          save_best_only=True)
        ]
    )
    
    print("Training Mouth Model")
    mouth_model = build_mouth_model()
    mouth_history = mouth_model.fit(
        mouth_train,
        validation_data=mouth_val,
        epochs=50,
        callbacks=[ 
            EarlyStopping(patience=5, monitor='val_loss'),
            ModelCheckpoint(os.path.join(MODEL_DIR, mouth_model_path),
                          save_best_only=True)
        ]
    )

  # Image về accuracy and Validation Accuracy 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(eye_history.history['accuracy'], label='Train Accuracy')
    plt.plot(eye_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Eye Model Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(mouth_history.history['accuracy'], label='Train Accuracy')
    plt.plot(mouth_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Mouth Model Accuracy')
    plt.legend()
    plt.show()

    return eye_history, mouth_history

if __name__ == "__main__":
    train_models()
