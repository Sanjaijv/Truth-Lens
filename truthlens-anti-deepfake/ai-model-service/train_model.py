import tensorflow as tf
import numpy as np
import os
import cv2
from models.tf_background_model import build_autoencoder

MODEL_PATH = 'trained_models/background_autoencoder.h5'
DATASET_ROOT = '/home/danico/Downloads/archive/Dataset'
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INPUT_SHAPE = (224, 224, 3)

def load_dataset(dataset_root, subfolder='real', target_size=INPUT_SHAPE[:2]):
    data_path = os.path.join(dataset_root, subfolder)
    if not os.path.exists(data_path):
        print(f"Error: Dataset subfolder not found at {data_path}.")
        return np.array([])

    print(f"Loading images from {data_path}...")
    images = []
    for filename in os.listdir(data_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(data_path, filename)
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.resize(img, target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                images.append(img)
            else:
                print(f"Warning: Could not read image file {filepath}")
    
    if not images:
        print(f"No images loaded from {data_path}.")
        return np.array([])

    images = np.array(images).astype(np.float32)
    print(f"Dataset loaded with shape: {images.shape}")
    return images

def train_model():
    print("Starting model training...")

    x_train = load_dataset(DATASET_ROOT, subfolder='real')

    if x_train.size == 0:
        print("No training data loaded. Exiting training.")
        return

    autoencoder = build_autoencoder(input_shape=INPUT_SHAPE)
    autoencoder.summary()

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')

    print(f"Training autoencoder for {EPOCHS} epochs with batch size {BATCH_SIZE}...")
    history = autoencoder.fit(
        x_train, x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2
    )
    print("Model training finished.")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    autoencoder.save(MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")

    return autoencoder

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    
    train_model()
