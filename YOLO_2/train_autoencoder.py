import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# --- Configuration ---
IMAGE_SIZE = (128, 128)  # Resize images to this size
BATCH_SIZE = 32
EPOCHS = 20  # Reduced for faster training on CPU
DATASET_PATH = Path("/home/janga/YOLO_2/synthetic_dataset")
MODEL_SAVE_PATH = Path("./autoencoder_model")

# --- Image Loading and Preprocessing ---
def create_dataset(data_dir):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir.parent, # Point to the parent directory of 'images'
        labels=None,     # No labels needed for autoencoder input
        label_mode=None, # No labels needed
        image_size=IMAGE_SIZE,
        interpolation='bilinear',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42 # for reproducibility
    )
    # Normalize images to [0, 1]
    dataset = dataset.map(lambda x: x / 255.0)
    return dataset

# --- Autoencoder Model Definition ---
def build_autoencoder(input_shape):
    # Encoder
    encoder_input = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x) # Output layer

    autoencoder = keras.Model(encoder_input, decoded)
    return autoencoder

# --- Main Training Function ---
def train_autoencoder():
    print(f"Loading images from: {DATASET_PATH}")
    dataset = create_dataset(DATASET_PATH)
    if dataset is None:
        print("Exiting training as no images were found.")
        return
    
    # Map the dataset to (input, target) pairs for autoencoder training
    dataset = dataset.map(lambda x: (x, x))

    # Get input shape from the first batch
    for image_batch, _ in dataset.take(1):
        input_shape = image_batch.shape[1:]
        break
    else:
        print("Could not determine input shape from dataset. Is it empty?")
        return

    print(f"Detected input image shape: {input_shape}")
    autoencoder = build_autoencoder(input_shape)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    print("\nStarting autoencoder training...")
    history = autoencoder.fit(
        dataset,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=dataset, # Using the same dataset for validation for simplicity
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )

    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    autoencoder.save(MODEL_SAVE_PATH)
    print(f"\nAutoencoder model saved to: {MODEL_SAVE_PATH}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(MODEL_SAVE_PATH / "training_history.png")
    print(f"Training history plot saved to: {MODEL_SAVE_PATH / 'training_history.png'}")

if __name__ == '__main__':
    train_autoencoder()
