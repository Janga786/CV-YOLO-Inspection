import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
from google.colab import drive, files
import io

# Mount Google Drive (optional, if you want to save/load from Drive)
# drive.mount('/content/drive')

# --- Configuration ---
IMAGE_SIZE = (128, 128)  # Resize images to this size
BATCH_SIZE = 32
EPOCHS = 50  # Can use more epochs with GPU
DATASET_PATH = Path("/content/synthetic_dataset")
MODEL_SAVE_PATH = Path("/content/autoencoder_model")

# --- Setup Functions ---
def upload_dataset():
    """Upload and extract dataset zip file"""
    print("Please upload your synthetic_dataset.zip file:")
    uploaded = files.upload()
    
    # Extract the uploaded zip file
    for filename in uploaded.keys():
        print(f'Extracting {filename}...')
        with zipfile.ZipFile(io.BytesIO(uploaded[filename]), 'r') as zip_ref:
            zip_ref.extractall('/content/')
    
    print("Dataset extracted successfully!")

def check_gpu():
    """Check if GPU is available"""
    print("GPU Available: ", tf.config.list_physical_devices('GPU'))
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU for training")
    else:
        print("Using CPU for training")

# --- Image Loading and Preprocessing ---
def create_dataset(data_dir):
    """Create TensorFlow dataset from images"""
    if not data_dir.exists():
        print(f"Dataset directory {data_dir} not found!")
        return None
    
    # Look for images directory
    images_dir = data_dir / "images"
    if not images_dir.exists():
        print(f"Images directory {images_dir} not found!")
        return None
    
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),  # Point to the parent directory of 'images'
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
    """Build the convolutional autoencoder model"""
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
    decoded = layers.Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(encoder_input, decoded)
    return autoencoder

# --- Training Function ---
def train_autoencoder():
    """Main training function"""
    # Check GPU availability
    check_gpu()
    
    # Load dataset
    print(f"Loading images from: {DATASET_PATH}")
    dataset = create_dataset(DATASET_PATH)
    if dataset is None:
        print("Exiting training as no images were found.")
        return
    
    # Map the dataset to (input, target) pairs for autoencoder training
    dataset = dataset.map(lambda x: (x, x))
    
    # Split dataset into train and validation
    dataset_size = sum(1 for _ in dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    # Prefetch for better performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Get input shape from the first batch
    for image_batch, _ in train_dataset.take(1):
        input_shape = image_batch.shape[1:]
        break
    else:
        print("Could not determine input shape from dataset. Is it empty?")
        return

    print(f"Detected input image shape: {input_shape}")
    print(f"Training batches: {train_size}, Validation batches: {val_size}")
    
    # Build and compile model
    autoencoder = build_autoencoder(input_shape)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    autoencoder.summary()

    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_SAVE_PATH / "best_model.h5"),
            monitor='val_loss',
            save_best_only=True
        )
    ]

    print("\nStarting autoencoder training...")
    history = autoencoder.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    autoencoder.save(MODEL_SAVE_PATH / "final_model")
    print(f"\nAutoencoder model saved to: {MODEL_SAVE_PATH}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(MODEL_SAVE_PATH / "training_history.png")
    plt.show()
    
    print(f"Training history plot saved to: {MODEL_SAVE_PATH / 'training_history.png'}")
    
    return autoencoder, history

# --- Visualization Functions ---
def visualize_reconstructions(autoencoder, dataset, num_images=5):
    """Visualize original vs reconstructed images"""
    # Get a batch of images
    for images, _ in dataset.take(1):
        break
    
    # Get predictions
    reconstructions = autoencoder.predict(images[:num_images])
    
    # Plot comparisons
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i])
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(MODEL_SAVE_PATH / "reconstruction_comparison.png")
    plt.show()

def download_results():
    """Download the trained model and results"""
    # Create a zip file with results
    import shutil
    shutil.make_archive('/content/autoencoder_results', 'zip', str(MODEL_SAVE_PATH))
    files.download('/content/autoencoder_results.zip')

# --- Main execution ---
if __name__ == '__main__':
    print("=== Colab Autoencoder Training Script ===")
    print("Run the cells below in order:")

# %%
# Cell 1: Upload Dataset
print("=== CELL 1: Upload Dataset ===")
upload_dataset()

# %%
# Cell 2: Train Autoencoder
print("=== CELL 2: Train Autoencoder ===")
autoencoder, history = train_autoencoder()

# %%
# Cell 3: Visualize Results
print("=== CELL 3: Visualize Results ===")
# Load dataset for visualization
dataset = create_dataset(DATASET_PATH)
if dataset is not None:
    dataset = dataset.map(lambda x: (x, x))
    visualize_reconstructions(autoencoder, dataset)
else:
    print("Could not load dataset for visualization")

# %%
# Cell 4: Download Results
print("=== CELL 4: Download Results ===")
download_results()