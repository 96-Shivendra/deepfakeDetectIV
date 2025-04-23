import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def create_data_generators(train_dir, val_dir, batch_size=32, target_size=(128, 128)):
    """
    Create data generators for training and validation data.
    Handles a directory structure where real/ and fake/ folders may contain subfolders of frames.
    
    Args:
        train_dir: Directory containing training data with 'real' and 'fake' subdirectories
        val_dir: Directory containing validation data with 'real' and 'fake' subdirectories
        batch_size: Batch size for training
        target_size: Target size for images (height, width)
        
    Returns:
        train_generator, validation_generator
    """
    # Define image augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescale for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators with class_mode='binary' for binary classification (real vs fake)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['real', 'fake'],  # Explicitly define class mapping: real=0, fake=1
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['real', 'fake'],  # Explicitly define class mapping: real=0, fake=1
        shuffle=False
    )
    
    return train_generator, val_generator

def create_callbacks(checkpoint_path):
    """
    Create callbacks for model training
    
    Args:
        checkpoint_path: Path to save model checkpoints
        
    Returns:
        List of callbacks
    """
    # Make sure checkpoint path ends with .keras
    if not checkpoint_path.endswith('.keras'):
        # Replace .h5 with .keras if present
        if checkpoint_path.endswith('.h5'):
            checkpoint_path = checkpoint_path.replace('.h5', '.keras')
        else:
            checkpoint_path = checkpoint_path + '.keras'
    
    # Create directory for checkpoint if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Checkpoint to save best model
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy', 
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]

def load_image(image_path):
    """
    Load and preprocess an image for the deepfake detection model
    """
    import cv2
    import numpy as np
    
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to RGB (from BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to expected input dimensions (128x128 based on your code)
        img_resized = cv2.resize(img_rgb, (128, 128))
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_resized / 255.0
        
        return img_normalized
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: History object returned from model.fit()
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()