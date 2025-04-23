import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, Concatenate
)
from tensorflow.keras.applications import EfficientNetB0

def build_simple_cnn(input_shape=(128, 128, 3)):
    """
    Build a simple CNN model for deepfake detection
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        
    Returns:
        Compiled model
    """
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 4
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Classification block
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_efficientnet_model(input_shape=(128, 128, 3)):
    """
    Build a model based on EfficientNet for deepfake detection
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        
    Returns:
        Compiled model
    """
    # Load EfficientNet with pre-trained weights
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification layers
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_dual_input_model(input_shape=(128, 128, 3)):
    """
    Build a dual-input model that processes both spatial and frequency domain
    features for deepfake detection
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        
    Returns:
        Compiled model
    """
    # Spatial domain branch (RGB image)
    spatial_input = Input(shape=input_shape, name='spatial_input')
    
    # Spatial processing
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(spatial_input)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    
    x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalAveragePooling2D()(x1)
    
    # Frequency domain branch (can be applied to the same input)
    # This simulates frequency analysis through learnable filters
    x2 = Conv2D(32, (7, 7), activation='relu', padding='same')(spatial_input)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalAveragePooling2D()(x2)
    
    # Merge branches
    merged = Concatenate()([x1, x2])
    
    # Final classification layers
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=spatial_input, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model