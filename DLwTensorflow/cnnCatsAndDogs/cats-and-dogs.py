"""
CNN Model for Cats vs Dogs Classification - Training from Scratch

INTENTION:
This script demonstrates how to build and train a Convolutional Neural Network (CNN) from scratch
to classify images of cats and dogs. It covers the complete workflow from data preparation to
model training and evaluation.

Key Learning Objectives:
- Data splitting and organization for image classification
- Image preprocessing and data augmentation
- Building CNN architecture with Conv2D, MaxPool2D, and Dense layers
- Training a CNN model from scratch
- Using callbacks for training optimization (EarlyStopping, ModelCheckpoint)
- Model evaluation and saving

Dataset: Cats and Dogs images
Data source: https://www.dropbox.com/s/h16vq9rab1itifs/CatDog.zip
"""

# Data in: https://www.dropbox.com/s/h16vq9rab1itifs/CatDog.zip
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import load_model
from skimage.io import imread, imshow

# ======================== DATA EXPLORATION ========================
# Check the original training data directory structure
train_dir = 'catdogdata/training_set'
d = {"dogs": 0, "cats": 0}

# Count images in each class directory
# This helps understand dataset size and balance
for dir in os.listdir(train_dir):
    path = os.path.join(train_dir, dir)
    d[dir] = len(os.listdir(path))
print("Dataset class distribution:", d)

# imshow(imread(os.path.join(train_dir, "dogs", "dog.1.jpg")))
# plt.show()


# ======================== DATA SPLITTING ========================
# Split data into training, validation, and test sets
# Why split:
#   - Training set: Used to train the model
#   - Validation set: Used during training to monitor performance and prevent overfitting
#   - Test set: Used only at the end for final unbiased evaluation

dir_name = ["train", "test", "val"]
ROOT_DIR = "./data_dir"  # Root directory where organized data will be stored

# Create directory structure if it doesn't exist
# Structure: data_dir/train/{cats,dogs}, data_dir/val/{cats,dogs}, data_dir/test/{cats,dogs}
if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)
    # Create subdirectories for each data split (train, test, val)
    for split_dir in dir_name:
        split_path = os.path.join(ROOT_DIR, split_dir)
        os.makedirs(split_path, exist_ok=True)
        # Create subdirectories for each class (cats, dogs) within each split
        for class_dir in os.listdir(train_dir):
            os.makedirs(os.path.join(split_path, class_dir), exist_ok=True)

# Copy and split data for each class (cats and dogs)
# Split strategy: 70% training, 15% validation, 15% test
# Why this split:
#   - 70% training: Sufficient data for learning patterns
#   - 15% validation: Enough to monitor generalization without wasting training data
#   - 15% test: Provides unbiased final evaluation
for dir in os.listdir(train_dir):
    source_path = os.path.join(train_dir, dir)
    all_images = os.listdir(source_path)
    
    # Shuffle images randomly to ensure balanced distribution across splits
    # Important: Shuffle before splitting to avoid bias (e.g., all early images in train)
    np.random.shuffle(all_images)
    n_total = len(all_images)
    
    # Calculate split sizes
    n_train = int(n_total * 0.7)   # 70% for training
    n_test = int(n_total * 0.15)    # 15% for testing
    
    # Split into three sets
    train_images = all_images[:n_train]
    test_images = all_images[n_train:n_train + n_test]
    val_images = all_images[n_train + n_test:]  # Remaining 15% for validation
    
    # Copy images to train directory
    train_dest = os.path.join(ROOT_DIR, "train", dir)
    os.makedirs(train_dest, exist_ok=True)
    for img in train_images:
        org = os.path.join(source_path, img)
        dest = os.path.join(train_dest, img)
        if os.path.exists(org):
            shutil.copy(org, dest)
    
    # Copy images to test directory
    test_dest = os.path.join(ROOT_DIR, "test", dir)
    os.makedirs(test_dest, exist_ok=True)
    for img in test_images:
        org = os.path.join(source_path, img)
        dest = os.path.join(test_dest, img)
        if os.path.exists(org):
            shutil.copy(org, dest)
    
    # Copy images to validation directory
    val_dest = os.path.join(ROOT_DIR, "val", dir)
    os.makedirs(val_dest, exist_ok=True)
    for img in val_images:
        org = os.path.join(source_path, img)
        dest = os.path.join(val_dest, img)
        if os.path.exists(org):
            shutil.copy(org, dest)

print("Data split complete")

# ======================== IMAGE PREPROCESSING AND DATA AUGMENTATION ========================
# ImageDataGenerator: Creates batches of preprocessed/augmented images on-the-fly during training
# Why use generators: Efficient memory usage, loads images in batches instead of all at once

# Training data generator with augmentation
# Augmentation increases dataset size and improves model robustness
trainingdatagenerator = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values [0,255] → [0,1]
    rotation_range=0.3,         # Random rotation up to 30% of 360° (108°)
    zoom_range=0.2,             # Random zoom in/out by 20%
    horizontal_flip=True,        # Randomly flip images horizontally (doubles data)
    vertical_flip=True           # Randomly flip images vertically (use carefully for animals)
)
# Parameters:
#   - target_size=(224, 224): Resize all images to 224×224 (required for CNN)
#   - class_mode="binary": Binary classification (cats=0, dogs=1)
train_data = trainingdatagenerator.flow_from_directory(
    ROOT_DIR + "/train", 
    target_size=(224, 224), 
    class_mode="binary"
)

# Validation data generator (NO augmentation - need original images for evaluation)
validationdatagenerator = ImageDataGenerator(rescale=1./255)  # Only normalization
val_data = validationdatagenerator.flow_from_directory(
    ROOT_DIR + "/val", 
    target_size=(224, 224), 
    class_mode="binary"
)

# Test data generator (NO augmentation - need original images for final evaluation)
testdatagenerator = ImageDataGenerator(rescale=1./255)  # Only normalization
test_data = testdatagenerator.flow_from_directory(
    ROOT_DIR + "/test", 
    target_size=(224, 224), 
    class_mode="binary"
)

# Display class mapping (how classes are encoded)
# class_indices: {'cats': 0, 'dogs': 1} or similar
print("Class indices: ", train_data.class_indices)
# Create reverse mapping: {0: 'cats', 1: 'dogs'} for predictions
class_id = {j: i for i, j in train_data.class_indices.items()}
print("Class ID: ", class_id)

# ======================== CNN MODEL ARCHITECTURE ========================
# Build Convolutional Neural Network from scratch
# Architecture pattern: Conv2D → Conv2D → MaxPool2D (repeated blocks) → Flatten → Dense

# Note: Since we're not using padding, images shrink with each convolution
# To compensate, we increase the number of filters in deeper layers
model = Sequential()

# Block 1: First convolutional layers (detect low-level features: edges, textures)
# Parameters:
#   - filters=16: Number of feature detectors (kernels)
#   - kernel_size=(3, 3): 3×3 convolution window
#   - activation="relu": ReLU activation (standard for CNNs)
#   - input_shape=(224, 224, 3): Input image dimensions (height, width, RGB channels)
# Output shape: (222, 222, 16) - slightly smaller due to no padding
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3)))

# Second Conv2D in block 1 (combines features from first layer)
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
# MaxPooling: Reduces spatial dimensions by half, increases receptive field
# Parameters:
#   - pool_size=(2, 2): 2×2 pooling window
#   - stride=2: Automatically set to pool_size (moves 2 pixels at a time)
# Output shape: (110, 110, 16) - dimensions halved
model.add(MaxPool2D(pool_size=(2, 2)))

# Block 2: Deeper layers (detect mid-level features: patterns, shapes)
# More filters (32) to capture more complex patterns
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))  # Output: (54, 54, 32)

# Block 3: Even deeper layers (detect high-level features: faces, body parts)
# Even more filters (64) for complex feature detection
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))  # Output: (26, 26, 64)

# Dropout regularization: Prevents overfitting in convolutional layers
# Parameters:
#   - rate=0.4: Randomly deactivate 40% of neurons during training
model.add(Dropout(0.4))

# Flatten: Convert 2D feature maps to 1D vector for dense layers
# Input: (26, 26, 64) → Output: (43,264) - flattened vector
model.add(Flatten())

# Dense (fully connected) layers for classification
# Parameters:
#   - units=64: 64 neurons in dense layer
#   - activation="relu": ReLU activation
model.add(Dense(units=64, activation="relu"))

# Another dropout layer before final classification
model.add(Dropout(0.4))

# Output layer: Binary classification
# Parameters:
#   - units=1: Single output neuron (probability of dog class)
#   - activation="sigmoid": Outputs probability [0, 1]
#     * Output < 0.5 → predict cat
#     * Output ≥ 0.5 → predict dog
model.add(Dense(units=1, activation="sigmoid"))

# Display model architecture summary
# Shows layer types, output shapes, and number of parameters
model.summary()

# Visualize model architecture as a diagram
# Saves to "model.png" showing layer connections and shapes
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

# ======================== MODEL COMPILATION ========================
# Compile the model: Configure how the model will be trained
# Parameters:
#   - optimizer="adam": Adaptive learning rate optimizer (better than SGD for most cases)
#   - loss="binary_crossentropy": Loss function for binary classification
#     * Measures difference between predicted probability and true binary label
#   - metrics=["accuracy"]: Track accuracy during training
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ======================== TRAINING CALLBACKS ========================
# Callbacks: Functions called during training to monitor and control the training process
# Available callbacks:
#   - EarlyStopping: Stop training when metric stops improving (prevents overfitting)
#   - ModelCheckpoint: Save model weights during training
#   - ReduceLROnPlateau: Reduce learning rate when stuck (helps convergence)
#   - TensorBoard: Log training metrics for visualization

# EarlyStopping: Automatically stop training if validation accuracy stops improving
# Parameters:
#   - monitor="val_accuracy": Watch validation accuracy metric
#   - min_delta=0.01: Minimum improvement (1%) to count as improvement
#   - patience=5: Wait 5 epochs without improvement before stopping
#   - verbose=1: Print messages when stopping
early_stopping = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=5, verbose=1)

# ModelCheckpoint: Save the best model during training
# Parameters:
#   - filepath="model.h5": Where to save the model
#   - monitor="val_accuracy": Save when validation accuracy improves
#   - save_best_only=True: Only save if better than previous best (saves disk space)
#   - verbose=1: Print when saving
model_checkpoint = ModelCheckpoint(filepath="model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

# Combine callbacks into a list
callbacks = [early_stopping, model_checkpoint]

# ======================== MODEL TRAINING ========================
# Train the model using data generators
# Parameters:
#   - generator=train_data: Training data generator (provides batches of images)
#   - epochs=30: Maximum number of training iterations (may stop earlier due to EarlyStopping)
#   - validation_data=val_data: Validation data generator (monitors generalization)
#   - callbacks=callbacks: List of callbacks (EarlyStopping, ModelCheckpoint)
#   - steps_per_epoch=32: Number of batches per epoch
#     * Calculated as: total_training_samples / batch_size
#     * If not specified, auto-calculates from generator
#   - validation_steps=16: Number of validation batches per epoch
#   - verbose=1: Show training progress (0=silent, 1=progress bar, 2=one line per epoch)
# Returns: history object containing training metrics for each epoch
history = model.fit_generator(
    generator=train_data, 
    epochs=30, 
    validation_data=val_data, 
    callbacks=callbacks, 
    steps_per_epoch=32, 
    validation_steps=16, 
    verbose=1
)

# ======================== MODEL EVALUATION ========================
# Load the best saved model (saved by ModelCheckpoint callback)
# This is the model with highest validation accuracy during training
final_model = tf.keras.models.load_model("model.h5")

# Evaluate the model on test set (unseen data)
# This provides final unbiased performance metric
# Returns: [loss, accuracy] on test data
final_model.evaluate_generator(generator=test_data)

# ======================== ESSENTIAL CONCEPTS SUMMARY ========================
"""
ESSENTIAL CONCEPTS IN THIS SCRIPT:

1. CONVOLUTIONAL NEURAL NETWORKS (CNNs):
   - Purpose: Process 2D image data while preserving spatial relationships
   - Key advantage: Translation invariance (can detect patterns anywhere in image)
   - Structure: Convolutional layers → Pooling layers → Dense layers

2. CONVOLUTIONAL LAYERS (Conv2D):
   - What: Apply filters (kernels) to detect features (edges, textures, patterns)
   - Filters: Small matrices that slide across the image
   - Output: Feature maps showing where features are detected
   - Why increase filters: Deeper layers need more filters for complex features

3. POOLING LAYERS (MaxPool2D):
   - What: Reduce spatial dimensions by taking maximum value in each region
   - Benefits: Reduces computation, prevents overfitting, increases receptive field
   - Common size: (2, 2) - halves dimensions

4. DATA AUGMENTATION:
   - What: Artificially increase dataset by transforming images
   - Transformations: Rotation, zoom, flipping, etc.
   - Why: Improves generalization, prevents overfitting
   - When: Only for training data, NOT for validation/test

5. DROPOUT REGULARIZATION:
   - What: Randomly deactivate neurons during training
   - Why: Prevents overfitting by forcing model to not rely on specific neurons
   - Rate: 0.4 means 40% of neurons randomly set to 0
   - Note: Automatically disabled during inference

6. FLATTEN LAYER:
   - What: Converts 2D feature maps to 1D vector
   - Why: Dense layers require 1D input
   - Example: (26, 26, 64) → (43,264)

7. TRAINING CALLBACKS:
   - EarlyStopping: Prevents overfitting by stopping when no improvement
   - ModelCheckpoint: Saves best model automatically
   - Benefits: Saves time, prevents overfitting, preserves best weights

8. DATA SPLITTING:
   - Training (70%): Learn patterns
   - Validation (15%): Monitor during training, prevent overfitting
   - Test (15%): Final unbiased evaluation
   - Why: Each set serves different purpose in model development

9. IMAGE PREPROCESSING:
   - Resize: All images must be same size (224×224)
   - Normalize: Pixel values [0,255] → [0,1] for better training
   - Why: Neural networks train better with normalized inputs

10. BINARY CLASSIFICATION:
    - Output: 1 neuron with sigmoid activation (probability [0,1])
    - Loss: binary_crossentropy
    - Decision: Threshold at 0.5 (≥0.5 = class 1, <0.5 = class 0)

KEY DIFFERENCES FROM DENSE NEURAL NETWORKS:
- CNNs preserve spatial structure (2D images)
- CNNs use weight sharing (same filter applied everywhere)
- CNNs have fewer parameters (more efficient)
- CNNs are translation-invariant (detect patterns anywhere)
"""
