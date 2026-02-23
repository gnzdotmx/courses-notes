"""
Example: Pre-trained MobileNetV2 for Face Mask Detection

INTENTION:
This script builds a convolutional neural network for face mask detection using
transfer learning with MobileNetV2. It:
- Prepares training and test data generators from a directory of images
- Loads a MobileNetV2 backbone pre-trained on ImageNet
- Freezes the backbone (feature extractor) so its weights are not updated
- Adds a small custom classification head for the two classes: with_mask, without_mask
- Compiles the model and saves it to disk for later training or inference

Key Learning Objectives:
- Understand how to use Keras `ImageDataGenerator` to load images from folders
- Understand how to load a pre-trained model (MobileNetV2) as a fixed feature extractor
- Understand how to stack a custom head (Dense layers) on top of a pre-trained base
- Understand the key parameters: `input_shape`, `include_top`, `weights`, `trainable`
- Understand why we freeze the base model and only train the new head at first

NOTE:
- This script builds, compiles, and saves the model, but does not call `model.fit()`.
  In a typical workflow, you would import this saved model in another script and train
  it using `final_train` and `final_test` generators defined here.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, BatchNormalization, Activation
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras as keras
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import cv2

# ======================== DATASET AND DATA GENERATORS ========================

# Paths to image folders.
# DESCRIPTION:
#   - `train_data`: Folder with subfolders for each class (with_mask, without_mask)
#   - `test_data`: Folder with subfolders for each class (same structure as train)
# INPUTS:
#   - Directory structure:
#       dataset/
#         train/
#           with_mask/
#           without_mask/
#         test/
#           with_mask/
#           without_mask/
# OUTPUT:
#   - These paths are used by Keras `flow_from_directory` to read images on the fly.
# NOTES:
#   - Data originally taken from: https://github.com/prajnasb/observations.git
train_data = 'dataset/train'
test_data = 'dataset/test'

# ImageDataGenerator for training and testing.
# DESCRIPTION:
#   - `ImageDataGenerator` applies preprocessing and (optionally) data augmentation.
#   - For training we apply rescaling + simple augmentation.
#   - For testing we only rescale (no augmentation).
# INPUTS:
#   - `rescale=1./255`: Normalizes pixel values from [0, 255] to [0, 1].
#   - `shear_range=0.2`: Randomly applies shear transformations (data augmentation).
#   - `horizontal_flip=True`: Random horizontal flipping (augmentation).
# OUTPUT:
#   - Generator objects that can create batches of (image, label) pairs from directories.
# NOTES:
#   - Augmentation only on training set to improve generalization.
#   - Test set should be a realistic, non-augmented sample of real-world data.
train_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, horizontal_flip=True)
test_gen = ImageDataGenerator(rescale=1./255)

# Batch size for generators.
# DESCRIPTION:
#   - Number of images per batch produced by the generators.
# INPUTS:
#   - `batch_size=32`: Common default that balances memory usage and training speed.
# OUTPUT:
#   - Used by `flow_from_directory` for both training and test generators.
# NOTES:
#   - Larger batch sizes use more memory but can speed up training (if GPU memory allows).
#   - Smaller batch sizes can be more stable for some optimization problems.
batch_size = 32

# Create training data generator that yields batches of images and labels.
# DESCRIPTION:
#   - Reads images from `train_data` directory.
#   - Resizes images to (224, 224) to match MobileNetV2 expected input size.
#   - Uses one-hot encoded labels for 2 classes (categorical).
# INPUTS:
#   - `directory=train_data`: Path to training images.
#   - `target_size=(224, 224)`: Resize all images to 224x224 pixels.
#   - `batch_size=batch_size`: Number of images per batch.
#   - `class_mode='categorical'`: Labels returned as one-hot vectors (2 elements).
#   - `classes=['with_mask', 'without_mask']`: Explicit class names and order.
#   - `shuffle=True`: Shuffles images each epoch.
# OUTPUT:
#   - `final_train`: Keras generator yielding (batch_images, batch_labels).
# NOTES:
#   - Order of `classes` list defines index mapping in labels:
#       with_mask -> index 0, without_mask -> index 1.
final_train = train_gen.flow_from_directory(
    train_data,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['with_mask', 'without_mask'],
    shuffle=True,
)

# Create test data generator.
# DESCRIPTION:
#   - Similar to training generator but:
#       - Uses `test_data` directory.
#       - Does not specify `classes` explicitly (infers from folder names).
#       - Still shuffles data by default for evaluation.
# INPUTS:
#   - `directory=test_data`: Path to test images.
#   - `target_size=(224, 224)`: Resize to match model input size.
#   - `batch_size=batch_size`: Same as training for consistency.
#   - `class_mode='categorical'`: One-hot labels.
#   - `shuffle=True`: Shuffles test images each epoch.
# OUTPUT:
#   - `final_test`: Keras generator for validation/testing.
# NOTES:
#   - In a strict evaluation setup, you might set `shuffle=False` for reproducible metrics.
final_test = test_gen.flow_from_directory(
    test_data,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
)


# ======================== PRE-TRAINED MOBILENETV2 BASE ========================

# Load MobileNetV2 base model pre-trained on ImageNet.
# DESCRIPTION:
#   - Uses MobileNetV2 as a convolutional feature extractor.
#   - Loads weights trained on ImageNet (large-scale image classification dataset).
# INPUTS:
#   - `alpha=1.0`: Width multiplier (1.0 = standard MobileNetV2).
#   - `input_shape=(224, 224, 3)`: Expected input image size (height, width, channels).
#   - `include_top=False`: Excludes original classification head (we will add our own).
#   - `weights='imagenet'`: Loads pre-trained weights from ImageNet.
# OUTPUT:
#   - `pretrained_model`: Keras Model that outputs high-level feature maps.
# NOTES:
#   - `include_top=False` is essential for transfer learning: we only want convolutional base.
#   - `input_shape` must match `target_size` used in data generators.
pretrained_model = MobileNetV2(
    alpha=1.0,
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
)

# Freeze the pre-trained base model.
# DESCRIPTION:
#   - Prevents the weights of MobileNetV2 from being updated during initial training.
#   - We only train the new classification head (Dense layers on top).
# INPUTS:
#   - `pretrained_model.trainable = False`
# OUTPUT:
#   - All layers in `pretrained_model` are set to non-trainable.
# NOTES:
#   - Freezing helps:
#       - Avoid overfitting on small datasets.
#       - Preserve pre-trained features learned from ImageNet.
#   - Later, you could unfreeze some top layers for fine-tuning.
pretrained_model.trainable = False

# ======================== MODEL ARCHITECTURE (TRANSFER LEARNING HEAD) ========================

# Build a Sequential model that stacks:
#   1. The frozen MobileNetV2 base (`pretrained_model`)
#   2. A custom classification head for 2 classes
# DESCRIPTION:
#   - Sequential API is used for simplicity since we are stacking layers linearly.
# OUTPUT:
#   - `model`: Full Keras model ready to compile and train.
model = Sequential()

# Add pre-trained base model as first layer.
# NOTES:
#   - Input images will first pass through MobileNetV2 feature extractor.
model.add(pretrained_model)

# GlobalAveragePooling2D layer to reduce spatial dimensions.
# DESCRIPTION:
#   - Converts convolutional feature maps (H x W x C) into a 1D vector of length C
#     by taking the average of each feature map.
# INPUTS:
#   - Output of `pretrained_model` (feature maps).
# OUTPUT:
#   - 1D feature vector per image.
# NOTES:
#   - Reduces number of parameters compared to Flatten.
#   - Often used in modern architectures to prevent overfitting.
model.add(GlobalAveragePooling2D())

# Dense layer for learning task-specific features.
# DESCRIPTION:
#   - Fully connected layer with 128 units and ReLU activation.
# INPUTS:
#   - 1D feature vector from GlobalAveragePooling2D.
# PARAMETERS:
#   - `units=128`: Size of hidden layer.
#   - `activation='relu'`: Non-linear activation (Rectified Linear Unit).
# OUTPUT:
#   - 128-dimensional feature representation.
# NOTES:
#   - This layer forms the main part of the custom head.
model.add(Dense(128, activation='relu'))

# Dropout for regularization.
# DESCRIPTION:
#   - Randomly sets a fraction of input units to 0 during training.
# PARAMETERS:
#   - `rate=0.5`: 50% of the units are dropped.
# OUTPUT:
#   - Same shape as input, but with some units dropped during training.
# NOTES:
#   - Helps prevent overfitting, especially on small datasets.
model.add(Dropout(0.5))

# Final classification layer.
# DESCRIPTION:
#   - Dense layer with 2 units (one per class) + softmax activation.
# PARAMETERS:
#   - `units=2`: Number of output classes (with_mask, without_mask).
#   - `activation='softmax'`: Outputs probability distribution over classes.
# OUTPUT:
#   - For each image: 2 probabilities that sum to 1.
# NOTES:
#   - Index 0 corresponds to 'with_mask', index 1 to 'without_mask'
#     (because of the `classes` ordering in `final_train`).
model.add(Dense(2, activation='softmax'))

# ======================== MODEL COMPILATION AND SAVING ========================

# Compile the model with loss, optimizer, and metrics.
# DESCRIPTION:
#   - Configures the learning process for training.
# PARAMETERS:
#   - `optimizer='adam'`: Adam optimizer (adaptive learning rate).
#   - `loss='categorical_crossentropy'`: Suitable for multi-class classification
#       with one-hot encoded labels.
#   - `metrics=['accuracy']`: Tracks accuracy during training and evaluation.
# OUTPUT:
#   - Compiled model ready for training.
# NOTES:
#   - For binary classification with one-hot labels, `categorical_crossentropy`
#     is equivalent to `binary_crossentropy` applied to 2 outputs.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture.
# DESCRIPTION:
#   - Shows each layer, its output shape, and number of parameters.
# USES:
#   - Debugging: Verify architecture is as expected.
#   - Understanding: See how many parameters are trainable vs non-trainable.
model.summary()


# ======================== MODEL TRAINING (FIT MODEL) ========================

# Number of epochs (full passes over the training data).
# DESCRIPTION:
#   - Controls how many times the model sees the entire training dataset.
# INPUTS:
#   - `epochs = 5`: Train for 5 full passes over `final_train`.
# NOTES:
#   - More epochs can improve performance but increase training time
#     and risk of overfitting, especially on small datasets.
epochs = 5

# Fit the model on the training data.
# DESCRIPTION:
#   - Trains the model using the `final_train` generator and evaluates
#     on `final_test` at the end of each epoch.
# INPUTS:
#   - `final_train`: Training generator yielding (images, labels) batches.
#   - `epochs=epochs`: Number of epochs to train.
#   - `validation_data=final_test`: Generator for validation (test) data.
# OUTPUT:
#   - Trained model with updated weights in the classification head (and any
#     unfrozen layers, if you later unfreeze part of the base).
# NOTES:
#   - Because `pretrained_model.trainable = False`, only the new Dense layers
#     (head) are trained here.
#   - `model.fit` returns a History object with loss/accuracy curves that can
#     be used for plotting and analysis if stored in a variable.
model.fit(final_train, epochs=epochs, validation_data=final_test)


# Save the model for later use.
# DESCRIPTION:
#   - Serializes model architecture and weights to an HDF5 file.
# INPUTS:
#   - File path: 'pretrained_model.h5'
# OUTPUT:
#   - File saved to disk containing the entire model.
# NOTES:
#   - Can later be loaded with `keras.models.load_model('pretrained_model.h5')`
#     in another script for training or inference.
model.save('mask_detection_model.h5')
