"""
Transfer Learning with VGG16 for Cats vs Dogs Classification

INTENTION:
This script demonstrates transfer learning using VGG16, a pre-trained CNN model. Instead of
training from scratch, we leverage a model trained on ImageNet (1.4M images, 1000 classes) and
adapt it for our binary classification task (cats vs dogs).

Key Learning Objectives:
- Understanding transfer learning concept and benefits
- Loading and using pre-trained models (VGG16)
- Freezing/unfreezing layers for fine-tuning
- Building custom classification heads on pre-trained bases
- Comparing transfer learning vs training from scratch

What is Transfer Learning?
Transfer learning uses a model trained on one task (ImageNet classification) and adapts it
to a new, related task (cats vs dogs binary classification). We keep the learned features
from ImageNet and only train new layers for our specific task.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ======================== COMPARISON WITH CNN FROM SCRATCH ========================
# Load the previously trained CNN model (from cats-and-dogs.py) for performance comparison
# This allows us to see if transfer learning performs better than training from scratch
ROOT_DIR = "./data_dir"
print("Loading previously trained CNN model...")
try:
    cnn_model = load_model("model.h5")
    print("CNN model loaded successfully!")
    
    # Evaluate CNN model on test data for comparison
    testdatagenerator = ImageDataGenerator(rescale=1./255)
    test_data = testdatagenerator.flow_from_directory(
        ROOT_DIR + "/test", 
        target_size=(224, 224), 
        class_mode="binary"
    )
    print("\nEvaluating CNN model on test data:")
    cnn_model.evaluate(test_data, verbose=1)
except FileNotFoundError:
    print("CNN model (model.h5) not found. Train the CNN model first in cats-and-dogs.py")
    cnn_model = None

# ======================== DATA GENERATORS FOR TRANSFER LEARNING ========================
# Set up data generators (same structure as CNN from scratch)
# Note: VGG16 expects 224×224 images (same as our CNN), so preprocessing is identical
print("\n" + "="*50)
print("Setting up data generators for VGG16 transfer learning")
print("="*50)

# Training data generator with augmentation
# Same augmentation as CNN from scratch to ensure fair comparison
trainingdatagenerator = ImageDataGenerator(
    rescale=1./255,              # Normalize to [0,1]
    rotation_range=0.3,          # Random rotation
    zoom_range=0.2,             # Random zoom
    horizontal_flip=True,        # Horizontal flip
    vertical_flip=True           # Vertical flip (use carefully)
)
train_data = trainingdatagenerator.flow_from_directory(
    ROOT_DIR + "/train", 
    target_size=(224, 224),      # VGG16 standard input size
    class_mode="binary",         # Binary classification
    batch_size=32               # Batch size for training
)

# Validation data generator (no augmentation)
validationdatagenerator = ImageDataGenerator(rescale=1./255)
val_data = validationdatagenerator.flow_from_directory(
    ROOT_DIR + "/val", 
    target_size=(224, 224), 
    class_mode="binary",
    batch_size=32
)

# Test data generator (no augmentation)
testdatagenerator = ImageDataGenerator(rescale=1./255)
test_data = testdatagenerator.flow_from_directory(
    ROOT_DIR + "/test", 
    target_size=(224, 224), 
    class_mode="binary",
    batch_size=32
)

print("Class indices: ", train_data.class_indices)
class_id = {j: i for i, j in train_data.class_indices.items()}
print("Class ID: ", class_id)

# ======================== LOAD PRE-TRAINED VGG16 BASE MODEL ========================
# VGG16: Pre-trained CNN model trained on ImageNet (1.4M images, 1000 classes)
# VGG16 Architecture: 13 convolutional layers + 3 fully connected layers
# Pre-trained weights contain learned features (edges, textures, objects) useful for any image task
print("\n" + "="*50)
print("Building VGG16 Transfer Learning Model")
print("="*50)

# Load VGG16 base model without top classification layer
# Parameters:
#   - input_shape=(224, 224, 3): Input image dimensions (height, width, RGB channels)
#     * Must match VGG16's training input size
#   - include_top=False: Exclude final classification layers (1000-class output)
#     * We want to add our own classification head for binary classification
#   - weights="imagenet": Load pre-trained weights from ImageNet
#     * These weights contain learned features we want to reuse
#     * Alternative: weights=None (random initialization, defeats purpose of transfer learning)
# Output: Feature maps (e.g., 7×7×512) instead of class probabilities
vgg_base = VGG16(
    input_shape=(224, 224, 3), 
    include_top=False,  # Don't include the final classification layer
    weights="imagenet"  # Use pre-trained ImageNet weights
)

print("\nVGG16 base model summary:")
vgg_base.summary()

# ======================== LAYER FREEZING STRATEGY ========================
# Freezing: Prevent weights from being updated during training
# Unfreezing: Allow weights to be updated during training

# Freeze early layers (feature extraction layers)
# Why freeze: Early layers learn general features (edges, textures) that work for any image task
# These features are already well-learned from ImageNet, no need to retrain
print("\nFreezing first 15 layers (feature extraction)...")
for layer in vgg_base.layers[:15]:
    layer.trainable = False  # Weights won't be updated

# Unfreeze later layers (fine-tuning)
# Why unfreeze: Later layers learn task-specific features that may need adaptation
# Fine-tuning these layers helps adapt to our specific task (cats vs dogs)
print("Unfreezing layers 15 onwards for fine-tuning...")
for layer in vgg_base.layers[15:]:
    layer.trainable = True  # Weights will be updated during training

# Strategy options:
#   - Feature extraction only: Freeze all base layers, only train new head (faster, less flexible)
#   - Partial fine-tuning: Unfreeze last 1-2 blocks (balanced approach)
#   - Full fine-tuning: Unfreeze all layers (requires more data, lower learning rate)

# ======================== BUILD CUSTOM CLASSIFICATION HEAD ========================
# Get the output feature maps from VGG16 base model
# Shape: (7, 7, 512) - spatial feature maps from last convolutional layer
vgg_output = vgg_base.output

# Add custom classification head for binary classification
# Option 1: GlobalMaxPooling2D (preferred for transfer learning)
# What: Takes maximum value across spatial dimensions (7×7 → 1 value per filter)
# Why: More parameter-efficient, reduces overfitting
# Output shape: (512,) - one value per filter
x = GlobalMaxPooling2D()(vgg_output)

# Option 2: Flatten (alternative, but uses more parameters)
# What: Converts 2D feature maps to 1D vector
# Output shape: (25,088,) - flattened vector (7 × 7 × 512)
# Why alternative: More parameters, may overfit with small datasets
# x = Flatten()(vgg_output)

# Dense layer for classification
# Parameters:
#   - units=512: Number of neurons (good capacity without overfitting)
#   - activation="relu": ReLU activation for hidden layer
x = Dense(512, activation="relu")(x)

# Dropout for regularization
# Parameters:
#   - rate=0.3: Randomly deactivate 30% of neurons during training
x = Dropout(0.3)(x)

# Output layer for binary classification
# Parameters:
#   - units=1: Single output neuron (probability of dog class)
#   - activation="sigmoid": Outputs probability [0, 1]
x = Dense(1, activation="sigmoid")(x)

# Create the complete transfer learning model
# Combines VGG16 base (input) with custom classification head (output)
# Functional API: Model(inputs=..., outputs=...) - more flexible than Sequential
transfer_model = Model(inputs=vgg_base.input, outputs=x)

print("\nTransfer Learning Model Summary:")
transfer_model.summary()

# ======================== MODEL COMPILATION ========================
# Compile with appropriate optimizer and learning rate for transfer learning
# CRITICAL: Use lower learning rate than training from scratch
# Why: Pre-trained weights are already good, need small adjustments, not large changes
transfer_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9), 
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)
# Parameters:
#   - optimizer: SGD with momentum (better for fine-tuning than Adam)
#   - learning_rate=0.0001: Very low learning rate (10x lower than typical 0.001)
#     * Prevents destroying pre-trained features
#     * Allows fine adjustments to adapt to new task
#   - momentum=0.9: Adds inertia to gradient updates (standard value)
#   - loss: Binary crossentropy (same as CNN from scratch)
#   - metrics: Accuracy (same as CNN from scratch)

# ======================== TRAINING CALLBACKS ========================
# Same callbacks as CNN from scratch: EarlyStopping and ModelCheckpoint
early_stopping = EarlyStopping(
    monitor="val_accuracy", 
    min_delta=0.01, 
    patience=5, 
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath="best_model.h5",  # Save as best_model.h5 (different from CNN model.h5)
    monitor="val_accuracy", 
    save_best_only=True, 
    verbose=1
)

callbacks = [early_stopping, model_checkpoint]

# ======================== TRAIN TRANSFER LEARNING MODEL ========================
# Train the model (typically needs fewer epochs than training from scratch)
print("\n" + "="*50)
print("Training Transfer Learning Model")
print("="*50)

history = transfer_model.fit(
    train_data, 
    epochs=10,              # Fewer epochs needed (pre-trained weights provide good starting point)
    validation_data=val_data, 
    callbacks=callbacks, 
    steps_per_epoch=128,    # Number of batches per epoch
    validation_steps=16,    # Number of validation batches
    verbose=1
)
# Note: Transfer learning typically converges faster than training from scratch
# EarlyStopping may stop training even earlier if no improvement

# Load the best model
print("\n" + "="*50)
print("Evaluating Best Transfer Learning Model")
print("="*50)
best_model = load_model("best_model.h5")
print("\nEvaluating on test data:")
best_model.evaluate(test_data, verbose=1)

# ======================== MAKE PREDICTIONS ON TEST IMAGE ========================
# Example: Predict on a single test image (same process as predict.py)
print("\n" + "="*50)
print("Making Prediction on Test Image")
print("="*50)

# Example: Predict on a test image
test_image_path = "test.jpg"  # Change this to your test image path

if os.path.exists(test_image_path):
    # Load and preprocess image (must match training preprocessing)
    # Step 1: Load and resize
    pil = load_img(test_image_path, target_size=(224, 224))
    # Step 2: Convert to array
    image = img_to_array(pil)
    # Step 3: Normalize to [0, 1]
    image = image / 255.0
    # Step 4: Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    pred = best_model.predict(image, verbose=0)
    # Extract probability and class
    pred_class = int(pred[0][0] > 0.5)  # Binary classification threshold (0.5)
    pred_prob = pred[0][0]
    
    # Get class name from mapping
    predicted_class = class_id[pred_class]
    
    print(f"Prediction probability: {pred_prob:.4f}")
    print(f"Predicted class: {predicted_class}")
    
    # Display image with prediction
    plt.imshow(image[0])
    plt.title(f"Predicted: {predicted_class} (Probability: {pred_prob:.4f})")
    plt.axis('off')
    plt.show()
else:
    print(f"Test image '{test_image_path}' not found.")
    print("To test prediction, provide a test image path.")

print("\nTransfer learning complete!")

# ======================== ESSENTIAL CONCEPTS SUMMARY ========================
"""
ESSENTIAL CONCEPTS IN THIS SCRIPT:

1. TRANSFER LEARNING:
   - What: Reuse a model trained on one task (ImageNet) for a new task (cats vs dogs)
   - Why: Faster training, better performance with less data, proven architectures
   - How: Keep pre-trained base, replace/add custom classification head

2. PRE-TRAINED MODELS (VGG16):
   - VGG16: 16-layer CNN trained on ImageNet (1.4M images, 1000 classes)
   - Learned features: Edges, textures, patterns, objects (useful for any image task)
   - Why VGG16: Well-established, good balance of depth and performance

3. LAYER FREEZING:
   - Freezing (trainable=False): Prevents weight updates (preserves learned features)
   - Unfreezing (trainable=True): Allows weight updates (adapts to new task)
   - Strategy: Freeze early layers (general features), unfreeze later layers (task-specific)

4. CUSTOM CLASSIFICATION HEAD:
   - What: New layers added on top of pre-trained base
   - Components: GlobalMaxPooling2D → Dense → Dropout → Dense (output)
   - Why: Pre-trained model outputs 1000 classes, we need binary classification

5. GLOBALMAXPOOLING2D VS FLATTEN:
   - GlobalMaxPooling2D: Takes max across spatial dimensions (more efficient)
   - Flatten: Converts 2D to 1D vector (more parameters)
   - Recommendation: Use GlobalMaxPooling2D for transfer learning

6. LEARNING RATE FOR TRANSFER LEARNING:
   - Must be LOWER than training from scratch (typically 0.0001 vs 0.001)
   - Why: Pre-trained weights are already good, need fine adjustments
   - Too high: Destroys pre-trained features, poor performance
   - Too low: Very slow convergence

7. OPTIMIZER CHOICE:
   - SGD with momentum: Preferred for fine-tuning (more stable)
   - Adam: Can work but SGD often better for transfer learning
   - Why: Pre-trained weights need careful, controlled updates

8. TRAINING EFFICIENCY:
   - Fewer epochs needed: Pre-trained weights provide good starting point
   - Faster convergence: Model already knows general image features
   - Less data needed: Can achieve good performance with smaller datasets

9. FUNCTIONAL API VS SEQUENTIAL:
   - Sequential: Simple, linear stack of layers
   - Functional API: More flexible, allows complex architectures
   - Transfer learning uses Functional API: Model(inputs=base.input, outputs=custom_head)

10. COMPARISON WITH TRAINING FROM SCRATCH:
    - Transfer Learning: Faster, better with limited data, uses proven architecture
    - From Scratch: More control, but needs more data and time
    - When to use: Transfer learning for most image tasks, from scratch for unique domains

KEY ADVANTAGES OF TRANSFER LEARNING:
- Leverages millions of images (ImageNet) without training on them
- Faster training (hours vs days/weeks)
- Better performance with limited data
- Proven architectures (VGG16, ResNet, etc.)
- Less computational resources needed
"""
