"""
Example 2: Neural Network Training with MNIST - Underfitting vs Improved Models

INTENTION:
This script demonstrates the complete workflow of training neural networks for multi-class 
classification using the MNIST dataset. It shows two models: one that underfits (insufficient 
capacity) and one that performs well, illustrating the importance of proper architecture design.

Key Learning Objectives:
- Data preprocessing for neural networks (normalization, reshaping, one-hot encoding)
- Building neural network architectures with Keras Sequential API
- Understanding model capacity and its impact on performance
- Training models and interpreting metrics (loss, accuracy)
- Using dropout regularization to prevent overfitting
- Understanding why validation metrics can outperform training metrics with dropout

The script trains two models:
1. First Model: Underfitting example (1 neuron → 10 neurons) - demonstrates insufficient capacity
2. Second Model: Improved model (multiple hidden layers with dropout) - demonstrates proper architecture
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Print TensorFlow version to verify installation
print("TensorFlow version:", tf.__version__)

# Load MNIST dataset
mnist = tf.keras.datasets.mnist

# Load data splits
# x_train: (60000, 28, 28) - training images
# y_train: (60000,) - training labels (0-9)
# x_test: (10000, 28, 28) - test images
# y_test: (10000,) - test labels (0-9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("\nDataset loaded")
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)

# print("\nDisplaying first image")
# print("y_train (label):", y_train[0])
# plt.imshow(x_train[0])
# plt.show()

print("\nNormalizing data")

print("\nCurrent range of values")
print("max(x_train):", np.max(x_train))
print("min(x_train):", np.min(x_train))
print("max(x_test):", np.max(x_test))
print("min(x_test):", np.min(x_test))

# Normalize pixel values from [0, 255] to [0, 1]
# Why normalize:
# - Neural networks train better with normalized inputs
# - Prevents large pixel values from dominating gradients
# - Improves numerical stability and convergence speed
# - Standard practice for image data
x_train = x_train / 255.0
x_test = x_test / 255.0

print("\nNew range of values")
print("max(x_train):", np.max(x_train))
print("min(x_train):", np.min(x_train))
print("max(x_test):", np.max(x_test))
print("min(x_test):", np.min(x_test))

# Reshape data from 2D (28x28) to 1D (784) - flatten images
# Why reshape:
# - Dense (fully connected) layers require 1D input
# - Each image becomes a vector of 784 features (28 × 28 = 784)
# - Preserves all pixel information, just changes shape
print("\nReshaping data from 28x28 to 784")
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
x_train = x_train.reshape(60000, 28*28)  # (60000, 28, 28) → (60000, 784)
x_test = x_test.reshape(10000, 28*28)     # (10000, 28, 28) → (10000, 784)

print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)


# Convert labels to categorical (one-hot encoding)
# Why one-hot encoding:
# - Required for multi-class classification with softmax output
# - Converts integer labels [0, 1, 2, ...] to vectors [[1,0,0,...], [0,1,0,...], ...]
# - Example: label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# - Each class gets its own output neuron in the final layer
print("\nConverting labels to categorical")
print("y_train.shape:", y_train[:5])  # Before: [5 0 4 ...] (integers)
print("y_test.shape:", y_test[:5])
# Parameters:
# - num_classes=10: Number of classes (digits 0-9)
# - dtype='float32': Data type for the encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
print("y_train.shape:", y_train[:5])
print("y_test.shape:", y_test[:5])


# ======================== FIRST MODEL ========================
# Build the model
print("\n================================================")
print("\n[FIRST MODEL] Building the model")
print("================================================\n")
# Build the first model (underfitting example)
# Sequential API: Layers are added one after another in sequence
model = tf.keras.models.Sequential()

# First layer (input + hidden layer combined)
# Parameters:
#   - units=1: Only 1 neuron - severely limits model capacity
#   - activation='sigmoid': Sigmoid activation (outputs 0-1)
#     Note: Sigmoid is typically for binary classification, not ideal here
#   - input_shape=(784,): Specifies input dimension (784 features from flattened image)
#   - name='input': Optional name for the layer
# This layer has only 1 neuron, which is insufficient for learning 10 digit classes
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(784,), name='input'))

# Output layer
# Parameters:
#   - units=10: 10 neurons (one for each digit class 0-9)
#   - activation='softmax': Softmax converts outputs to probabilities that sum to 1
#     Each neuron outputs probability for one digit class
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Print the model summary
print("\nModel summary")
model.summary()
# Deep learning distinguishes from traditional machine learning  because deep learning uses multiple layers of neurons to learn features from the data.
# It eliminates the need for manual feature engineering.

# Model Compilation
# Compilation configures the model for training by specifying:
#   - Optimizer: Algorithm to update weights during training
#   - Loss function: How to measure prediction error
#   - Metrics: How to evaluate model performance
print("\nCompiling the model")

# Parameters:
#   - optimizer='SGD': Stochastic Gradient Descent
#     * Simple optimizer that updates weights based on gradient
#     * Good starting point, though Adam is often better
#   - loss='categorical_crossentropy': Loss function for multi-class classification
#     * Measures difference between predicted probabilities and true one-hot labels
#     * Lower is better (perfect = 0)
#     * Use 'binary_crossentropy' for binary classification (2 classes)
#   - metrics=['accuracy']: Metric to track during training
#     * Accuracy = (Correct Predictions) / (Total Predictions)
#     * Higher is better (perfect = 1.0 or 100%)
#     * Other metrics: precision, recall, f1_score (for imbalanced data)
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Training
# Training process: Forward pass → Calculate loss → Backward pass (backpropagation) → Update weights
print("\nTraining the model")

# Parameters:
#   - x_train, y_train: Training data and labels
#   - epochs=25: Number of complete passes through the training dataset
#     * Each epoch sees all 60,000 training images once
#     * More epochs = more learning, but risk of overfitting
#   - batch_size=128: Number of samples processed before updating weights
#     * Smaller batches: More frequent updates, more stable gradients, slower
#     * Larger batches: Fewer updates, faster, but may need more memory
#     * 128 is a good balance for most cases
#   - validation_split=0.2: Use 20% of training data for validation
#     * Validation set monitors generalization during training
#     * Helps detect overfitting (training accuracy >> validation accuracy)
model.fit(x_train, y_train, epochs=25, batch_size=128, validation_split=0.2)

# Output
# Epoch 19/25
# 375/375 [==============================] - 0s 449us/step - loss: 2.0259 - accuracy: 0.2166 - val_loss: 2.0176 - val_accuracy: 0.2126
# Epoch 20/25
# 375/375 [==============================] - 0s 517us/step - loss: 2.0172 - accuracy: 0.2166 - val_loss: 2.0092 - val_accuracy: 0.2129
# Epoch 21/25
# 375/375 [==============================] - 0s 454us/step - loss: 2.0089 - accuracy: 0.2184 - val_loss: 2.0011 - val_accuracy: 0.2155
# Epoch 22/25
# 375/375 [==============================] - 0s 451us/step - loss: 2.0008 - accuracy: 0.2206 - val_loss: 1.9933 - val_accuracy: 0.2182
# Epoch 23/25
# 375/375 [==============================] - 0s 455us/step - loss: 1.9930 - accuracy: 0.2229 - val_loss: 1.9858 - val_accuracy: 0.2199
# Epoch 24/25
# 375/375 [==============================] - 0s 449us/step - loss: 1.9854 - accuracy: 0.2246 - val_loss: 1.9784 - val_accuracy: 0.2223
# Epoch 25/25
# 375/375 [==============================] - 0s 450us/step - loss: 1.9780 - accuracy: 0.2269 - val_loss: 1.9712 - val_accuracy: 0.2232

# ======================== METRIC EXPLANATIONS ========================
# loss: Training loss (categorical crossentropy) - measures prediction error on training data
#       Lower is better. Final: 1.9780
# accuracy: Training accuracy - percentage of correct predictions on training data
#           Higher is better. Final: 0.2269 (22.69%)
# val_loss: Validation loss - measures prediction error on validation set (unseen data)
#          Lower is better. Final: 1.9712
# val_accuracy: Validation accuracy - percentage of correct predictions on validation set
#              Higher is better. Final: 0.2232 (22.32%)

# ======================== FIRST MODEL ANALYSIS ========================
# Performance: Poor (22% accuracy, barely better than random 10%)
# - Loss decreasing slowly but remains high (~2.0)
# - Training and validation metrics are similar (no overfitting, but severe underfitting)
# - Model cannot learn meaningful patterns due to insufficient capacity
# - Final accuracy: 22.69% training, 22.32% validation


# ======================== SECOND MODEL ========================
# Build the model
print("\n================================================")
print("\n[SECOND MODEL] Building the model")
print("================================================\n")
# Build the second model (improved architecture)
model = tf.keras.models.Sequential()

# First hidden layer
# Parameters:
#   - units=10: 10 neurons (more capacity than previous model)
#   - activation='relu': Rectified Linear Unit - most common activation for hidden layers
#     * ReLU: f(x) = max(0, x) - outputs 0 for negative inputs, x for positive inputs
#     * Helps prevent vanishing gradient problem
#   - input_shape=(784,): Input dimension
model.add(tf.keras.layers.Dense(units=10, activation='relu', input_shape=(784,), name='dense1'))

# Second hidden layer
model.add(tf.keras.layers.Dense(units=10, activation='relu'))

# Dropout layer - Regularization technique
# Parameters:
#   - rate=0.2: Randomly sets 20% of neurons to 0 during training
#     * Prevents overfitting by forcing model to not rely on specific neurons
#     * During inference, all neurons are active (dropout disabled)
#     * Common rates: 0.2-0.5 (20-50%)
model.add(tf.keras.layers.Dropout(0.2))

# Third hidden layer
model.add(tf.keras.layers.Dense(units=10, activation='relu'))

# Another dropout layer
model.add(tf.keras.layers.Dropout(0.2))

# Output layer
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Compile with same settings
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the improved model
# Parameters:
#   - validation_data=(x_test, y_test): Use test set for validation (for comparison)
#     * Note: In practice, use separate validation set, not test set
#   - epochs=100: More epochs to allow model to learn complex patterns
#   - Returns history object containing training metrics for each epoch
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=100)

# Output
# Epoch 91/100
# 469/469 [==============================] - 0s 563us/step - loss: 0.5375 - accuracy: 0.8389 - val_loss: 0.2940 - val_accuracy: 0.9274
# Epoch 92/100
# 469/469 [==============================] - 0s 566us/step - loss: 0.5339 - accuracy: 0.8382 - val_loss: 0.2919 - val_accuracy: 0.9270
# Epoch 93/100
# 469/469 [==============================] - 0s 593us/step - loss: 0.5424 - accuracy: 0.8356 - val_loss: 0.2970 - val_accuracy: 0.9267
# Epoch 94/100
# 469/469 [==============================] - 0s 618us/step - loss: 0.5370 - accuracy: 0.8373 - val_loss: 0.2888 - val_accuracy: 0.9303
# Epoch 95/100
# 469/469 [==============================] - 0s 565us/step - loss: 0.5365 - accuracy: 0.8379 - val_loss: 0.2927 - val_accuracy: 0.9267
# Epoch 96/100
# 469/469 [==============================] - 0s 564us/step - loss: 0.5301 - accuracy: 0.8404 - val_loss: 0.2938 - val_accuracy: 0.9260
# Epoch 97/100
# 469/469 [==============================] - 0s 579us/step - loss: 0.5361 - accuracy: 0.8385 - val_loss: 0.2905 - val_accuracy: 0.9277
# Epoch 98/100
# 469/469 [==============================] - 0s 570us/step - loss: 0.5326 - accuracy: 0.8401 - val_loss: 0.2941 - val_accuracy: 0.9263
# Epoch 99/100
# 469/469 [==============================] - 0s 565us/step - loss: 0.5354 - accuracy: 0.8382 - val_loss: 0.2908 - val_accuracy: 0.9286
# Epoch 100/100
# 469/469 [==============================] - 0s 577us/step - loss: 0.5265 - accuracy: 0.8394 - val_loss: 0.2900 - val_accuracy: 0.9285

# ======================== METRIC EXPLANATIONS ========================
# loss: Training loss (categorical crossentropy) - measures prediction error on training data
#       Lower is better. Final: 0.5265
# accuracy: Training accuracy - percentage of correct predictions on training data
#           Higher is better. Final: 0.8394 (83.94%)
# val_loss: Validation loss - measures prediction error on test set (unseen data)
#          Lower is better. Final: 0.2900
# val_accuracy: Validation accuracy - percentage of correct predictions on test set
#              Higher is better. Final: 0.9285 (92.85%)

# ======================== SECOND MODEL ANALYSIS ========================
# Performance: Good (92.85% validation accuracy, significant improvement)
# - Validation accuracy (92.85%) higher than training accuracy (83.94%) - indicates good generalization
# - Validation loss (0.29) much lower than training loss (0.53) - model generalizes well
# - Dropout regularization working effectively (prevents overfitting)
# - Model successfully learned digit patterns with multiple hidden layers
# - Final accuracy: 83.94% training, 92.85% validation


# Visualize training progress
# history.history contains metrics recorded during training
# Keys: 'loss', 'accuracy', 'val_loss', 'val_accuracy'

# Display first few loss values
print("\nFirst 5 values of loss")
print(history.history['loss'][:5])
print("\nFirst 5 values of val_loss")
print(history.history['val_loss'][:5])

# Plot loss curves
# Loss should decrease over time (lower is better)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy curves
# Accuracy should increase over time (higher is better)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ======================== WHY VALIDATION PERFORMS BETTER THAN TRAINING ========================
# Validation metrics (accuracy: 92.85%, loss: 0.29) are better than training metrics 
# (accuracy: 83.94%, loss: 0.53) because:
# 
# 1. Dropout Regularization Effect:
#    - During training: Dropout layers randomly deactivate 20% of neurons (rate=0.2)
#      - Model must learn robust features with fewer neurons active
#      - Training loss/accuracy computed with reduced network capacity
#    - During validation: Dropout is automatically disabled
#      - Full model capacity is used (all neurons active)
#      - Better predictions with complete network
#
# 2. This is actually GOOD:
#    - Indicates excellent generalization (model works well on unseen data)
#    - Dropout successfully prevents overfitting
#    - Model learned robust features that work better with full capacity
#
# 3. Normal behavior with dropout:
#    - Training metrics reflect "harder" learning conditions (with dropout)
#    - Validation metrics reflect "easier" prediction conditions (without dropout)
#    - Gap between them shows regularization is working effectively


# Model Evaluation
# Evaluate model performance on test set
# Returns: [loss, accuracy] (or other metrics specified during compilation)
print("\nEvaluating the model")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Model Prediction
# Generate predictions for test images
# Returns: Array of probability distributions (one per image)
# Each prediction is array of 10 probabilities (one for each digit class)
print("\nMaking predictions on test set")
predictions = model.predict(x_test, verbose=0)
print(f"Predictions shape: {predictions.shape}")  # (10000, 10)
print(f"First prediction (probabilities): {predictions[0]}")
print(f"Predicted class for first image: {np.argmax(predictions[0])}")  # Class with highest probability

# Model Configuration
# Get model architecture configuration (layers, parameters)
print("\nModel configuration")
config = model.get_config()
print(f"Number of layers: {len(config['layers'])}")

# Visualize Model Architecture
# Creates a diagram showing the model structure
# Parameters:
#   - to_file='model.png': Save diagram to file
#   - show_shapes=True: Display input/output shapes for each layer
print("\nPlotting the model architecture")
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
print("Model architecture saved to 'model.png'")
