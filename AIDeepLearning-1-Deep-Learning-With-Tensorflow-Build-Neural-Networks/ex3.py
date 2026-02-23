"""
Example 3: Three Types of Neural Network Models - Regression, Binary, and Multi-Class Classification

INTENTION:
This script demonstrates how to build and train three different types of neural network models
for different problem types using the same MNIST dataset. It shows the architectural and
compilation differences required for each problem type.

Key Learning Objectives:
- Understand differences between regression, binary classification, and multi-class classification
- Learn appropriate architectures for each problem type:
  * Regression: Continuous output (1 neuron, no activation)
  * Binary Classification: Binary output (1 neuron, sigmoid activation)
  * Multi-Class Classification: Categorical output (N neurons, softmax activation)
- Learn appropriate loss functions and metrics for each problem type
- Compare model performance across different problem types

The script trains three models:
1. Regression Model: Predicts continuous values (normalized pixel sum)
2. Binary Classification Model: Predicts binary outcomes (even/odd digits)
3. Multi-Class Classification Model: Predicts digit class (0-9)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Print TensorFlow version to verify installation
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

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

# Instead of dealing with 0 to 255, we can normalize to 0 to 1
x_train = x_train / 255.0
x_test = x_test / 255.0

print("\nNew range of values")
print("max(x_train):", np.max(x_train))
print("min(x_train):", np.min(x_train))
print("max(x_test):", np.max(x_test))
print("min(x_test):", np.min(x_test))

# Reshape data from 28x28 to 784 (from matrix to vector)
print("\nReshaping data from 28x28 to 784")
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)


# Convert labels to categorical
# This is because the model expects the labels to be in a categorical format
# A categorical format is a one-hot encoded vector of 0s and 1s

print("\nConverting labels to categorical")
print("y_train.shape:", y_train[:5])
print("y_test.shape:", y_test[:5])
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
print("y_train.shape:", y_train[:5])
print("y_test.shape:", y_test[:5])



# ======================== COMMON NEURAL NETWORK - REGRESSION ========================
# Build the model
# Input layer: Number of neurons is the number of features in the dataset.
#    - X_train.shape = (60000, 784), so 784 neurons in the input layer.
#    - 784 is the result of 28x28 pixels (image size) in a flattened format.
# Hidden layers: Thumb rule is 3 to 20 neurons in the hidden layer.
# Number of neurons in the hidden layer - 10 to 30 will be used
# Activation function: ReLU is a good choice for hidden layers. Because it is a good choice for regression tasks.
# Output layer: 1 neuron without an activation function. Because it is a regression task.
# Loss function: Mean squared error is a good choice for regression tasks.
# ======================== COMMON NEURAL NETWORK - Binary Classification ========================
# Build the model
# Input layer: Number of neurons is the number of features in the dataset.
#    - X_train.shape = (60000, 784), so 784 neurons in the input layer.
#    - 784 is the result of 28x28 pixels (image size) in a flattened format.
# Hidden layers: Thumb rule is 3 to 20 neurons in the hidden layer.
# Number of neurons in the hidden layer - 10 to 30 will be used
# Activation function: Sigmoid is a good choice for binary classification tasks.
# Output layer: 1 neuron with Sigmoid activation function. Because it is a binary classification task.
# Loss function: Binary crossentropy is a good choice for binary classification tasks.
# ======================== COMMON NEURAL NETWORK - Multi-Class Classification ========================
# Build the model
# Input layer: Number of neurons is the number of features in the dataset.
#    - X_train.shape = (60000, 784), so 784 neurons in the input layer.
#    - 784 is the result of 28x28 pixels (image size) in a flattened format.
# Hidden layers: Thumb rule is 3 to 20 neurons in the hidden layer.
# Number of neurons in the hidden layer - 10 to 30 will be used
# Activation function: Softmax is a good choice for multi-class classification tasks.
# Output layer: Number of neurons is the number of classes in the dataset. 
# Loss function: Categorical crossentropy is a good choice for multi-class classification tasks.

# ======================== Training hyperparameters ========================
# Batch size: We use the batch which can be fit on my memory
# Number of epochs: Maximum but control with tensorflow callbacks
# Optimizer: Adam is a good choice for most tasks.
# Loss function: Depends on the task.
# Metrics: We use accuracy but we can use other metrics depending on the task.




# ======================== REGRESSION MODEL ========================
print("\n================================================")
print("\n[REGRESSION MODEL] Building the model")
print("================================================\n")

# REGRESSION: Predict continuous values (e.g., pixel intensity sum, house prices, temperature)
# Convert labels to regression target (sum of pixel values as example)
# axis=1: Sum across columns (pixels) for each image
# Divide by 784 to normalize (max possible sum is 784 if all pixels = 1)
y_train_reg = np.sum(x_train, axis=1) / 784.0  # Normalized sum [0, 1]
y_test_reg = np.sum(x_test, axis=1) / 784.0

# Build regression model
model_reg = tf.keras.models.Sequential()

# Hidden layer 1
# Parameters:
#   - units=30: 30 neurons in first hidden layer
#   - activation='relu': ReLU activation (standard for hidden layers)
#   - input_shape=(784,): Input dimension
model_reg.add(tf.keras.layers.Dense(units=30, activation='relu', input_shape=(784,)))

# Hidden layer 2
model_reg.add(tf.keras.layers.Dense(units=20, activation='relu'))

# Output layer for regression
# Parameters:
#   - units=1: Single output neuron (predicts one continuous value)
#   - No activation: Linear output (can output any value, not restricted to [0,1])
#     * Regression outputs continuous values, not probabilities
model_reg.add(tf.keras.layers.Dense(units=1))

# Compile regression model
# Parameters:
#   - optimizer='adam': Adaptive learning rate optimizer (better than SGD for most cases)
#   - loss='mse': Mean Squared Error - standard loss for regression
#     * MSE = average of (predicted - actual)²
#     * Penalizes large errors more than small errors
#   - metrics=['mae']: Mean Absolute Error - easier to interpret than MSE
#     * MAE = average of |predicted - actual|
#     * Tells average error magnitude (e.g., MAE=0.1 means average error is 0.1)
model_reg.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train regression model
# Parameters:
#   - verbose=0: Suppress training output (set to 1 to see progress)
print("Training regression model...")
history_reg = model_reg.fit(x_train, y_train_reg, validation_data=(x_test, y_test_reg), 
                            batch_size=128, epochs=50, verbose=0)
print(f"Final training loss: {history_reg.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history_reg.history['val_loss'][-1]:.4f}")

# ======================== BINARY CLASSIFICATION MODEL ========================
print("\n================================================")
print("\n[BINARY CLASSIFICATION MODEL] Building the model")
print("================================================\n")

# BINARY CLASSIFICATION: Predict binary outcomes (e.g., even/odd, spam/not spam, yes/no)
# Convert digit labels to binary: even digits (0,2,4,6,8) → 0, odd digits (1,3,5,7,9) → 1
# argmax(axis=1): Get original digit class from one-hot encoded labels
# % 2: Modulo 2 to get even (0) or odd (1)
y_train_binary = (y_train.argmax(axis=1) % 2).astype('float32')
y_test_binary = (y_test.argmax(axis=1) % 2).astype('float32')

# Build binary classification model
model_binary = tf.keras.models.Sequential()

# Hidden layers (same as regression)
model_binary.add(tf.keras.layers.Dense(units=30, activation='relu', input_shape=(784,)))
model_binary.add(tf.keras.layers.Dense(units=20, activation='relu'))

# Output layer for binary classification
# Parameters:
#   - units=1: Single output neuron (predicts probability of class 1)
#   - activation='sigmoid': Sigmoid outputs probability [0, 1]
#     * Output < 0.5 → predict class 0
#     * Output ≥ 0.5 → predict class 1
#     * Sigmoid: f(x) = 1 / (1 + e^(-x))
model_binary.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile binary classification model
# Parameters:
#   - loss='binary_crossentropy': Loss function for binary classification
#     * Measures difference between predicted probability and true binary label
#     * Works with sigmoid output
#   - metrics=['accuracy']: Percentage of correct predictions
model_binary.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training binary classification model...")
history_binary = model_binary.fit(x_train, y_train_binary, validation_data=(x_test, y_test_binary), 
                                  batch_size=128, epochs=50, verbose=0)
print(f"Final training accuracy: {history_binary.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history_binary.history['val_accuracy'][-1]:.4f}")

# ======================== MULTI-CLASS CLASSIFICATION MODEL ========================
print("\n================================================")
print("\n[MULTI-CLASS CLASSIFICATION MODEL] Building the model")
print("================================================\n")

# MULTI-CLASS CLASSIFICATION: Predict one of multiple classes (e.g., digit 0-9, image categories)
# Using original categorical labels (already one-hot encoded from earlier preprocessing)

# Build multi-class classification model
model_multiclass = tf.keras.models.Sequential()

# Hidden layers
model_multiclass.add(tf.keras.layers.Dense(units=30, activation='relu', input_shape=(784,)))
model_multiclass.add(tf.keras.layers.Dense(units=20, activation='relu'))

# Dropout for regularization (prevents overfitting)
# Parameters:
#   - rate=0.2: Randomly deactivate 20% of neurons during training
model_multiclass.add(tf.keras.layers.Dropout(0.2))

# Output layer for multi-class classification
# Parameters:
#   - units=10: 10 neurons (one for each digit class 0-9)
#   - activation='softmax': Softmax converts outputs to probability distribution
#     * Each neuron outputs probability for one class
#     * All probabilities sum to 1.0
#     * Class with highest probability is the prediction
#     * Softmax: f(x_i) = e^(x_i) / Σ(e^(x_j)) for all j
model_multiclass.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Compile multi-class classification model
# Parameters:
#   - loss='categorical_crossentropy': Loss for multi-class with one-hot encoded labels
#     * Measures difference between predicted probability distribution and true one-hot label
#     * Works with softmax output
#     * Use 'sparse_categorical_crossentropy' if labels are integers (not one-hot)
#   - metrics=['accuracy']: Percentage of correct class predictions
model_multiclass.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training multi-class classification model...")
history_multiclass = model_multiclass.fit(x_train, y_train, validation_data=(x_test, y_test), 
                                          batch_size=128, epochs=50, verbose=0)
print(f"Final training accuracy: {history_multiclass.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history_multiclass.history['val_accuracy'][-1]:.4f}")

# ======================== MODEL COMPARISON ========================
print("\n================================================")
print("\nMODEL COMPARISON SUMMARY")
print("================================================\n")
print("Regression Model:")
print(f"  - Loss (MSE): {history_reg.history['val_loss'][-1]:.4f}")
print(f"  - MAE: {history_reg.history['val_mae'][-1]:.4f}")
print("\nBinary Classification Model:")
print(f"  - Accuracy: {history_binary.history['val_accuracy'][-1]:.4f}")
print(f"  - Loss: {history_binary.history['val_loss'][-1]:.4f}")
print("\nMulti-Class Classification Model:")
print(f"  - Accuracy: {history_multiclass.history['val_accuracy'][-1]:.4f}")
print(f"  - Loss: {history_multiclass.history['val_loss'][-1]:.4f}")

# ======================== PLOTTING RESULTS ========================
# Visualize training progress for all three models side-by-side
plt.figure(figsize=(15, 4))

# Plot 1: Regression Model - Loss (MSE)
# For regression, we plot loss (MSE) - lower is better
plt.subplot(1, 3, 1)
plt.plot(history_reg.history['loss'], label='Training Loss (MSE)')
plt.plot(history_reg.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Regression Model - Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot 2: Binary Classification - Accuracy
# For classification, we plot accuracy - higher is better
plt.subplot(1, 3, 2)
plt.plot(history_binary.history['accuracy'], label='Training Accuracy')
plt.plot(history_binary.history['val_accuracy'], label='Validation Accuracy')
plt.title('Binary Classification - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot 3: Multi-Class Classification - Accuracy
plt.subplot(1, 3, 3)
plt.plot(history_multiclass.history['accuracy'], label='Training Accuracy')
plt.plot(history_multiclass.history['val_accuracy'], label='Validation Accuracy')
plt.title('Multi-Class Classification - Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ======================== KEY DIFFERENCES BETWEEN MODEL TYPES ========================
# Regression:
#   - Output: Continuous values (1 neuron, no activation)
#   - Loss: MSE (Mean Squared Error) or MAE (Mean Absolute Error)
#   - Metrics: MAE, MSE
#   - Example: Predicting house prices, temperature, pixel intensity
#
# Binary Classification:
#   - Output: Probability (1 neuron, sigmoid activation)
#   - Loss: Binary crossentropy
#   - Metrics: Accuracy
#   - Example: Spam detection, medical diagnosis (yes/no)
#
# Multi-Class Classification:
#   - Output: Probability distribution (N neurons, softmax activation)
#   - Loss: Categorical crossentropy
#   - Metrics: Accuracy
#   - Example: Digit recognition (0-9), image classification

# ======================== EVALUATION ========================
# Evaluate all models on test set to get final performance metrics
# evaluate() returns list of metrics in order: [loss, metric1, metric2, ...]
print("\nEvaluating all models on test set...")

# Regression Model Evaluation
# Returns: [loss (MSE), MAE]
print("\nRegression Model:")
reg_results = model_reg.evaluate(x_test, y_test_reg, verbose=0)
print(f"  Loss (MSE): {reg_results[0]:.6f}, MAE: {reg_results[1]:.6f}")

# Binary Classification Model Evaluation
# Returns: [loss (binary_crossentropy), accuracy]
print("\nBinary Classification Model:")
binary_results = model_binary.evaluate(x_test, y_test_binary, verbose=0)
print(f"  Loss: {binary_results[0]:.6f}, Accuracy: {binary_results[1]:.6f}")

# Multi-Class Classification Model Evaluation
# Returns: [loss (categorical_crossentropy), accuracy]
print("\nMulti-Class Classification Model:")
multiclass_results = model_multiclass.evaluate(x_test, y_test, verbose=0)
print(f"  Loss: {multiclass_results[0]:.6f}, Accuracy: {multiclass_results[1]:.6f}")

# ======================== PERFORMANCE INTERPRETATION ========================
# Regression Model (MAE: 0.000131):
#   - MAE measures average absolute error between predictions and actual values
#   - Lower is better (closer to 0)
#   - Since target values are normalized [0, 1], MAE of 0.000131 is excellent
#   - Means predictions are off by only 0.0131% on average
#   - MAE close to 1 would be very poor (100% error)
#
# Binary Classification (Accuracy: 98.54%):
#   - Higher accuracy is better (closer to 1.0 or 100%)
#   - 98.54% means model correctly predicts even/odd 98.54% of the time
#
# Multi-Class Classification (Accuracy: 96.45%):
#   - Higher accuracy is better (closer to 1.0 or 100%)
#   - 96.45% means model correctly identifies digit class 96.45% of the time
#   - Very good performance for 10-class classification

# ======================== UNDERSTANDING SEQUENTIAL MODELS AND MODEL DIFFERENCES ========================
"""
WHY ALL MODELS START WITH: model = tf.keras.models.Sequential()

1. What is Sequential?
   - Sequential is a Keras model type that builds neural networks layer-by-layer
   - It creates a linear stack of layers where data flows from input → layer1 → layer2 → ... → output
   - Each layer receives input from the previous layer and passes output to the next layer
   - Think of it as a pipeline: data flows in one direction (forward pass)

2. Why Use Sequential?
   - Simple and intuitive: Perfect for most standard neural networks
   - Easy to build: Just add layers one after another with model.add()
   - Clear structure: Easy to understand the data flow
   - Suitable for: Feedforward networks, CNNs, simple architectures
   - Not suitable for: Models with multiple inputs/outputs, shared layers, complex branching

3. How Sequential Works:
   - Step 1: Create empty Sequential container
     model = tf.keras.models.Sequential()
     * This creates an empty model with no layers yet
   
   - Step 2: Add layers sequentially
     model.add(layer1)  # First layer (input layer)
     model.add(layer2) # Second layer (hidden layer)
     model.add(layer3) # Third layer (output layer)
     * Layers are added in order: input → hidden → output
     * Each layer automatically knows its input shape from the previous layer
   
   - Step 3: Compile the model
     model.compile(optimizer=..., loss=..., metrics=...)
     * Configures how the model will be trained
     * Defines the optimization algorithm, loss function, and metrics
   
   - Step 4: Train the model
     model.fit(x_train, y_train, ...)
     * Trains the model on data
     * Updates weights through backpropagation

4. Alternative: Functional API
   - Sequential is one way to build models
   - Functional API allows more complex architectures:
     inputs = Input(shape=(784,))
     x = Dense(30, activation='relu')(inputs)
     outputs = Dense(1)(x)
     model = Model(inputs=inputs, outputs=outputs)
   - For these simple examples, Sequential is perfect and easier to use


KEY DIFFERENCES BETWEEN THE THREE MODELS:

All three models use Sequential() and have the SAME structure for hidden layers:
  - Hidden Layer 1: 30 neurons, ReLU activation
  - Hidden Layer 2: 20 neurons, ReLU activation

The differences occur in THREE critical areas:

─────────────────────────────────────────────────────────────────────────────
1. OUTPUT LAYER (The Final Layer)
─────────────────────────────────────────────────────────────────────────────

REGRESSION MODEL:
  model_reg.add(tf.keras.layers.Dense(units=1))
  - units=1: Single output neuron (predicts one continuous value)
  - NO activation: Linear output (can output any real number)
  - Why: Regression needs to predict continuous values (e.g., 0.523, 1.234, -0.456)
  - Output range: (-∞, +∞) - any real number

BINARY CLASSIFICATION MODEL:
  model_binary.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
  - units=1: Single output neuron (predicts probability of class 1)
  - activation='sigmoid': Outputs probability between 0 and 1
  - Why: Binary classification needs probability (e.g., 0.7 = 70% chance of class 1)
  - Output range: [0, 1] - probability
  - Decision: If output ≥ 0.5 → predict class 1, else predict class 0

MULTI-CLASS CLASSIFICATION MODEL:
  model_multiclass.add(tf.keras.layers.Dense(units=10, activation='softmax'))
  - units=10: 10 output neurons (one for each digit class 0-9)
  - activation='softmax': Converts outputs to probability distribution
  - Why: Multi-class needs probabilities for all classes that sum to 1.0
  - Output: Array of 10 probabilities, e.g., [0.01, 0.05, 0.80, 0.10, ...]
  - Decision: Class with highest probability is the prediction

─────────────────────────────────────────────────────────────────────────────
2. LOSS FUNCTION (In model.compile())
─────────────────────────────────────────────────────────────────────────────

REGRESSION MODEL:
  model_reg.compile(..., loss='mse', ...)
  - loss='mse': Mean Squared Error
  - Formula: MSE = (1/n) * Σ(predicted - actual)²
  - Why: Measures average squared difference between predictions and actual values
  - Penalizes large errors more than small errors
  - Lower is better (perfect = 0)

BINARY CLASSIFICATION MODEL:
  model_binary.compile(..., loss='binary_crossentropy', ...)
  - loss='binary_crossentropy': Binary cross-entropy loss
  - Formula: L = -(y*log(ŷ) + (1-y)*log(1-ŷ))
  - Why: Measures difference between predicted probability and true binary label
  - Works with sigmoid output (probabilities)
  - Lower is better (perfect = 0)

MULTI-CLASS CLASSIFICATION MODEL:
  model_multiclass.compile(..., loss='categorical_crossentropy', ...)
  - loss='categorical_crossentropy': Categorical cross-entropy loss
  - Formula: L = -Σ(y_i * log(ŷ_i)) for all classes i
  - Why: Measures difference between predicted probability distribution and true one-hot label
  - Works with softmax output (probability distributions)
  - Lower is better (perfect = 0)

─────────────────────────────────────────────────────────────────────────────
3. METRICS (In model.compile())
─────────────────────────────────────────────────────────────────────────────

REGRESSION MODEL:
  model_reg.compile(..., metrics=['mae'])
  - metrics=['mae']: Mean Absolute Error
  - Formula: MAE = (1/n) * Σ|predicted - actual|
  - Why: Easier to interpret than MSE (tells average error magnitude)
  - Example: MAE=0.1 means predictions are off by 0.1 on average
  - Lower is better

BINARY CLASSIFICATION MODEL:
  model_binary.compile(..., metrics=['accuracy'])
  - metrics=['accuracy']: Percentage of correct predictions
  - Formula: Accuracy = (Correct Predictions) / (Total Predictions)
  - Why: Simple and intuitive metric for classification
  - Example: 0.98 = 98% accuracy (98 out of 100 predictions correct)
  - Higher is better (perfect = 1.0)

MULTI-CLASS CLASSIFICATION MODEL:
  model_multiclass.compile(..., metrics=['accuracy'])
  - metrics=['accuracy']: Percentage of correct class predictions
  - Same formula as binary classification
  - Why: Standard metric for multi-class problems
  - Higher is better (perfect = 1.0)

─────────────────────────────────────────────────────────────────────────────
4. TARGET DATA (y_train/y_test)
─────────────────────────────────────────────────────────────────────────────

REGRESSION MODEL:
  y_train_reg = np.sum(x_train, axis=1) / 784.0
  - Format: Continuous values (floats)
  - Example: [0.523, 0.789, 0.234, ...]
  - Range: [0, 1] (normalized)
  - Why: Regression predicts continuous values, not classes

BINARY CLASSIFICATION MODEL:
  y_train_binary = (y_train.argmax(axis=1) % 2).astype('float32')
  - Format: Binary labels (0 or 1)
  - Example: [0, 1, 0, 1, 1, 0, ...]
  - Values: Only 0 or 1
  - Why: Binary classification needs binary labels (even=0, odd=1)

MULTI-CLASS CLASSIFICATION MODEL:
  y_train (original one-hot encoded from preprocessing)
  - Format: One-hot encoded vectors
  - Example: [[0,0,0,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0,0], ...]
  - Shape: (60000, 10) - 10 values per sample (one for each class)
  - Why: Multi-class with softmax needs one-hot encoding

─────────────────────────────────────────────────────────────────────────────
5. ADDITIONAL LAYERS
─────────────────────────────────────────────────────────────────────────────

REGRESSION MODEL:
  - Only hidden layers: 30 → 20 neurons
  - No dropout

BINARY CLASSIFICATION MODEL:
  - Only hidden layers: 30 → 20 neurons
  - No dropout

MULTI-CLASS CLASSIFICATION MODEL:
  - Hidden layers: 30 → 20 neurons
  - PLUS: Dropout(0.2) layer before output
  - Why: Multi-class with 10 classes is more complex, dropout prevents overfitting


SUMMARY: WHY THESE DIFFERENCES MATTER

1. Output Layer Differences:
   - Regression: No activation → can output any value → suitable for continuous predictions
   - Binary: Sigmoid → outputs probability [0,1] → suitable for binary decisions
   - Multi-class: Softmax → outputs probability distribution → suitable for multiple classes

2. Loss Function Differences:
   - Each loss function is mathematically designed for its problem type
   - MSE works with continuous values
   - Cross-entropy works with probabilities
   - Using wrong loss function would give poor results

3. Target Data Differences:
   - Must match the problem type
   - Regression needs continuous targets
   - Classification needs class labels (binary or one-hot)

4. The Sequential Structure:
   - All models use Sequential() because they have simple linear flow
   - Input → Hidden Layers → Output
   - Sequential is perfect for these architectures
   - Same structure, different final layer and compilation settings

KEY TAKEAWAY:
The Sequential API provides a simple way to build models, but the CHOICE of:
  - Output layer (units, activation)
  - Loss function
  - Metrics
  - Target data format
...determines what type of problem the model solves (regression vs classification).
"""
