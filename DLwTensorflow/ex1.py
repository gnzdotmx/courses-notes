"""
Example 1: MNIST Dataset Loading and Visualization

INTENTION:
This script demonstrates the basic workflow of loading and exploring the MNIST handwritten 
digit dataset. It serves as an introduction to working with image data in TensorFlow/Keras.

Key Learning Objectives:
- Load the MNIST dataset using TensorFlow/Keras
- Understand the dataset structure (training vs test sets)
- Visualize image data using matplotlib
- Explore data shapes and dimensions

The MNIST dataset contains:
- 60,000 training images of handwritten digits (0-9)
- 10,000 test images
- Each image is 28x28 pixels (grayscale)
- Labels are integers from 0 to 9
"""

import tensorflow as tf
import matplotlib.pyplot as plt

# Print TensorFlow version to verify installation
# Version information helps ensure compatibility with code examples
print("TensorFlow version:", tf.__version__)

# Load MNIST dataset
# MNIST is a built-in dataset in Keras, commonly used for learning deep learning
mnist = tf.keras.datasets.mnist

# Load data splits
# Returns two tuples: (training_data, training_labels) and (test_data, test_labels)
# x_train: 60,000 images of shape (28, 28) - grayscale pixel values [0, 255]
# y_train: 60,000 labels - integers from 0 to 9
# x_test: 10,000 images of shape (28, 28) - grayscale pixel values [0, 255]
# y_test: 10,000 labels - integers from 0 to 9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("\nDataset loaded")
print("x_train.shape:", x_train.shape)  # Expected: (60000, 28, 28)
print("y_train.shape:", y_train.shape)  # Expected: (60000,)
print("x_test.shape:", x_test.shape)    # Expected: (10000, 28, 28)
print("y_test.shape:", y_test.shape)    # Expected: (10000,)

# Display the first training image
# This helps visualize what the data looks like
print("\nDisplaying first image")
print("y_train (label):", y_train[0])  # The true label for the first image
print("x_train (image):", x_train[0])  # The pixel values (28x28 matrix)

# Visualize the image
# imshow displays the 2D array as an image
# cmap='gray' is optional but helps visualize grayscale images better
plt.imshow(x_train[0], cmap='gray')
plt.title(f"First training image - Label: {y_train[0]}")
plt.axis('off')  # Hide axes for cleaner visualization
plt.show()
