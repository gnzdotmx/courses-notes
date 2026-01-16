#!/usr/bin/env python3
"""
Image Prediction Script for Cats vs Dogs Classification

INTENTION:
This script demonstrates how to use a trained CNN model to make predictions on new images.
It covers the complete inference workflow: loading a saved model, preprocessing images,
making predictions, and displaying results.

Key Learning Objectives:
- Loading saved Keras models
- Image preprocessing for inference (must match training preprocessing)
- Making predictions with trained models
- Interpreting prediction probabilities
- Binary classification thresholding

Usage: 
    python predict.py <image_path> [--model <model_path>]
    python predict.py test.jpg
    python predict.py image.jpg --model best_model.h5
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

def predict_image(image_path, model_path="best_model.h5", show_image=True):
    """
    Make prediction on a single image using the trained model.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the saved model (default: best_model.h5)
        show_image: Whether to display the image with prediction (default: True)
    
    Returns:
        tuple: (predicted_class, probability, class_name)
    """
    # ======================== VALIDATION ========================
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train a model first or specify the correct model path.")
        sys.exit(1)
    
    # ======================== MODEL LOADING ========================
    # Load the trained model from saved file
    # load_model() restores the complete model including:
    #   - Architecture (layers)
    #   - Weights (learned parameters)
    #   - Compilation settings (optimizer, loss, metrics)
    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # ======================== CLASS MAPPING ========================
    # Define class names for binary classification
    # This mapping must match the training data class indices
    # During training: class_mode="binary" creates indices {'cats': 0, 'dogs': 1}
    # Important: Ensure this matches your training setup
    class_names = {0: "cat", 1: "dog"}
    
    # ======================== IMAGE PREPROCESSING ========================
    # CRITICAL: Preprocessing must match training preprocessing exactly
    # Any mismatch will lead to poor predictions
    print(f"\nProcessing image: {image_path}")
    try:
        # Step 1: Load and resize image
        # Parameters:
        #   - target_size=(224, 224): Must match model input size
        #     * For CNN from scratch: (224, 224)
        #     * For VGG16 transfer learning: (224, 224)
        #   - load_img() returns PIL Image object
        pil_image = load_img(image_path, target_size=(224, 224))
        
        # Step 2: Convert PIL Image to numpy array
        # Output shape: (224, 224, 3) - height, width, RGB channels
        # Pixel values: [0, 255] (integers)
        image = img_to_array(pil_image)
        
        # Step 3: Normalize pixel values to [0, 1]
        # CRITICAL: Must match training normalization (rescale=1./255)
        # Why: Model was trained on normalized images, expects same format
        # Formula: normalized = original / 255.0
        image = image / 255.0
        # Now pixel values are floats in range [0.0, 1.0]
        
        # Step 4: Add batch dimension
        # Model expects input shape: (batch_size, height, width, channels)
        # Current shape: (224, 224, 3)
        # Required shape: (1, 224, 224, 3) - batch of 1 image
        # expand_dims(axis=0) adds dimension at position 0
        image_batch = np.expand_dims(image, axis=0)
        
        print("Image preprocessed successfully!")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)
    
    # ======================== PREDICTION ========================
    # Make prediction using the trained model
    print("\nMaking prediction...")
    try:
        # Run forward pass through the model
        # Parameters:
        #   - image_batch: Preprocessed image with batch dimension (1, 224, 224, 3)
        #   - verbose=0: Suppress prediction progress output
        # Returns: Array of predictions (one per image in batch)
        #   Shape: (batch_size, 1) for binary classification
        #   Example: [[0.87]] means 87% probability of class 1 (dog)
        predictions = model.predict(image_batch, verbose=0)
        
        # Extract probability for the single image
        # predictions[0] = first (and only) image in batch
        # predictions[0][0] = probability value (single output neuron)
        probability = predictions[0][0]
        
        # Binary classification thresholding
        # Model outputs probability [0, 1] from sigmoid activation
        # Decision rule:
        #   - probability > 0.5 → predict class 1 (dog)
        #   - probability ≤ 0.5 → predict class 0 (cat)
        # Why 0.5: Standard threshold for binary classification (equal cost for both classes)
        predicted_class = 1 if probability > 0.5 else 0
        class_name = class_names[predicted_class]
        
        print(f"\n{'='*50}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*50}")
        print(f"Image: {image_path}")
        print(f"Predicted class: {class_name}")
        print(f"Probability: {probability:.4f} ({probability*100:.2f}%)")
        
        # Calculate confidence level based on distance from decision boundary (0.5)
        # High confidence: Far from 0.5 (>0.3 away) - e.g., 0.9 or 0.1
        # Medium confidence: Moderate distance (>0.1 away) - e.g., 0.7 or 0.3
        # Low confidence: Close to 0.5 (≤0.1 away) - e.g., 0.6 or 0.4
        confidence_threshold = abs(probability - 0.5)
        if confidence_threshold > 0.3:
            confidence = "High"
        elif confidence_threshold > 0.1:
            confidence = "Medium"
        else:
            confidence = "Low"
        print(f"Confidence: {confidence}")
        print(f"{'='*50}\n")
        
        # Display image with prediction overlay
        if show_image:
            plt.figure(figsize=(8, 8))
            # Display the normalized image (values [0,1] are automatically scaled for display)
            plt.imshow(image)
            plt.title(
                f"Predicted: {class_name.upper()}\nProbability: {probability:.4f} ({probability*100:.2f}%)", 
                fontsize=14, 
                fontweight='bold'
            )
            plt.axis('off')  # Hide axes for cleaner visualization
            plt.tight_layout()
            plt.show()
        
        return class_name, probability, predicted_class
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and run prediction."""
    parser = argparse.ArgumentParser(
        description='Predict cats vs dogs from an image using trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py test.jpg
  python predict.py /path/to/image.jpg --model best_model.h5
  python predict.py image.jpg --model model.h5 --no-show
        """
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file to predict'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='best_model.h5',
        help='Path to the model file (default: best_model.h5)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the image (useful for batch processing)'
    )
    
    args = parser.parse_args()
    
    # Run prediction
    predict_image(
        image_path=args.image_path,
        model_path=args.model,
        show_image=not args.no_show
    )


if __name__ == "__main__":
    main()

# ======================== ESSENTIAL CONCEPTS SUMMARY ========================
"""
ESSENTIAL CONCEPTS IN THIS SCRIPT:

1. INFERENCE VS TRAINING:
   - Training: Model learns from data (forward + backward pass, weight updates)
   - Inference: Model makes predictions on new data (forward pass only, no weight updates)
   - This script performs INFERENCE only

2. MODEL LOADING:
   - load_model(): Restores complete model (architecture + weights + settings)
   - Model file (.h5): Contains everything needed for predictions
   - No need to rebuild architecture or compile again

3. IMAGE PREPROCESSING FOR INFERENCE:
   - CRITICAL: Must match training preprocessing exactly
   - Steps:
     a) Resize to model input size (224×224)
     b) Convert to array (PIL → numpy)
     c) Normalize pixels [0,255] → [0,1]
     d) Add batch dimension (1, 224, 224, 3)
   - Why: Model expects same format it was trained on

4. BATCH DIMENSION:
   - Models expect batch dimension even for single images
   - Single image: (224, 224, 3) → (1, 224, 224, 3)
   - Multiple images: (N, 224, 224, 3) where N = batch size
   - expand_dims(axis=0): Adds batch dimension at position 0

5. PREDICTION OUTPUT:
   - Binary classification: Single probability value [0, 1]
   - Interpretation:
     * 0.0-0.5: Class 0 (cat) with varying confidence
     * 0.5-1.0: Class 1 (dog) with varying confidence
     * 0.5: Maximum uncertainty (equal probability)
   - Threshold: 0.5 is standard, but can be adjusted for imbalanced data

6. CONFIDENCE INTERPRETATION:
   - High confidence: Probability far from 0.5 (e.g., 0.9 or 0.1)
   - Low confidence: Probability close to 0.5 (e.g., 0.6 or 0.4)
   - Why important: Helps identify uncertain predictions

7. COMMON ERRORS TO AVOID:
   - Wrong image size: Model expects specific input size
   - Missing normalization: Model trained on [0,1], not [0,255]
   - Missing batch dimension: Model expects (batch, height, width, channels)
   - Wrong preprocessing order: Resize → Array → Normalize → Batch

8. BATCH PROCESSING:
   - Can process multiple images at once
   - More efficient than one-by-one
   - Example: image_batch = np.array([img1, img2, img3])
   - Output: Array of predictions, one per image

KEY DIFFERENCES FROM TRAINING:
- No data augmentation (use original images)
- No labels needed (making predictions, not learning)
- Single forward pass (no backward pass, no weight updates)
- Model in evaluation mode (dropout disabled, batch norm in inference mode)
"""
