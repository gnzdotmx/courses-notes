"""
Example: Use trained face mask detection model to predict a single image.

INTENTION:
This script loads the trained `mask_detection_model.h5` model and runs inference
on a single image path. It:
- Loads the saved Keras model from disk
- Loads and preprocesses an input image
- Runs `model.predict` to get class probabilities
- Prints the predicted class (with_mask / without_mask) and confidence
- Optionally displays the image using Matplotlib

NOTE:
- Make sure the image path you pass to `predict_image` is valid **relative to your
  current working directory** (where you run `python 2_predict.py`).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# ======================== LOAD TRAINED MODEL ========================

# Path to the saved model.
MODEL_PATH = "mask_detection_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find model file '{MODEL_PATH}'. "
        "Train and save the model first (see 1_pretrain_model.py)."
    )

# Load the trained model from disk.
model = load_model(MODEL_PATH)


# ======================== PREDICTION FUNCTION ========================

def predict_image(image_path: str):
    """
    Load an image from disk, run the mask detection model, and print prediction.

    Parameters
    ----------
    image_path : str
        Path to the image file to classify.

    Behavior
    --------
    - Loads the image with OpenCV for visualization.
    - Loads the image with Keras utilities for model input.
    - Preprocesses the image to match training (resize + scale to [0, 1]).
    - Runs model.predict to get probabilities for each class.
    - Prints the predicted class and confidence.
    """

    # -------- Load image with OpenCV for visualization --------
    image_bgr = cv2.imread(image_path)

    # If imread fails (returns None), the path is wrong or the file is missing.
    if image_bgr is None:
        raise FileNotFoundError(
            f"OpenCV could not read image at '{image_path}'. "
            f"Check that the file exists. Absolute path tried: "
            f"{os.path.abspath(image_path)}"
        )

    # Resize for consistent display.
    image_bgr = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Convert BGR (OpenCV default) to RGB for Matplotlib.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Show the image.
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show(block=False)
    


    # -------- Load and preprocess image for model input --------

    # Use Keras `load_img` to load and resize the image.
    keras_img = load_img(image_path, target_size=(224, 224))

    # Convert PIL image to NumPy array.
    img_array = img_to_array(keras_img)

    # Scale pixel values to [0, 1] (same as training `rescale=1./255`).
    img_array = img_array / 255.0

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # Run the model to get probabilities for each class.
    # Output shape: (1, 2) for two classes [with_mask, without_mask].
    probs = model.predict(img_batch)[0]

    # Get predicted class index: 0 or 1.
    pred_class = int(np.argmax(probs))

    print("Raw probabilities [with_mask, without_mask]:", probs)

    if pred_class == 0:
        print(f"Prediction: with_mask (confidence = {probs[0] * 100:.2f}%)")
    else:
        print(f"Prediction: without_mask (confidence = {probs[1] * 100:.2f}%)")

    plt.show(block=True)
# ======================== EXAMPLE USAGE ========================

# Example: adjust this path to point to an existing image on your system.
predict_image("dataset/val/with_mask/20-with-mask.jpg")
