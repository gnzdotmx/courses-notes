"""
Real-time Face Mask Detection from Webcam

INTENTION:
This script performs real-time face mask detection using the computer's webcam.
It:
- Loads the trained mask detection model
- Captures video frames from the webcam
- Uses frame sampling to reduce computational load (processes every Nth frame)
- Preprocesses frames and runs predictions
- Displays live video feed with mask detection results overlaid
- Provides smooth real-time experience by reusing predictions between sampled frames

Key Features:
- Frame sampling algorithm for performance optimization
- Real-time video display with prediction overlays
- Smooth updates using prediction caching
- Configurable sampling rate and display settings
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# ======================== CONFIGURATION ========================

# Path to the trained model file.
# DESCRIPTION:
#   - Location of the saved model from 1_pretrain_model.py
#   - Must exist before running this script
MODEL_PATH = "mask_detection_model.h5"

# Frame sampling rate.
# DESCRIPTION:
#   - Process every Nth frame for prediction (reduces computational load)
#   - Higher value = fewer predictions = better performance but less responsive
#   - Lower value = more predictions = more responsive but higher CPU/GPU usage
# INPUTS:
#   - Integer value (e.g., 5 means process every 5th frame)
# OUTPUT:
#   - Used to determine when to run model prediction
# NOTES:
#   - Typical values: 3-10 for good balance
#   - For slower hardware, use higher values (10-15)
#   - For faster hardware, use lower values (2-5)
FRAME_SAMPLE_RATE = 5

# Camera index.
# DESCRIPTION:
#   - Which camera to use (0 = default/first camera)
#   - Increase if you have multiple cameras
# INPUTS:
#   - Integer (0, 1, 2, etc.)
CAMERA_INDEX = 0

# Display window name.
# DESCRIPTION:
#   - Name of the OpenCV window that displays the video feed
DISPLAY_WINDOW_NAME = "Face Mask Detection - Press 'q' to quit"

# Model input size (must match training).
# DESCRIPTION:
#   - Image dimensions expected by the model
#   - Must match the input_shape used during training (224x224 for MobileNetV2)
MODEL_INPUT_SIZE = (224, 224)

# Confidence threshold for displaying predictions.
# DESCRIPTION:
#   - Minimum confidence required to display prediction
#   - Below this threshold, shows "Uncertain" instead of prediction
# INPUTS:
#   - Float between 0.0 and 1.0
# NOTES:
#   - Higher = more conservative (only shows high-confidence predictions)
#   - Lower = shows more predictions but may include uncertain ones
CONFIDENCE_THRESHOLD = 0.5


# ======================== MODEL LOADING ========================

def load_mask_detection_model(model_path: str):
    """
    Load the trained face mask detection model from disk.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file (.h5 format)
    
    Returns
    -------
    model : tensorflow.keras.Model
        Loaded Keras model ready for inference
    
    Raises
    ------
    FileNotFoundError
        If the model file does not exist
    
    Behavior
    --------
    - Checks if model file exists
    - Loads model architecture and weights
    - Returns model ready for predictions
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find model file '{model_path}'. "
            "Train and save the model first (see 1_pretrain_model.py)."
        )
    
    model = load_model(model_path)
    return model


# ======================== FRAME PREPROCESSING ========================

def preprocess_frame(frame: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Preprocess a video frame for model input.
    
    Parameters
    ----------
    frame : np.ndarray
        Raw frame from camera (BGR format, shape: (H, W, 3))
    target_size : tuple
        Target image size (height, width) - must match model input
    
    Returns
    -------
    preprocessed_frame : np.ndarray
        Preprocessed frame ready for model (shape: (1, H, W, 3), values [0, 1])
    
    Behavior
    --------
    - Resizes frame to target size
    - Converts BGR to RGB (OpenCV uses BGR, model expects RGB)
    - Normalizes pixel values from [0, 255] to [0, 1]
    - Adds batch dimension for model input
    
    Notes
    -----
    - Preprocessing must match exactly how training images were processed
    - Inconsistent preprocessing leads to poor predictions
    """
    # Resize frame to model input size.
    # INPUTS:
    #   - frame: Original frame from camera
    #   - target_size: (224, 224) for MobileNetV2
    # OUTPUT:
    #   - Resized frame (may change aspect ratio)
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR (OpenCV default) to RGB (model expects RGB).
    # INPUTS:
    #   - resized: Frame in BGR color format
    # OUTPUT:
    #   - Frame in RGB color format
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize to [0, 1].
    # INPUTS:
    #   - rgb_frame: RGB frame with values [0, 255]
    # OUTPUT:
    #   - Normalized frame with values [0, 1]
    # NOTES:
    #   - Same normalization as training: rescale=1./255
    normalized = rgb_frame.astype(np.float32) / 255.0
    
    # Add batch dimension: (H, W, 3) -> (1, H, W, 3).
    # INPUTS:
    #   - normalized: Frame with shape (224, 224, 3)
    # OUTPUT:
    #   - Batch with shape (1, 224, 224, 3)
    # NOTES:
    #   - Model expects batch of images (even if batch size is 1)
    batch = np.expand_dims(normalized, axis=0)
    
    return batch


# ======================== PREDICTION ========================

def predict_mask(model, preprocessed_frame: np.ndarray) -> tuple:
    """
    Run mask detection prediction on a preprocessed frame.
    
    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained mask detection model
    preprocessed_frame : np.ndarray
        Preprocessed frame (shape: (1, 224, 224, 3))
    
    Returns
    -------
    prediction : tuple
        (class_index, confidence, probabilities)
        - class_index: 0 (with_mask) or 1 (without_mask)
        - confidence: Confidence score for predicted class (0.0 to 1.0)
        - probabilities: Array [P(with_mask), P(without_mask)]
    
    Behavior
    --------
    - Runs model forward pass
    - Extracts class probabilities
    - Determines predicted class (highest probability)
    - Returns prediction results
    """
    # Run model prediction.
    # INPUTS:
    #   - preprocessed_frame: Batch of preprocessed frames
    # OUTPUT:
    #   - Probabilities for batch: shape (1, 2)
    #   - Format: [[P(with_mask), P(without_mask)]]
    predictions = model.predict(preprocessed_frame, verbose=0)
    
    # Extract probabilities for first (and only) frame in batch.
    # INPUTS:
    #   - predictions: Array with shape (1, 2)
    # OUTPUT:
    #   - probs: Array with shape (2,)
    #   - Format: [P(with_mask), P(without_mask)]
    probs = predictions[0]
    
    # Get predicted class (index with highest probability).
    # INPUTS:
    #   - probs: Probability array [P(with_mask), P(without_mask)]
    # OUTPUT:
    #   - class_index: 0 (with_mask) or 1 (without_mask)
    class_index = int(np.argmax(probs))
    
    # Get confidence (probability of predicted class).
    # INPUTS:
    #   - probs: Probability array
    #   - class_index: Index of predicted class
    # OUTPUT:
    #   - confidence: Probability value for predicted class (0.0 to 1.0)
    confidence = float(probs[class_index])
    
    return class_index, confidence, probs


# ======================== DISPLAY OVERLAY ========================

def draw_prediction_overlay(frame: np.ndarray, class_index: int, 
                           confidence: float, probabilities: np.ndarray) -> np.ndarray:
    """
    Draw prediction results as overlay on video frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Original video frame (BGR format)
    class_index : int
        Predicted class (0 = with_mask, 1 = without_mask)
    confidence : float
        Confidence score for prediction (0.0 to 1.0)
    probabilities : np.ndarray
        Array [P(with_mask), P(without_mask)]
    
    Returns
    -------
    annotated_frame : np.ndarray
        Frame with prediction overlay drawn on it
    
    Behavior
    --------
    - Draws text overlay showing prediction and confidence
    - Uses color coding: Green for "with_mask", Red for "without_mask"
    - Displays probabilities for both classes
    - Adds visual indicators (rectangles, text)
    """
    # Create a copy to avoid modifying original frame.
    annotated = frame.copy()
    
    # Get frame dimensions for positioning text.
    height, width = frame.shape[:2]
    
    # Determine label and color based on prediction.
    # INPUTS:
    #   - class_index: 0 (with_mask) or 1 (without_mask)
    # OUTPUT:
    #   - label: Text label for display
    #   - color: BGR color tuple (Green for mask, Red for no mask)
    if class_index == 0:
        label = "WITH MASK"
        color = (0, 255, 0)  # Green in BGR
    else:
        label = "WITHOUT MASK"
        color = (0, 0, 255)  # Red in BGR
    
    # Draw background rectangle for text (improves readability).
    # INPUTS:
    #   - annotated: Frame to draw on
    #   - pt1, pt2: Rectangle corners
    #   - color: Rectangle color (black with transparency effect)
    #   - thickness: -1 = filled rectangle
    cv2.rectangle(annotated, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (400, 120), color, 2)
    
    # Draw main prediction label.
    # INPUTS:
    #   - annotated: Frame to draw on
    #   - text: Text to display
    #   - org: Text position (x, y)
    #   - font: Font type
    #   - scale: Font size
    #   - color: Text color
    #   - thickness: Text thickness
    cv2.putText(annotated, label, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    
    # Draw confidence percentage.
    # INPUTS:
    #   - confidence: Confidence value (0.0 to 1.0)
    # OUTPUT:
    #   - Displays "Confidence: XX.XX%"
    confidence_text = f"Confidence: {confidence * 100:.1f}%"
    cv2.putText(annotated, confidence_text, (20, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw detailed probabilities (optional, for debugging).
    # INPUTS:
    #   - probabilities: Array [P(with_mask), P(without_mask)]
    # OUTPUT:
    #   - Displays both probabilities
    prob_text = f"P(mask)={probabilities[0]:.2f} P(no_mask)={probabilities[1]:.2f}"
    cv2.putText(annotated, prob_text, (20, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw status indicator (colored circle or bar).
    # INPUTS:
    #   - annotated: Frame to draw on
    #   - center: Circle center position
    #   - radius: Circle radius
    #   - color: Circle color (matches label color)
    #   - thickness: -1 = filled circle
    cv2.circle(annotated, (width - 30, 30), 15, color, -1)
    
    return annotated


# ======================== MAIN VIDEO LOOP ========================

def run_real_time_detection(model, camera_index: int, sample_rate: int, 
                           confidence_threshold: float):
    """
    Main function that runs real-time mask detection from webcam.
    
    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained mask detection model
    camera_index : int
        Camera device index (0 = default camera)
    sample_rate : int
        Process every Nth frame (frame sampling rate)
    confidence_threshold : float
        Minimum confidence to display prediction
    
    Behavior
    --------
    - Opens camera and starts video capture
    - Processes frames with sampling (every Nth frame)
    - Reuses predictions between sampled frames for smooth display
    - Displays video feed with prediction overlays
    - Handles user input (quit on 'q' key)
    - Releases resources on exit
    
    Frame Sampling Algorithm:
    - Frame counter tracks current frame number
    - When frame_number % sample_rate == 0: Run prediction
    - Otherwise: Reuse previous prediction
    - This reduces computational load while maintaining smooth display
    """
    # Open camera for video capture.
    # INPUTS:
    #   - camera_index: Camera device index (0 = default)
    # OUTPUT:
    #   - cap: VideoCapture object
    # NOTES:
    #   - Returns None if camera cannot be opened
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}. Check camera connection.")
    
    # Set camera properties for better performance.
    # INPUTS:
    #   - cv2.CAP_PROP_FRAME_WIDTH: Desired frame width
    #   - cv2.CAP_PROP_FRAME_HEIGHT: Desired frame height
    #   - cv2.CAP_PROP_FPS: Frames per second
    # NOTES:
    #   - Lower resolution = faster processing
    #   - Adjust based on your camera capabilities
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize variables for frame sampling.
    # INPUTS:
    #   - frame_counter: Tracks current frame number
    #   - last_prediction: Caches last prediction result
    # OUTPUT:
    #   - Used to determine when to run new prediction
    frame_counter = 0
    last_prediction = None
    
    print("Starting real-time mask detection...")
    print("Press 'q' to quit")
    print(f"Processing every {sample_rate} frame(s) for performance")
    
    try:
        while True:
            # Read frame from camera.
            # INPUTS:
            #   - cap: VideoCapture object
            # OUTPUT:
            #   - ret: Boolean (True if frame read successfully)
            #   - frame: Frame data (BGR format, numpy array)
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Frame sampling: Process every Nth frame.
            # INPUTS:
            #   - frame_counter: Current frame number
            #   - sample_rate: Sampling rate (process every Nth frame)
            # OUTPUT:
            #   - should_predict: Boolean (True if should run prediction)
            # NOTES:
            #   - Reduces computational load by skipping frames
            #   - Example: sample_rate=5 means process frames 0, 5, 10, 15, ...
            should_predict = (frame_counter % sample_rate == 0)
            
            if should_predict:
                # Preprocess frame for model input.
                # INPUTS:
                #   - frame: Raw frame from camera
                #   - MODEL_INPUT_SIZE: (224, 224)
                # OUTPUT:
                #   - preprocessed: Preprocessed frame ready for model
                preprocessed = preprocess_frame(frame, MODEL_INPUT_SIZE)
                
                # Run prediction.
                # INPUTS:
                #   - model: Trained model
                #   - preprocessed: Preprocessed frame
                # OUTPUT:
                #   - class_index: Predicted class (0 or 1)
                #   - confidence: Confidence score
                #   - probabilities: Probability array
                class_index, confidence, probabilities = predict_mask(model, preprocessed)
                
                # Update cached prediction.
                # INPUTS:
                #   - last_prediction: Tuple of (class_index, confidence, probabilities)
                # OUTPUT:
                #   - Cached for reuse in non-sampled frames
                last_prediction = (class_index, confidence, probabilities)
            else:
                # Reuse previous prediction for smooth display.
                # INPUTS:
                #   - last_prediction: Cached prediction from last sampled frame
                # OUTPUT:
                #   - Uses cached values for display
                # NOTES:
                #   - Provides smooth updates between predictions
                #   - Reduces visual flickering
                if last_prediction is not None:
                    class_index, confidence, probabilities = last_prediction
                else:
                    # No prediction yet, skip this frame.
                    frame_counter += 1
                    continue
            
            # Only display prediction if confidence meets threshold.
            # INPUTS:
            #   - confidence: Confidence score (0.0 to 1.0)
            #   - confidence_threshold: Minimum confidence required
            # OUTPUT:
            #   - Determines whether to show prediction or "Uncertain"
            if confidence >= confidence_threshold:
                # Draw prediction overlay on frame.
                # INPUTS:
                #   - frame: Original frame
                #   - class_index: Predicted class
                #   - confidence: Confidence score
                #   - probabilities: Probability array
                # OUTPUT:
                #   - annotated_frame: Frame with overlay drawn
                annotated_frame = draw_prediction_overlay(
                    frame, class_index, confidence, probabilities
                )
            else:
                # Low confidence: Show "Uncertain" message.
                # INPUTS:
                #   - frame: Original frame
                # OUTPUT:
                #   - annotated_frame: Frame with "Uncertain" message
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, "UNCERTAIN - Low Confidence", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                           (0, 165, 255), 2)  # Orange color
            
            # Display frame with overlay.
            # INPUTS:
            #   - DISPLAY_WINDOW_NAME: Window name
            #   - annotated_frame: Frame with prediction overlay
            # OUTPUT:
            #   - Displays video feed in OpenCV window
            cv2.imshow(DISPLAY_WINDOW_NAME, annotated_frame)
            
            # Check for user input (quit on 'q' key).
            # INPUTS:
            #   - cv2.waitKey(1): Wait 1ms for key press
            # OUTPUT:
            #   - key: Key code (or -1 if no key pressed)
            # NOTES:
            #   - 'q' key (ASCII 113) quits the loop
            #   - waitKey(1) allows video to display smoothly
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            
            # Increment frame counter for sampling.
            frame_counter += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up resources.
        # INPUTS:
        #   - cap: VideoCapture object to release
        #   - cv2.destroyAllWindows(): Close all OpenCV windows
        # OUTPUT:
        #   - Releases camera and closes display windows
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


# ======================== MAIN ENTRY POINT ========================

def main():
    """
    Main entry point for the real-time mask detection application.
    
    Behavior
    --------
    - Loads the trained model
    - Initializes camera and starts real-time detection
    - Handles errors gracefully
    """
    try:
        # Load the trained model.
        # INPUTS:
        #   - MODEL_PATH: Path to saved model file
        # OUTPUT:
        #   - model: Loaded Keras model
        print(f"Loading model from {MODEL_PATH}...")
        model = load_mask_detection_model(MODEL_PATH)
        print("Model loaded successfully")
        
        # Start real-time detection.
        # INPUTS:
        #   - model: Loaded model
        #   - CAMERA_INDEX: Camera device index
        #   - FRAME_SAMPLE_RATE: Frame sampling rate
        #   - CONFIDENCE_THRESHOLD: Minimum confidence threshold
        # OUTPUT:
        #   - Starts video capture and detection loop
        run_real_time_detection(
            model=model,
            camera_index=CAMERA_INDEX,
            sample_rate=FRAME_SAMPLE_RATE,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have trained and saved the model first.")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Check that your camera is connected and not being used by another application.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
