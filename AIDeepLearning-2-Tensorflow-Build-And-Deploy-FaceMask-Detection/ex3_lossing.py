"""
Example 3: Calculating Loss for a Linear Model

INTENTION:
This script demonstrates how to calculate the loss (error) of a linear model by comparing
predictions with actual target values. It shows the fundamental concept of evaluating
model performance using Mean Squared Error (MSE) loss.

Key Learning Objectives:
- Understanding loss functions and their purpose
- Calculating prediction error (difference between predicted and actual values)
- Using squared error to measure model performance
- Interpreting loss values (lower is better, 0 = perfect)
- Understanding how model parameters affect loss

Loss Function: Sum of Squared Errors (SSE)
Formula: loss = Σ(predicted - actual)²
Lower loss = better model performance
"""

# ======================== TENSORFLOW SETUP ========================
# Enable TensorFlow 1.x compatibility mode for TensorFlow 2.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x eager execution

# ======================== MODEL PARAMETERS ========================
# Define trainable model parameters (weights and bias)
# These are the parameters we want to learn/optimize

# Weight parameter (slope of the line)
# Initial value: 0.3
# Note: If we set W=-1 and b=1, the model will be perfect and loss will be 0
#       because the true relationship is y = -1*x + 1
#       For x=[1,2,3,4], this gives y=[0,-1,-2,-3] (matches our targets)
W = tf.Variable([.3], tf.float32)

# Bias parameter (y-intercept)
# Initial value: -0.3
# Current model: y = 0.3*x - 0.3
b = tf.Variable([-.3], tf.float32)

# ======================== MODEL DEFINITION ========================
# Define the linear model: y_predicted = W*x + b
# This is the model's prediction function

# Input placeholder: Feature values (x values)
# Will receive input data during execution via feed_dict
x = tf.placeholder(tf.float32)

# Linear model: Compute predictions
# Formula: y_predicted = W * x + b
# This creates a tensor representing model predictions
linear_model = W * x + b

# Target placeholder: Actual/true output values (y values)
# These are the correct answers we want the model to predict
# Will be provided during execution via feed_dict
y = tf.placeholder(tf.float32)

# ======================== LOSS CALCULATION ========================
# Calculate how far off our predictions are from the actual values
# Loss measures model performance: lower loss = better model

# Step 1: Calculate prediction errors (residuals)
# squared_deltas = (predicted - actual)²
# This computes the squared difference for each data point
# Squaring ensures:
#   - All errors are positive (no cancellation)
#   - Large errors are penalized more heavily
# Example: If predicted=0.5 and actual=0, error = (0.5-0)² = 0.25
squared_deltas = tf.square(linear_model - y)

# Step 2: Sum all squared errors
# loss = Σ(predicted - actual)² (Sum of Squared Errors - SSE)
# This gives total error across all data points
# Lower values = better model (0 = perfect predictions)
loss = tf.reduce_sum(squared_deltas)

# Alternative: Mean Squared Error (MSE) - more common
# MSE = (1/n) * Σ(predicted - actual)²
# Uncomment to use MSE instead:
# loss = tf.reduce_mean(squared_deltas)

# ======================== EXECUTION ========================
# Initialize variables and run the computation

# Create initialization operation for variables
init = tf.global_variables_initializer()

# Create session to execute the graph
session = tf.Session()

# Initialize variables (W and b set to their initial values)
session.run(init)

# Run loss calculation with actual data
# Input data:
#   - x = [1, 2, 3, 4]: Input features
#   - y = [0, -1, -2, -3]: Target values (what we want to predict)
# 
# Model predictions with current parameters (W=0.3, b=-0.3):
#   x=1: y_pred = 0.3*1 - 0.3 = 0.0,  y_actual=0,  error=0.0² = 0.0
#   x=2: y_pred = 0.3*2 - 0.3 = 0.3,  y_actual=-1, error=(0.3-(-1))² = 1.69
#   x=3: y_pred = 0.3*3 - 0.3 = 0.6,  y_actual=-2, error=(0.6-(-2))² = 6.76
#   x=4: y_pred = 0.3*4 - 0.3 = 0.9,  y_actual=-3, error=(0.9-(-3))² = 15.21
#   Total loss = 0.0 + 1.69 + 6.76 + 15.21 = 23.66
loss_value = session.run(loss, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print("Loss value:", loss_value)
# Expected output: ~23.66 (high loss = poor model performance)

# Close session to free resources
session.close()

# ======================== UNDERSTANDING THE RESULTS ========================
"""
LOSS INTERPRETATION:
- Loss = 0: Perfect predictions (model matches all targets exactly)
- Low loss: Good model (predictions close to targets)
- High loss: Poor model (predictions far from targets)

CURRENT MODEL PERFORMANCE:
- Loss: ~23.66 (relatively high)
- Model: y = 0.3*x - 0.3
- Problem: Model parameters (W=0.3, b=-0.3) don't match true relationship
- True relationship: y = -1*x + 1 (if W=-1, b=1, loss would be 0)

HOW TO IMPROVE:
- Need to train/optimize W and b to minimize loss
- Use optimization algorithms (gradient descent) to find better parameters
- This is what training does: adjust parameters to reduce loss
"""

# ======================== KEY CONCEPTS ========================
"""
1. LOSS FUNCTION:
   - Measures how wrong the model's predictions are
   - Compares predicted values vs actual target values
   - Lower loss = better model performance
   - Loss = 0 means perfect predictions

2. SQUARED ERROR:
   - Error = predicted - actual
   - Squared error = (predicted - actual)²
   - Why square: Makes all errors positive, penalizes large errors more
   - Alternative: Absolute error = |predicted - actual| (less common)

3. SUM OF SQUARED ERRORS (SSE):
   - loss = Σ(predicted - actual)²
   - Sums errors across all data points
   - Used in this example

4. MEAN SQUARED ERROR (MSE):
   - MSE = (1/n) * Σ(predicted - actual)²
   - Average of squared errors
   - More common than SSE (normalized by number of samples)
   - Better for comparing models with different dataset sizes

5. LOSS AS MODEL EVALUATION:
   - Loss tells us how good/bad the model is
   - High loss = model needs improvement
   - Low loss = model is performing well
   - Loss guides training: adjust parameters to minimize loss

6. FEED_DICT WITH MULTIPLE PLACEHOLDERS:
   - Can provide values for multiple placeholders
   - Format: {placeholder1: value1, placeholder2: value2}
   - Both x and y are provided in the same feed_dict

7. MODEL PARAMETERS vs LOSS:
   - Different parameter values give different loss values
   - Goal: Find parameters that minimize loss
   - This is what optimization/training does
"""

# ======================== VISUALIZATION EXAMPLE ========================
"""
To better understand, imagine plotting:
- x-axis: Input values [1, 2, 3, 4]
- y-axis: Output values

True relationship (y = -1*x + 1):
  Points: (1,0), (2,-1), (3,-2), (4,-3) - perfect line

Current model (y = 0.3*x - 0.3):
  Points: (1,0), (2,0.3), (3,0.6), (4,0.9) - wrong line

Loss measures how far the model's line is from the true line.
"""
