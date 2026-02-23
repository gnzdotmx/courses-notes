"""
Example 7: Simple Addition and Linear Regression with TensorFlow

INTENTION:
This script demonstrates two concepts:
1. Simple addition of constants in TensorFlow
2. Complete linear regression example: learning to fit a line to data points

Key Learning Objectives:
- Basic arithmetic operations with constants
- Creating synthetic data for training
- Building a linear regression model
- Training model to learn parameters (W and b)
- Using Mean Squared Error (MSE) loss
- Gradient descent optimization
- Observing parameter convergence during training
"""

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()  # Disable TensorFlow 2.x eager execution

# ======================== PART 1: SIMPLE ADDITION ========================
# Simple example: Adding two constants

# Create session to execute operations
session = tf.Session()

# tf.constant() - Create constant tensor
# DESCRIPTION: Creates an immutable tensor with a fixed value
# INPUTS:
#   - value: 10 (the constant value)
#   - dtype: Automatically inferred as tf.int32
# OUTPUT: Constant tensor with value 10
# NOTES:
#   - Constants are immutable (cannot be changed)
#   - Value is known at graph construction time
#   - No initialization needed (unlike Variables)
a = tf.constant(10)

# tf.constant() - Create another constant
# DESCRIPTION: Creates an immutable tensor with a fixed value
# INPUTS:
#   - value: 22 (the constant value)
# OUTPUT: Constant tensor with value 22
b = tf.constant(22)

# Addition operation and execution
# DESCRIPTION: Adds two tensors and executes the computation
# INPUTS:
#   - a: Constant tensor (10)
#   - b: Constant tensor (22)
# OUTPUT: Computed value 32 (10 + 22)
# NOTES:
#   - a + b creates a new tensor representing the sum
#   - sess.run() executes the computation and returns the result
#   - Can use + operator or tf.add(a, b) - both are equivalent
print("Simple addition result:", session.run(a + b))  # Output: 32

session.close()

# ======================== PART 2: LINEAR REGRESSION ========================
# Complete example: Learning to fit a line to data points

# Generate synthetic training data
# DESCRIPTION: Create artificial data points following a linear relationship
# INPUTS:
#   - np.random.rand(100): Generates 100 random numbers in [0, 1)
#   - .astype(np.float32): Converts to float32 (required for TensorFlow)
# OUTPUT: x_data - numpy array of 100 random float32 values
# NOTES:
#   - Synthetic data: y = 0.1 * x + 0.3 (true relationship)
#   - We'll train model to learn W=0.1 and b=0.3
#   - Using synthetic data allows us to verify model learns correctly
x_data = np.random.rand(100).astype(np.float32)

# Calculate target values (y) based on true relationship
# DESCRIPTION: Generate y values using the true linear relationship
# INPUTS:
#   - x_data: Input features (100 random values)
#   - Formula: y = 0.1 * x + 0.3
# OUTPUT: y_data - numpy array of 100 target values
# NOTES:
#   - True relationship: y = 0.1*x + 0.3
#   - Model should learn W=0.1 and b=0.3
#   - This is the "ground truth" we want the model to discover
y_data = x_data * 0.1 + 0.3

# ======================== MODEL PARAMETERS ========================
# Define trainable parameters that the model will learn

# tf.random_uniform() - Generate random values from uniform distribution
# DESCRIPTION: Creates tensor with random values from uniform distribution
# INPUTS:
#   - shape: [1] (shape of output tensor - 1 element)
#   - minval: -1.0 (minimum value)
#   - maxval: 1.0 (maximum value)
# OUTPUT: Tensor with random value in range [-1.0, 1.0)
# NOTES:
#   - Used for random weight initialization
#   - Starting with random values, model will learn optimal W
#   - Different initializations can affect training (but should converge to same result)
#   - Returns: Random value between -1.0 and 1.0

# tf.Variable() - Create trainable variable for weight
# DESCRIPTION: Creates mutable tensor for model weight parameter
# INPUTS:
#   - initial_value: tf.random_uniform([1], -1.0, 1.0) (random initial value)
# OUTPUT: Variable tensor (will be optimized during training)
# NOTES:
#   - W represents the slope of the line (should learn to be 0.1)
#   - Starts with random value, will be updated by optimizer
#   - Must be initialized before use
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# tf.zeros() - Create tensor filled with zeros
# DESCRIPTION: Creates tensor with all elements set to zero
# INPUTS:
#   - shape: [1] (shape of output tensor - 1 element)
# OUTPUT: Tensor with value [0.0]
# NOTES:
#   - Common initialization for bias terms
#   - Starting bias at 0 is often a good default

# tf.Variable() - Create trainable variable for bias
# DESCRIPTION: Creates mutable tensor for model bias parameter
# INPUTS:
#   - initial_value: tf.zeros([1]) (initial value of 0.0)
# OUTPUT: Variable tensor (will be optimized during training)
# NOTES:
#   - b represents the y-intercept (should learn to be 0.3)
#   - Starts at 0, will be updated by optimizer
#   - Must be initialized before use
b = tf.Variable(tf.zeros([1]))

# ======================== MODEL DEFINITION ========================
# Define the linear model: y = W*x + b

# Linear model computation
# DESCRIPTION: Computes model predictions using current parameters
# INPUTS:
#   - W: Weight variable (slope)
#   - x_data: Input features (100 data points)
#   - b: Bias variable (y-intercept)
# OUTPUT: Tensor representing predictions y = W*x_data + b
# NOTES:
#   - Element-wise multiplication and addition
#   - Predictions for all 100 data points
#   - Initially wrong (random W, b=0), will improve during training
y = W * x_data + b

# ======================== LOSS FUNCTION ========================
# Measure how wrong our predictions are

# tf.square() - Element-wise squaring
# DESCRIPTION: Computes square of each element in tensor
# INPUTS:
#   - x: (y - y_data) - prediction errors (residuals)
# OUTPUT: Tensor with squared errors
# NOTES:
#   - Squares the difference between predictions and targets
#   - Makes all errors positive, penalizes large errors more

# tf.reduce_mean() - Mean of elements across dimensions
# DESCRIPTION: Computes the mean (average) of tensor elements
# INPUTS:
#   - input_tensor: tf.square(y - y_data) (squared errors)
# OUTPUT: Scalar representing Mean Squared Error (MSE)
# NOTES:
#   - MSE = (1/n) * Σ(predicted - actual)²
#   - Average squared error across all data points
#   - Lower is better (0 = perfect predictions)
#   - More common than sum of squared errors (normalized by dataset size)
loss = tf.reduce_mean(tf.square(y - y_data))

# ======================== OPTIMIZATION ========================
# Create optimizer to minimize loss

# tf.train.GradientDescentOptimizer() - Gradient descent optimizer
# DESCRIPTION: Optimizer that minimizes loss using gradient descent algorithm
# INPUTS:
#   - learning_rate: 0.5 (step size for parameter updates)
# OUTPUT: Optimizer object
# NOTES:
#   - Learning rate 0.5 is relatively high (good for this simple problem)
#   - Higher learning rate = faster convergence but risk of overshooting
#   - For this problem, 0.5 works well (converges quickly)
optimizer = tf.train.GradientDescentOptimizer(0.5)

# optimizer.minimize() - Create training operation
# DESCRIPTION: Creates operation that performs one optimization step
# INPUTS:
#   - loss: Loss tensor to minimize
# OUTPUT: Training operation
# NOTES:
#   - When executed, calculates gradients and updates W and b
#   - Moves parameters in direction that reduces loss
#   - Each execution improves model slightly
train = optimizer.minimize(loss)

# ======================== INITIALIZATION ========================
# Initialize all variables before training

# tf.initialize_all_variables() - Initialize all variables
# DESCRIPTION: Creates operation to initialize all variables in the graph
# INPUTS: None (automatically finds all variables)
# OUTPUT: Initialization operation
# NOTES:
#   - Must run before using variables
#   - Sets W and b to their initial values
#   - Note: In newer TensorFlow, use tf.global_variables_initializer()
#   - This is deprecated but still works
init = tf.initialize_all_variables()

# Create session and initialize
session = tf.Session()
session.run(init)

# ======================== TRAINING LOOP ========================
# Iteratively improve model by minimizing loss

# Training loop: Fit the line to data
# DESCRIPTION: Repeatedly update parameters to minimize loss
# INPUTS:
#   - range(201): 201 training iterations (steps)
# OUTPUT: Trained model with optimized W and b
# NOTES:
#   - Each iteration: Calculate loss → Compute gradients → Update W and b
#   - Model gradually learns W≈0.1 and b≈0.3
#   - Loss decreases with each iteration
#   - After 201 steps, model should be well-fitted
for step in range(201):
    # Execute one training step
    # DESCRIPTION: Runs optimizer to update parameters
    # INPUTS: None (uses loss, W, b, x_data, y_data from graph)
    # OUTPUT: None (updates W and b in-place)
    # NOTES:
    #   - Updates W and b to reduce loss
    #   - No feed_dict needed (x_data and y_data are numpy arrays, not placeholders)
    #   - Each step moves parameters closer to optimal values
    session.run(train)
    
    # Print progress every 20 steps
    # DESCRIPTION: Monitor training progress
    # INPUTS:
    #   - step: Current iteration number
    #   - session.run(W): Current weight value
    #   - session.run(b): Current bias value
    # OUTPUT: Prints step number and parameter values
    # NOTES:
    #   - Shows how W and b change during training
    #   - W should approach 0.1, b should approach 0.3
    #   - Useful for debugging and understanding convergence
    if step % 20 == 0:
        current_W = session.run(W)
        current_b = session.run(b)
        current_loss = session.run(loss)
        print(f"Step {step}: W = {current_W[0]:.4f}, b = {current_b[0]:.4f}, Loss = {current_loss:.6f}")

session.close()

# ======================== EXPECTED RESULTS ========================
"""
After training, the model should have learned:
- W ≈ 0.1 (close to true slope)
- b ≈ 0.3 (close to true y-intercept)
- Loss ≈ 0.0 (very small, close to perfect fit)

The model successfully learned the linear relationship: y = 0.1*x + 0.3
"""

# ======================== KEY CONCEPTS ========================
"""
1. SYNTHETIC DATA:
   - Created artificial data with known relationship
   - Allows verification that model learns correctly
   - True relationship: y = 0.1*x + 0.3

2. RANDOM INITIALIZATION:
   - W starts with random value (between -1 and 1)
   - b starts at 0
   - Model learns optimal values through training

3. MEAN SQUARED ERROR (MSE):
   - Loss = average of (predicted - actual)²
   - Measures average prediction error
   - Lower loss = better model

4. GRADIENT DESCENT:
   - Automatically finds W and b that minimize loss
   - Updates parameters in direction that reduces error
   - Converges to optimal values

5. TRAINING PROGRESS:
   - Initially: W is random, b=0, high loss
   - Gradually: W→0.1, b→0.3, loss→0
   - Model learns the true relationship

6. NO PLACEHOLDERS:
   - x_data and y_data are numpy arrays (not placeholders)
   - Data is embedded in graph (simpler for this example)
   - For larger datasets, use placeholders with feed_dict
"""
