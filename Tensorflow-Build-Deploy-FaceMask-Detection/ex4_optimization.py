"""
Example 3: Calculating Loss for a Linear Model


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
W = tf.Variable([.3], tf.float32)

# Bias parameter (y-intercept)
# Initial value: -0.3
# Current model: y = 0.3*x - 0.3
b = tf.Variable([-.3], tf.float32)

# ======================== MODEL DEFINITION ========================
# Define the linear model: y_predicted = W*x + b
# This is the model's prediction function

# Input placeholder: Feature values (x values)
x = tf.placeholder(tf.float32)

# Linear model: Compute predictions
linear_model = W * x + b

# Target placeholder: Actual/true output values (y values)
y = tf.placeholder(tf.float32)

# ======================== LOSS CALCULATION ========================
# Calculate how far off our predictions are from the actual values
# Loss measures model performance: lower loss = better model

# Step 1: Calculate prediction errors (residuals)
squared_deltas = tf.square(linear_model - y)

# Step 2: Sum all squared errors
loss = tf.reduce_sum(squared_deltas)


# ======================== EXECUTION ========================
# Initialize variables and run the computation

# Create initialization operation for variables
init = tf.global_variables_initializer()

# Create session to execute the graph
session = tf.Session()

# Initialize variables (W and b set to their initial values)
session.run(init)

# Run loss calculation with actual data
# Model predictions with current parameters (W=0.3, b=-0.3):
loss_value = session.run(loss, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print("Loss value:", loss_value)
# Expected output: ~23.66 (high loss = poor model performance)

# ======================== OPTIMIZATION ========================
# WHY OPTIMIZATION?
# Current model has high loss (~23.66) because parameters (W=0.3, b=-0.3) are wrong
# True relationship: y = -1*x + 1 (W should be -1, b should be 1)
# Optimization automatically finds the best W and b values to minimize loss
# This is what "training" a model means: adjusting parameters to reduce error

# Create Gradient Descent Optimizer
# WHAT: Algorithm that finds optimal parameters by following the gradient (slope) of the loss function
# HOW: Calculates how loss changes with respect to each parameter, then updates parameters
# Parameters:
#   - learning_rate=0.01: Step size for parameter updates
#     * Controls how big steps we take when updating W and b
#     * Too high (e.g., 0.1): May overshoot optimal values, unstable training
#     * Too low (e.g., 0.001): Very slow convergence, needs many iterations
#     * 0.01 is a good balance for this problem
# WHAT IT DOES:
#   - Calculates gradients: ∂loss/∂W and ∂loss/∂b (how loss changes with each parameter)
#   - Updates parameters: W_new = W_old - learning_rate * ∂loss/∂W
#   - Moves parameters in direction that reduces loss (gradient descent)
optimizer = tf.train.GradientDescentOptimizer(0.01)

# Create training operation
# WHAT: Operation that performs one optimization step
# INPUT: Requires loss tensor (already defined above)
# OUTPUT: Operation that updates W and b when executed
# WHAT IT DOES WHEN RUN:
#   1. Calculates gradients of loss with respect to W and b
#   2. Updates W: W = W - 0.01 * gradient_W
#   3. Updates b: b = b - 0.01 * gradient_b
#   4. Each update moves parameters closer to optimal values
# NOTE: This doesn't execute yet - it's just defined. Execution happens in training loop.
train = optimizer.minimize(loss)

# ======================== TRAINING LOOP ========================
# WHY TRAINING LOOP?
# One optimization step improves parameters slightly, but not enough
# Need to repeat many times to reach optimal values
# Each iteration reduces loss and improves model accuracy

# Train for 1000 iterations (epochs)
# Each iteration: Calculate gradients → Update W and b → Reduce loss
for i in range(1000):
    # Execute one training step
    # INPUT (via feed_dict):
    #   - x: [1, 2, 3, 4] - Input features
    #   - y: [0, -1, -2, -3] - Target values (true relationship: y = -1*x + 1)
    # WHAT HAPPENS:
    #   1. Computes predictions: y_pred = W*x + b (using current W and b)
    #   2. Calculates loss: loss = Σ(y_pred - y)²
    #   3. Computes gradients: How loss changes with W and b
    #   4. Updates parameters: W and b adjusted to reduce loss
    # OUTPUT: Updated W and b values (stored in variables, not returned)
    # NOTE: W and b are modified in-place, so each iteration improves them
    session.run(train, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# ======================== RESULTS ========================
# After 1000 iterations, W and b should be close to optimal values
# Expected: W ≈ -1.0, b ≈ 1.0 (matching true relationship y = -1*x + 1)
# OUTPUT: List containing [W, b] - the optimized parameter values
# These are the learned parameters that minimize loss
optimized_params = session.run([W, b])
print("Optimized parameters:")
print(f"  W = {optimized_params[0][0]:.4f} (target: -1.0)")
print(f"  b = {optimized_params[1][0]:.4f} (target: 1.0)")

# Verify final loss (should be close to 0)
final_loss = session.run(loss, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print(f"Final loss: {final_loss:.6f} (target: 0.0)")

# Verify predictions match targets
final_predictions = session.run(linear_model, feed_dict={x: [1, 2, 3, 4]})
print(f"Predictions: {final_predictions}")
print(f"Targets:     {[0, -1, -2, -3]}")

# Close session to free resources
session.close()

# ======================== KEY CONCEPTS ========================
"""
OPTIMIZATION SUMMARY:

1. WHY OPTIMIZE?
   - Initial parameters (W=0.3, b=-0.3) give high loss (~23.66)
   - Need to find better parameters that minimize loss
   - Optimization automatically finds optimal W and b

2. HOW GRADIENT DESCENT WORKS:
   - Calculates gradient (slope) of loss function
   - Gradient points in direction of steepest increase
   - Move opposite to gradient (descent) to reduce loss
   - Formula: param_new = param_old - learning_rate * gradient

3. TRAINING PROCESS:
   - Start with random/initial parameters
   - For each iteration:
     a) Calculate predictions with current parameters
     b) Calculate loss (how wrong predictions are)
     c) Calculate gradients (how to adjust parameters)
     d) Update parameters (move toward better values)
   - Repeat until loss is minimized (convergence)

4. LEARNING RATE:
   - Controls step size for parameter updates
   - Too high: Overshoots minimum, may diverge
   - Too low: Slow convergence, many iterations needed
   - Need to find balance (often requires experimentation)

5. CONVERGENCE:
   - Model converges when loss stops decreasing
   - Parameters reach values that minimize loss
   - For this problem: W→-1, b→1, loss→0

6. WHAT WE LEARNED:
   - Started with: W=0.3, b=-0.3, loss=23.66
   - After optimization: W≈-1.0, b≈1.0, loss≈0.0
   - Model now correctly represents: y = -1*x + 1
"""
