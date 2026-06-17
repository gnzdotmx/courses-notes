"""
Example 2: Simple Linear Model with Variables and Placeholders

INTENTION:
This script demonstrates building a simple linear model (y = W*x + b) using TensorFlow 1.x
concepts: Variables (trainable parameters), Placeholders (input data), and Sessions.

Key Learning Objectives:
- Understanding Variables (trainable model parameters)
- Understanding Placeholders (input data nodes)
- Building a simple linear model
- Variable initialization
- Feeding data to placeholders during execution

IMPORTANT: This code uses TensorFlow 1.x syntax.
If using TensorFlow 2.x, you need to enable compatibility mode (see below).
"""

# ======================== TENSORFLOW 2.x COMPATIBILITY MODE ========================
# Enable TensorFlow 1.x behavior in TensorFlow 2.x
# This allows using tf.placeholder, tf.Session, etc.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x eager execution

# Alternative: If you have TensorFlow 1.x installed, use:
# import tensorflow as tf

# ======================== BUILD PHASE: DEFINE MODEL ========================
# Define model parameters as Variables (trainable parameters)
# Variables are mutable tensors that can be updated during training

# Weight parameter (slope of the line)
# Parameters:
#   - [.3]: Initial value (0.3)
#   - tf.float32: Data type (32-bit floating point)
# Variables must be initialized before use
W = tf.Variable([.3], tf.float32)

# Bias parameter (y-intercept of the line)
# Initial value: 0.3
b = tf.Variable([.3], tf.float32)

# Define input placeholder (receives data at runtime)
# Placeholders are input nodes that don't have values until execution
# Parameters:
#   - tf.float32: Data type for input values
#   - shape: Not specified (None) - can accept any shape
#   - Values will be provided via feed_dict during sess.run()
x = tf.placeholder(tf.float32)

# Define linear model: y = W*x + b
# This creates a computation graph node
# Operation is NOT executed yet - just defined in the graph
linear_model = W * x + b

# ======================== VARIABLE INITIALIZATION ========================
# Create operation to initialize all variables
# Variables must be initialized before they can be used
# This creates an operation that sets variables to their initial values
init = tf.global_variables_initializer()

# ======================== EXECUTION PHASE: RUN THE MODEL ========================
# Create session to execute the computation graph
sess = tf.Session()

# Initialize variables (must run before using variables)
# This sets W = [0.3] and b = [0.3]
sess.run(init)

# Run the linear model with input data
# Parameters:
#   - linear_model: Tensor to compute
#   - feed_dict: Dictionary mapping placeholders to actual values
#     * x: [1, 2, 3, 4] - input values
# Computation: y = 0.3 * [1,2,3,4] + 0.3 = [0.6, 0.9, 1.2, 1.5]
result = sess.run(linear_model, feed_dict={x: [1, 2, 3, 4]})
print("Linear model predictions:", result)
# Expected output: [0.6 0.9 1.2 1.5]

# Close session to free resources
sess.close()

# ======================== ALTERNATIVE: USING CONTEXT MANAGER ========================
# Better practice: Use 'with' statement for automatic cleanup
"""
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)
    
    # Run model with different inputs
    result1 = sess.run(linear_model, feed_dict={x: [1, 2, 3, 4]})
    result2 = sess.run(linear_model, feed_dict={x: [5, 10, 15]})
    
    print("Predictions for [1,2,3,4]:", result1)
    print("Predictions for [5,10,15]:", result2)
"""

# ======================== KEY CONCEPTS ========================
"""
1. VARIABLES vs CONSTANTS:
   - Constants: Immutable, fixed values (tf.constant)
   - Variables: Mutable, can be updated (tf.Variable)
   - Variables are used for model parameters (weights, biases)
   - Variables must be initialized before use

2. PLACEHOLDERS:
   - Input nodes that receive data at runtime
   - No values until execution (via feed_dict)
   - Used for training data, inputs that change each run
   - Shape and dtype specified, but values provided later

3. VARIABLE INITIALIZATION:
   - Variables start with initial values but aren't "active" until initialized
   - tf.global_variables_initializer() creates initialization operation
   - Must run sess.run(init) before using variables

4. FEED_DICT:
   - Dictionary mapping placeholders to actual values
   - Used during sess.run() to provide input data
   - Format: {placeholder: value}
   - Example: {x: [1, 2, 3, 4]}

5. LINEAR MODEL:
   - Formula: y = W*x + b
   - W: Weight (slope) - how much x affects y
   - b: Bias (y-intercept) - base value when x=0
   - This is the simplest machine learning model

6. TENSORFLOW 1.x vs 2.x:
   - TF 1.x: Uses placeholders, sessions, graph mode (this example)
   - TF 2.x: Eager execution by default (no placeholders/sessions)
   - To use TF 1.x code in TF 2.x: Enable compatibility mode (see top of file)
"""

# ======================== TENSORFLOW 2.x EQUIVALENT (FOR REFERENCE) ========================
"""
If using TensorFlow 2.x with eager execution (no compatibility mode):

import tensorflow as tf

# Variables (same concept)
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([.3], dtype=tf.float32)

# No placeholder needed - just use regular Python variables
x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)

# Model (executes immediately)
linear_model = W * x + b

# No session needed - operations execute immediately
print("Linear model predictions:", linear_model.numpy())
"""
