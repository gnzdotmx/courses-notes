"""
Example 1: TensorFlow Sessions and Basic Computation

INTENTION:
This script demonstrates the fundamental concept of TensorFlow 1.x sessions and how
computations are executed in a graph-based manner. It shows the two-phase approach:
1. Build phase: Define computation graph (operations and tensors)
2. Execution phase: Run graph in a session to compute values

Key Learning Objectives:
- Understanding TensorFlow sessions
- Creating constants (immutable tensors)
- Building computation graphs
- Executing operations using sess.run()
- Graph-based vs eager execution

Note: This example uses TensorFlow 1.x style (sessions required).
TensorFlow 2.x uses eager execution by default (no sessions needed).

"""

import tensorflow as tf

# ======================== SESSION CREATION ========================
# Create a TensorFlow session
# Sessions are required in TensorFlow 1.x to execute operations
# A session allocates resources (CPU/GPU memory) and runs the computation graph
sess = tf.Session()

# ======================== BUILD PHASE: DEFINE COMPUTATION GRAPH ========================
# Define constants (immutable tensors) in the computation graph
# These operations are NOT executed yet - they just define the graph structure

# Create constant tensor 'a' with value 10
# tf.constant() creates an immutable tensor (cannot be changed after creation)
# dtype: Automatically inferred as tf.int32 (default for integer literals)
a = tf.constant(10)

# Create constant tensor 'b' with value 32
b = tf.constant(32)

# Define an operation: addition of two tensors
# This creates a new tensor in the graph representing the sum
# The operation is NOT executed yet - it's just added to the graph
# Result: A tensor representing (a + b), which will be 42 when executed
sum_operation = a + b

# ======================== EXECUTION PHASE: RUN THE GRAPH ========================
# Execute the computation using sess.run()
# sess.run() triggers the actual computation:
#   1. Evaluates the graph to compute the requested tensor
#   2. Returns the computed value as a numpy array or Python scalar
#   3. Only executes operations needed to compute the requested tensor
result = sess.run(sum_operation)
print(result)  # Output: 42

# ======================== CLEANUP ========================
# Close the session to free up resources
# Important: Always close sessions to release CPU/GPU memory
sess.close()

# ======================== ALTERNATIVE: USING CONTEXT MANAGER (RECOMMENDED) ========================
# Better practice: Use 'with' statement to automatically close session
# This ensures session is always closed, even if an error occurs
"""
with tf.Session() as sess:
    a = tf.constant(10)
    b = tf.constant(32)
    result = sess.run(a + b)
    print(result)  # Output: 42
# Session automatically closed when exiting 'with' block
"""

# ======================== KEY CONCEPTS ========================
"""
1. GRAPH VS EXECUTION:
   - Graph building: Defining operations (a, b, a+b) - happens immediately
   - Execution: Running sess.run() - happens only when explicitly called
   - This is "lazy evaluation" - operations are deferred until execution

2. SESSIONS:
   - Required in TensorFlow 1.x to execute operations
   - Allocate and manage computational resources
   - Run the computation graph
   - Must be closed to free resources

3. CONSTANTS:
   - Immutable tensors with fixed values
   - Defined at graph creation time
   - Cannot be changed after creation
   - Use for fixed values, hyperparameters, etc.

4. TENSOR OPERATIONS:
   - Operations return new tensors (don't modify inputs)
   - Operations are symbolic (not executed until sess.run())
   - Can chain operations: a + b + c

5. SESS.RUN():
   - Executes the computation graph
   - Can run single tensor or list of tensors
   - Returns computed values (numpy arrays or Python scalars)
   - Only executes operations needed for requested tensors
"""
