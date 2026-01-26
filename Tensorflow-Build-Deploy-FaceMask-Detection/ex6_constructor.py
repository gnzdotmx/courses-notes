"""
Example 6: TensorFlow Constructors and Operations

INTENTION:
This script demonstrates various TensorFlow constructors and operations, including:
- Variables and their initialization
- Basic arithmetic operations (add, multiply)
- Reduction operations (reduce_prod, reduce_sum)
- Placeholders and feed_dict
- Evaluating tensors

Key Learning Objectives:
- Understanding tf.Variable constructor
- Understanding tf.add() and tf.multiply() operations
- Understanding reduction operations (reduce_prod, reduce_sum)
- Understanding placeholder and feed_dict usage
- Understanding eval() vs sess.run()
"""

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()  # Disable TensorFlow 2.x eager execution

# ======================== PART 1: VARIABLES AND BASIC OPERATIONS ========================

# tf.Variable() - Create a trainable variable
# DESCRIPTION: Creates a mutable tensor that can be updated during training
# INPUTS:
#   - initial_value: 4 (initial value for the variable)
#   - name: "my_variable" (optional name for the variable in the graph)
# OUTPUT: Variable tensor with value 4
# NOTES:
#   - Variables must be initialized before use
#   - Variables are trainable (can be updated by optimizers)
#   - Different from constants (immutable) and placeholders (receive values at runtime)
my_variable = tf.Variable(4, name="my_variable")

# tf.add() - Element-wise addition
# DESCRIPTION: Adds two tensors element-wise
# INPUTS:
#   - x: 5 (scalar constant)
#   - y: my_variable (Variable tensor with value 4)
# OUTPUT: Tensor representing 5 + 4 = 9
# NOTES:
#   - Can add scalars, vectors, matrices (must be broadcastable)
#   - Returns new tensor (doesn't modify inputs)
#   - Operation is lazy (not executed until sess.run() or eval())
add = tf.add(5, my_variable)

# tf.multiply() - Element-wise multiplication
# DESCRIPTION: Multiplies two tensors element-wise
# INPUTS:
#   - x: 8 (scalar constant)
#   - y: my_variable (Variable tensor with value 4)
# OUTPUT: Tensor representing 8 * 4 = 32
# NOTES:
#   - Element-wise multiplication (not matrix multiplication)
#   - For matrix multiplication, use tf.matmul()
#   - Returns new tensor (doesn't modify inputs)
multiply = tf.multiply(8, my_variable)

# Create session to execute operations
session = tf.Session()

# Variable.initializer - Initialize a single variable
# DESCRIPTION: Initializes the variable to its initial value
# INPUTS: None (uses variable's initial_value)
# OUTPUT: Initialization operation
# NOTES:
#   - Must initialize variables before using them
#   - Alternative: tf.global_variables_initializer() initializes all variables
#   - This initializes my_variable to value 4
session.run(my_variable.initializer)

# tensor.eval() - Evaluate a tensor in a session
# DESCRIPTION: Computes and returns the value of a tensor
# INPUTS:
#   - session: Session object to run the computation
# OUTPUT: Computed value (numpy array or Python scalar)
# NOTES:
#   - Equivalent to sess.run(tensor)
#   - Convenient for evaluating single tensors
#   - Requires active session
print("Variable value:", my_variable.eval(session=session))  # Output: 4
print("Add result:", add.eval(session=session))              # Output: 9 (5 + 4)
print("Multiply result:", multiply.eval(session=session))    # Output: 32 (8 * 4)

session.close()

# ======================== PART 2: PLACEHOLDERS AND REDUCTION OPERATIONS ========================

# tf.placeholder() - Create input node for feeding data
# DESCRIPTION: Creates a placeholder node that receives data at runtime
# INPUTS:
#   - dtype: tf.int32 (data type for the placeholder)
#   - shape: [2] (shape of the input - 1D array with 2 elements)
#   - name: "input" (optional name for the placeholder)
# OUTPUT: Placeholder tensor
# NOTES:
#   - No values until provided via feed_dict during sess.run()
#   - Shape can be None (accepts any shape) or specific shape
#   - Used for feeding training data, inputs that change each run
a = tf.placeholder(tf.int32, shape=[2], name="input")

# tf.reduce_prod() - Product of elements across dimensions
# DESCRIPTION: Computes the product of all elements in a tensor
# INPUTS:
#   - input_tensor: a (placeholder with shape [2])
#   - name: "prod_b" (optional name for the operation)
# OUTPUT: Tensor representing product of all elements
# NOTES:
#   - For a=[1, 2]: reduce_prod = 1 * 2 = 2
#   - Reduces tensor to scalar (or specified dimensions)
#   - Can specify axis to reduce along specific dimensions
#   - Default: reduces all dimensions to scalar
b = tf.reduce_prod(a, name="prod_b")

# tf.reduce_sum() - Sum of elements across dimensions
# DESCRIPTION: Computes the sum of all elements in a tensor
# INPUTS:
#   - input_tensor: a (placeholder with shape [2])
#   - name: "sum_c" (optional name for the operation)
# OUTPUT: Tensor representing sum of all elements
# NOTES:
#   - For a=[1, 2]: reduce_sum = 1 + 2 = 3
#   - Reduces tensor to scalar (or specified dimensions)
#   - Can specify axis to reduce along specific dimensions
#   - Default: reduces all dimensions to scalar
c = tf.reduce_sum(a, name="sum_c")

# tf.add() - Add two tensors
# DESCRIPTION: Adds two tensors element-wise
# INPUTS:
#   - x: b (product result, scalar: 2)
#   - y: c (sum result, scalar: 3)
#   - name: "add_d" (optional name for the operation)
# OUTPUT: Tensor representing 2 + 3 = 5
# NOTES:
#   - Combines results from reduce_prod and reduce_sum
#   - For a=[1, 2]: d = reduce_prod([1,2]) + reduce_sum([1,2]) = 2 + 3 = 5
d = tf.add(b, c, name="add_d")

# Create new session for second part
session = tf.Session()

# feed_dict - Dictionary mapping placeholders to values
# DESCRIPTION: Provides actual values for placeholders during execution
# INPUTS:
#   - Dictionary: {placeholder: value}
#   - a: np.array([1, 2], dtype=np.int32) - actual values for placeholder 'a'
# OUTPUT: Dictionary ready to use in sess.run()
# NOTES:
#   - Values must match placeholder's dtype and shape
#   - Can provide values for multiple placeholders
#   - Used during sess.run() to feed data to placeholders
input_dictionary = {a: np.array([1, 2], dtype=np.int32)}

# sess.run() with feed_dict - Execute computation with placeholder values
# DESCRIPTION: Runs the computation graph with provided placeholder values
# INPUTS:
#   - fetches: d (tensor to compute - the final result)
#   - feed_dict: input_dictionary (maps placeholder 'a' to [1, 2])
# OUTPUT: Computed value (numpy array or Python scalar)
# NOTES:
#   - Computes: d = reduce_prod([1,2]) + reduce_sum([1,2]) = 2 + 3 = 5
#   - feed_dict provides values for placeholders
#   - Returns the computed value directly
result = session.run(d, feed_dict=input_dictionary)
print("Final result:", result)  # Output: 5

session.close()

# ======================== FUNCTION REFERENCE SUMMARY ========================
"""
TENSORFLOW FUNCTIONS USED:

1. tf.Variable(initial_value, name=None)
   - Creates trainable variable
   - Input: initial_value (scalar, array, or tensor), name (optional)
   - Output: Variable tensor
   - Must be initialized before use

2. tf.add(x, y, name=None)
   - Element-wise addition
   - Input: Two tensors (must be broadcastable)
   - Output: Tensor with sum of elements
   - Example: tf.add(5, 4) = 9

3. tf.multiply(x, y, name=None)
   - Element-wise multiplication
   - Input: Two tensors (must be broadcastable)
   - Output: Tensor with product of elements
   - Example: tf.multiply(8, 4) = 32

4. tf.placeholder(dtype, shape=None, name=None)
   - Creates input node for data
   - Input: dtype (data type), shape (optional), name (optional)
   - Output: Placeholder tensor
   - Values provided via feed_dict

5. tf.reduce_prod(input_tensor, axis=None, name=None)
   - Product of all elements
   - Input: Tensor, axis (optional - which dimensions to reduce)
   - Output: Scalar or reduced tensor
   - Example: reduce_prod([1, 2]) = 2

6. tf.reduce_sum(input_tensor, axis=None, name=None)
   - Sum of all elements
   - Input: Tensor, axis (optional - which dimensions to reduce)
   - Output: Scalar or reduced tensor
   - Example: reduce_sum([1, 2]) = 3

7. tensor.eval(session=None)
   - Evaluate tensor in session
   - Input: session (Session object)
   - Output: Computed value (numpy array or scalar)
   - Equivalent to sess.run(tensor)

8. sess.run(fetches, feed_dict=None)
   - Execute computation graph
   - Input: fetches (tensor or list of tensors), feed_dict (optional)
   - Output: Computed value(s)
   - feed_dict provides values for placeholders
"""
