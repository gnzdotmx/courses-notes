"""
Example 5: Tensor Evaluation Methods

INTENTION:
This script demonstrates different ways to evaluate tensors in TensorFlow 1.x:
1. Using session.as_default() with tensor.eval()
2. Using feed_dict to replace intermediate tensor values

Key Learning Objectives:
- Understanding tensor.eval() method
- Using session.as_default() context manager
- Understanding feed_dict with non-placeholder tensors
- Replacing intermediate tensor values during execution
- Different ways to evaluate tensors

Evaluation Methods:
- sess.run(tensor): Standard method (most common)
- tensor.eval(session=sess): Alternative method (requires session)
- tensor.eval() with session.as_default(): Convenient when using default session
- feed_dict with any tensor: Replace tensor values during execution
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x eager execution

# ======================== EXAMPLE 1: EVALUATING A CONSTANT ========================
# Demonstrates using session.as_default() with tensor.eval()

# tf.constant() - Create constant tensor
# DESCRIPTION: Creates an immutable tensor with fixed value
# INPUTS:
#   - value: 3 (the constant value)
#   - dtype: Automatically inferred as tf.int32
# OUTPUT: Constant tensor with value 3
# NOTES:
#   - Constants are immutable (cannot be changed)
#   - Value is known at graph construction time
a = tf.constant(3)

# Create session to execute operations
session = tf.Session()

# session.as_default() - Set session as default for eval()
# DESCRIPTION: Context manager that sets the session as default
# INPUTS: None (uses the session object)
# OUTPUT: Context manager
# NOTES:
#   - When inside this context, tensor.eval() doesn't need session parameter
#   - Convenient when evaluating multiple tensors
#   - Alternative: tensor.eval(session=session) (explicit session)
#   - Equivalent to: print(a.eval(session=session))
with session.as_default():
    # tensor.eval() - Evaluate tensor in default session
    # DESCRIPTION: Computes and returns the value of a tensor
    # INPUTS: None (uses default session from as_default())
    # OUTPUT: Computed value (numpy array or Python scalar)
    # NOTES:
    #   - Requires active session (either default or explicit)
    #   - Equivalent to sess.run(tensor)
    #   - Convenient for evaluating single tensors
    print("Constant value:", a.eval())  # Output: 3

session.close()

# ======================== EXAMPLE 2: REPLACING TENSOR VALUES WITH FEED_DICT ========================
# Demonstrates using feed_dict to replace intermediate tensor values (not just placeholders)

# tf.add() - Element-wise addition
# DESCRIPTION: Adds two tensors element-wise
# INPUTS:
#   - x: 2 (scalar constant)
#   - y: 3 (scalar constant)
# OUTPUT: Tensor representing 2 + 3 = 5
# NOTES:
#   - Creates tensor a2 with value 5
#   - This is an intermediate tensor in the computation graph
a2 = tf.add(2, 3)  # a2 = 5

# tf.multiply() - Element-wise multiplication
# DESCRIPTION: Multiplies two tensors element-wise
# INPUTS:
#   - x: a2 (tensor with value 5)
#   - y: 4 (scalar constant)
# OUTPUT: Tensor representing 5 * 4 = 20
# NOTES:
#   - Creates tensor b2 that depends on a2
#   - Normal computation: b2 = a2 * 4 = 5 * 4 = 20
#   - But we can replace a2's value using feed_dict
b2 = tf.multiply(a2, 4)  # b2 = a2 * 4 = 5 * 4 = 20 (normally)

# Create session
session = tf.Session()

# feed_dict with non-placeholder tensors
# DESCRIPTION: Replace tensor values during execution (not just placeholders)
# INPUTS:
#   - Dictionary: {tensor: replacement_value}
#   - a2: 15 (replaces a2's computed value of 5 with 15)
# OUTPUT: Dictionary ready to use in sess.run()
# NOTES:
#   - feed_dict can replace ANY tensor value, not just placeholders
#   - Useful for debugging, testing, or overriding computed values
#   - When b2 is computed, a2 will be treated as 15 instead of 5
#   - Result: b2 = 15 * 4 = 60 (instead of 20)
replace_dict = {a2: 15}  # Replace a2 with 15

# sess.run() with feed_dict - Execute with replaced values
# DESCRIPTION: Runs computation with tensor values replaced
# INPUTS:
#   - fetches: b2 (tensor to compute)
#   - feed_dict: replace_dict (replaces a2 with 15)
# OUTPUT: Computed value
# NOTES:
#   - Normally: b2 = a2 * 4 = 5 * 4 = 20
#   - With feed_dict: b2 = 15 * 4 = 60 (a2 replaced with 15)
#   - feed_dict overrides the computed value of a2
result = session.run(b2, feed_dict=replace_dict)
print("Result with replaced a2:", result)  # Output: 60 (not 20)

session.close()

# ======================== KEY CONCEPTS ========================
"""
EVALUATION METHODS:

1. sess.run(tensor):
   - Standard method for evaluating tensors
   - Explicit: Always specify session
   - Can evaluate single tensor or list of tensors
   - Most common and recommended method

2. tensor.eval(session=sess):
   - Alternative method for evaluating tensors
   - Requires explicit session parameter
   - Convenient for single tensor evaluation
   - Equivalent to sess.run(tensor)

3. tensor.eval() with session.as_default():
   - Convenient when evaluating multiple tensors
   - Sets session as default, so eval() doesn't need session parameter
   - Use context manager: with session.as_default():
   - Cleaner code when evaluating many tensors

4. feed_dict WITH NON-PLACEHOLDERS:
   - feed_dict can replace ANY tensor value, not just placeholders
   - Useful for:
     * Debugging: Override computed values to test behavior
     * Testing: Replace intermediate values to see effects
     * Overriding: Force specific values in computation graph
   - Format: {tensor: replacement_value}
   - Overrides the tensor's computed value during execution

5. WHEN TO USE EACH METHOD:
   - sess.run(): Most common, explicit, recommended
   - tensor.eval(): Convenient for single tensor, requires session
   - session.as_default() + eval(): Clean when evaluating many tensors
   - feed_dict with tensors: Debugging, testing, overriding values

6. COMPUTATION GRAPH WITH FEED_DICT:
   - Normal execution: Computes all tensors in order
   - With feed_dict: Replaces specified tensor values before computation
   - Example: a2 normally = 5, but feed_dict replaces it with 15
   - Downstream tensors (b2) use the replaced value
"""
