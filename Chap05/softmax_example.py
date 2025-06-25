import numpy as np

def softmax_explained(z):
    """
    Step-by-step explanation of softmax function
    z: input vector of arbitrary values
    """
    print(f"Input vector z: {z}")
    
    # Step 1: Compute exponentials
    exp_z = np.exp(z)
    print(f"Step 1 - Exponentials e^z: {exp_z}")
    
    # Step 2: Sum all exponentials
    sum_exp = np.sum(exp_z)
    print(f"Step 2 - Sum of exponentials: {sum_exp}")
    
    # Step 3: Divide each exponential by the sum
    softmax_result = exp_z / sum_exp
    print(f"Step 3 - Softmax result: {softmax_result}")
    print(f"Step 3 - Sum of probabilities: {np.sum(softmax_result):.6f}")
    
    return softmax_result

# Example 1: Simple case
print("=== Example 1: Simple case ===")
z1 = np.array([1, 2, 3])
result1 = softmax_explained(z1)

print("\n=== Example 2: With negative numbers ===")
z2 = np.array([-1, 0, 1])
result2 = softmax_explained(z2)

print("\n=== Example 3: Large numbers (showing numerical stability) ===")
z3 = np.array([100, 101, 102])
result3 = softmax_explained(z3)

# === Key Properties ===
# 1. All outputs are positive (because e^x > 0 for any x)
# 2. All outputs sum to 1 (because we divide by the sum)
# 3. Larger inputs get larger probabilities
# 4. The function preserves the relative ordering of inputs

# Example with shape (3, 4) for simplicity
# - 3 classes, with 4 input samples
exp_model_out = np.array([
    [1.0, 2.0, 3.0, 4.0],  # Class 0 exponentials
    [0.5, 1.0, 1.5, 2.0],  # Class 1 exponentials  
    [0.2, 0.4, 0.6, 0.8]   # Class 2 exponentials
])

# axis=0 means sum DOWN the columns (vertically)
sum_exp = np.sum(exp_model_out, axis=0)
print(sum_exp)  # [1.7, 3.4, 5.1, 6.8]

softmax_out = exp_model_out / sum_exp

print(softmax_out)
print(np.sum(softmax_out, axis=0))
