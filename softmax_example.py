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

print("\n=== Key Properties ===")
print("1. All outputs are positive (because e^x > 0 for any x)")
print("2. All outputs sum to 1 (because we divide by the sum)")
print("3. Larger inputs get larger probabilities")
print("4. The function preserves the relative ordering of inputs") 