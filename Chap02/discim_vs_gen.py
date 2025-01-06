import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data points (x, y)
x = np.array([1, 2, 3]).reshape(-1, 1)  # Input as column vector
y = np.array([2, 2.5, 3.5]).reshape(-1, 1)  # Output as column vector

# Discriminative Model: Predict y given x
model_disc = LinearRegression()
model_disc.fit(x, y)
phi_0_disc, phi_1_disc = model_disc.intercept_[0], model_disc.coef_[0, 0]

# Generative Model: Predict x given y
model_gen = LinearRegression()
model_gen.fit(y, x)
phi_0_gen, phi_1_gen = model_gen.intercept_[0], model_gen.coef_[0, 0]

# Inverse Generative Function
phi_1_gen_inv = 1 / phi_1_gen
phi_0_gen_inv = -phi_0_gen / phi_1_gen

# Display results
print("Discriminative Model: y = {:.2f} + {:.2f}x".format(phi_0_disc, phi_1_disc))
print("Generative Model: x = {:.2f} + {:.2f}y".format(phi_0_gen, phi_1_gen))
print("Inverse Generative Model: y = {:.2f} + {:.2f}x".format(phi_0_gen_inv, phi_1_gen_inv))

# Plot data and models
plt.scatter(x, y, color='red', label='Data Points')

# Discriminative model line
plt.plot(x, phi_0_disc + phi_1_disc * x, label='Discriminative Model')

# Inverse generative model line
plt.plot(x, phi_0_gen_inv + phi_1_gen_inv * x, label='Inverse Generative Model', linestyle='--')

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Comparison of Discriminative and Generative Models")
plt.show()
