import numpy as np

# Define the activation function (ReLU in this case)
def relu(x):
    return np.maximum(0, x)

# Inputs (4 features)
inputs = np.array([1.0, 2.0, 3.0, 0.5])

# Weights (3 neurons, each with 4 weights)
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, 0.27, 0.17, 0.87]])

# Biases (1 bias per neuron)
biases = np.array([2.0, 3.0, 0.5])

# Compute the weighted sum
"""
Mathematically:
z_j = sum(w_ij * x_i) + b_j for each neuron j

For neuron 1:
z_1 = (0.2 * 1.0) + (0.8 * 2.0) + (-0.5 * 3.0) + (1.0 * 0.5) + 2.0
    = 0.2 + 1.6 - 1.5 + 0.5 + 2.0
    = 2.8

For neuron 2:
z_2 = (0.5 * 1.0) + (-0.91 * 2.0) + (0.26 * 3.0) + (-0.5 * 0.5) + 3.0
    = 0.5 - 1.82 + 0.78 - 0.25 + 3.0
    = 2.21

For neuron 3:
z_3 = (-0.26 * 1.0) + (0.27 * 2.0) + (0.17 * 3.0) + (0.87 * 0.5) + 0.5
    = -0.26 + 0.54 + 0.51 + 0.435 + 0.5
    = 1.725
"""
weighted_sum = np.dot(weights, inputs) + biases

"""
Mathematically:
y_j = ReLU(z_j)

For neuron 1:
y_1 = ReLU(2.8)
    = 2.8

For neuron 2:
y_2 = ReLU(2.21)
    = 2.21

For neuron 3:
y_3 = ReLU(1.725)
    = 1.725
"""
outputs = relu(weighted_sum)

print("Inputs:\n", inputs)
print("Weights:\n", weights)
print("Biases:\n", biases)
print("Weighted Sum:\n", weighted_sum)
print("Outputs:\n", outputs)
