# # Simulate a set of 4 inputs acting on 3 perceptrons.
# layer_outputs = list()
# for neuron_weights, bias in zip(weights, biases):
# 	output = 0
# 	for layer_input, weight in zip(inputs, neuron_weights):
# 		output += layer_input * weight
# 	output += bias
# 	layer_outputs.append(output)

# print("3 Perceptron Layer Output: ", layer_outputs)

import numpy as np

inputs  = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
		   [0.5, -0.91, 0.26, -0.5],
		   [-0.26, -0.27, 0.17, 0.87]]
biases  = [2, 3, 0.5]
output = np.dot(weights, inputs) + biases
print("3 Perceptron Layer Output: ", output)