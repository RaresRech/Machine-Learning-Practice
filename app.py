#
#   MACHINE LEARNING INTRO
#
#   Perceptron assignment: "Classify 2D points based on a line on the y axis"
#   Rares Rechesan 
#   https://github.com/RaresRech
#   19.12.2023
#   
#   This script is responsable for the perceptron class and training. The saved output can be displayed with the plot.py script.
#

# Dependencies
import numpy as np
import random
import json
import os

class Perceptron:
    """
    A simple implementation of a Perceptron, a fundamental building block in neural networks.

    Attributes:
    - training_data_pos (list): List of positive training examples.
    - training_data_neg (list): List of negative training examples.
    - weights (numpy.ndarray): Weight vector for the perceptron.
    - bias (float): Bias term for the perceptron.

    Methods:
    - classify_data(data): Classifies the given data point using the current perceptron parameters.
    - train(learning_rate, iterations): Trains the perceptron using the provided training data.

    """

    def __init__(self, training_data_pos, training_data_neg):
        """
        Initializes the Perceptron with positive and negative training data.

        Args:
        - training_data_pos (list): List of positive training examples.
        - training_data_neg (list): List of negative training examples.
        """
        self.training_data_pos = training_data_pos
        self.training_data_neg = training_data_neg
        self.weights = np.zeros(len(training_data_pos[0]))
        self.bias = 0

    def classify_data(self, data):
        """
        Classifies the given data point using the current perceptron parameters.

        Args:
        - data (numpy.ndarray): Input data point to be classified.

        Returns:
        - int: 1 if the data point is classified as positive, -1 if classified as negative.
        """
        z = np.dot(self.weights, data) + self.bias
        return 1 if z >= 0 else -1

    def train(self, learning_rate, iterations):
        """
        Trains the perceptron using the provided training data.

        Args:
        - learning_rate (float): The learning rate for the training process.
        - iterations (int): Number of training iterations.
        """
        for i in range(iterations):
            example_pos = random.choice(self.training_data_pos)
            example_neg = random.choice(self.training_data_neg)

            # Positive example
            if self.classify_data(example_pos) <= 0:
                self.weights += learning_rate * example_pos
                self.bias += learning_rate

            # Negative example
            if self.classify_data(example_neg) >= 0:
                self.weights -= learning_rate * example_neg
                self.bias -= learning_rate
            
            print("Iteration " + str(i) + ":\n BIAS = " + str(self.bias) + ", WEIGHTS =" + str(self.weights) + "\n")
        print("\n\nNeuron trained.")
        return
    
def classify_and_save(perceptron, test_data, output_file):
    """
    Classifies a set of test data using a trained perceptron and saves the results to a JSON file.

    Args:
    - perceptron (Perceptron): Trained perceptron.
    - test_data (list): List of test data points.
    - output_file (str): Path to the output JSON file.
    """
    results = []

    for coord in test_data:
        result = perceptron.classify_data(np.array(coord))
        results.append({"coordinates": coord, "classification": result})

    with open(output_file, 'w') as json_file:
        json.dump(results, json_file)

    print(f"Results saved to {output_file}\n")


def parse_coordinates(input_string):
    """
    Parses a string containing coordinates into a tuple of floats.

    Args:
    - input_string (str): String containing coordinates in the format "x,y".

    Returns:
    - tuple: Tuple of floats representing the x and y coordinates.
    """
    x_str, y_str = input_string.split(',')
    x = float(x_str)
    y = float(y_str)
    return x, y

# Remove existing classification results file
os.remove("classification_results.json")

# Initialize training data lists
training_pos = []
training_neg = []

# User input for manual or automatic training
training_type = input("Manual training (m) or auto training (a)\n\n")

# Manual training input
if training_type == "m":
    data_count = int(input("Enter number of training data:\n"))

    for i in range(data_count):
        data = input("Positive example #" + str(i) + ":")
        training_pos.append(parse_coordinates(data))
        print("\n")
    
    print('\n')

    for i in range(data_count):
        data = input("Negative example #" + str(i) + ":")
        training_neg.append(parse_coordinates(data))
        print("\n")

    print("\n")

# Automatic training input
else:
    data_count = int(input("Enter number of training data:\n"))
    max_coord = int(input("Maximum coordinates: \n"))
    threshold = int(input("Threshold: \n"))

    training_pos = [(random.uniform(0, 100), random.uniform(threshold, max_coord)) for _ in range(data_count)]
    training_neg = [(random.uniform(0, 100), random.uniform(0, threshold)) for _ in range(data_count)]

# User prompt to start training
choice = input("Neuron is ready to train. Enter to train...")

# Create Perceptron instance and train
neuron = Perceptron(training_pos, training_neg)
neuron.train(1, 100000)

# Generate test coordinates
test_coordinates = [(random.uniform(0, max_coord), random.uniform(0, max_coord)) for _ in range(100)]

# Test and print results
for i in test_coordinates:
    result = neuron.classify_data(i)
    print(f"\n The test coordinate {i} is classified as: {result}")

print("\n\n")

# Save classification results to a JSON file
output_file = "classification_results.json"
classify_and_save(neuron, test_coordinates, output_file)
print('\nTreshold: '+str(threshold))