#
#   MACHINE LEARNING INTRO
#
#   Perceptron assignment: "Classify 2D points based on a line on the y axis"
#   Rares Rechesan 
#   https://github.com/RaresRech
#   19.12.2023
#   
#   This script is responsable for plotting the json data generated in the app.py script
#

# Dependencies
import json
import matplotlib.pyplot as plt

def plot_classified_points(json_file):
    """
    Plots points from a JSON file with classification results.

    Args:
    - json_file (str): Path to the JSON file containing classification results.
    """
    with open(json_file, 'r') as file:
        results = json.load(file)

    for result in results:
        coordinates = result["coordinates"]
        classification = result["classification"]
        color = 'red' if classification == 1 else 'blue'
        marker = '*' if classification == 1 else 'x'
        plt.scatter(*coordinates, color=color, marker=marker, s=200)

    plt.show()

if __name__ == "__main__":
    # Path to the JSON file with classification results
    json_file_path = "classification_results.json"

    # Plot classified points
    plot_classified_points(json_file_path)
