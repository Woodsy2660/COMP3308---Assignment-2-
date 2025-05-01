# utils.py

import csv
import math

def load_training_data(filename):
    """
    Read a CSV (no header) where the last column is the class label.
    Returns:
      training_features: list[list[float]]
      training_labels:   list[str]      # "yes" or "no"
    """
    training_features = []
    training_labels   = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            feature_values = [float(value) for value in row[:-1]]
            label          = row[-1]
            training_features.append(feature_values)
            training_labels.append(label)
    return training_features, training_labels

def load_testing_data(filename):
    """
    Read a CSV (no header) with only feature columns.
    Returns:
      testing_features: list[list[float]]
    """
    testing_features = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            feature_values = [float(value) for value in row]
            testing_features.append(feature_values)
    return testing_features

def calculate_euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two equal-length vectors.
    """
    squared_differences = [(x - y) ** 2 for x, y in zip(point1, point2)]
    distance            = math.sqrt(sum(squared_differences))
    return distance 