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
        # First check if file uses tabs or commas as delimiter
        first_line = csvfile.readline().strip()
        delimiter = '\t' if '\t' in first_line else ','
        
        # Reset file pointer to beginning
        csvfile.seek(0)
        
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            # Skip empty rows
            if not row:
                continue
                
            # Skip rows with insufficient data
            if len(row) < 2:  # Need at least one feature and a label
                continue
                
            # Skip rows with non-numeric values
            try:
                feature_values = [float(value) for value in row[:-1]]
                label = row[-1]
                training_features.append(feature_values)
                training_labels.append(label)
            except ValueError:
                # Skip this row (likely a header)
                continue
    return training_features, training_labels

def load_testing_data(filename):
    """
    Read a CSV (no header) with only feature columns.
    Returns:
      testing_features: list[list[float]]
    """
    testing_features = []
    with open(filename, newline='') as csvfile:
        # First check if file uses tabs or commas as delimiter
        first_line = csvfile.readline().strip()
        delimiter = '\t' if '\t' in first_line else ','
        
        # Reset file pointer to beginning
        csvfile.seek(0)
        
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            # Skip empty rows
            if not row:
                continue
                
            try:
                # Note: For testing data, the last column might be a label that we need to exclude
                if row[-1] in ['yes', 'no']:
                    feature_values = [float(value) for value in row[:-1]]
                else:
                    feature_values = [float(value) for value in row]
                testing_features.append(feature_values)
            except ValueError:
                # Skip header rows or rows with non-numeric values
                continue
    return testing_features

def calculate_euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two equal-length vectors.
    """
    squared_differences = [(x - y) ** 2 for x, y in zip(point1, point2)]
    distance            = math.sqrt(sum(squared_differences))
    return distance 