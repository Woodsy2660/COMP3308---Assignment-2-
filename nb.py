# nb.py

from utilities import load_training_data, load_testing_data
import math
from collections import defaultdict, Counter

def classify_nb(training_file, testing_file):
    """
    Gaussian Naive Bayes classifier.

    Args:
      training_file: path to CSV with features + label
      testing_file:  path to CSV with features only

    Returns:
      List[str] of predicted labels ("yes"/"no") in order.
    """
    training_features, training_labels = load_training_data(training_file)
    testing_features                   = load_testing_data(testing_file)
    
    # If there's no training data, return empty predictions
    if not training_features:
        return []
        
    # If there's no testing data, return empty predictions
    if not testing_features:
        return []

    number_of_features = len(training_features[0])
    
    # Ensure testing features have same dimensions as training features
    testing_features_normalized = []
    for feature_vector in testing_features:
        # Truncate or pad feature vector to match training dimensions
        if len(feature_vector) > number_of_features:
            testing_features_normalized.append(feature_vector[:number_of_features])
        elif len(feature_vector) < number_of_features:
            # Pad with zeros if needed
            padded = feature_vector + [0.0] * (number_of_features - len(feature_vector))
            testing_features_normalized.append(padded)
        else:
            testing_features_normalized.append(feature_vector)
    
    testing_features = testing_features_normalized

    # group feature values by class label
    data_by_class = defaultdict(lambda: [[] for _ in range(number_of_features)])
    class_counts  = Counter(training_labels)
    
    # Ensure we have both 'yes' and 'no' classes
    if 'yes' not in class_counts:
        class_counts['yes'] = 0
    if 'no' not in class_counts:
        class_counts['no'] = 0
        
    for feature_vector, label in zip(training_features, training_labels):
        for index, value in enumerate(feature_vector):
            data_by_class[label][index].append(value)

    total_examples = sum(class_counts.values())
    if total_examples == 0:
        # No examples to train on
        return ['yes'] * len(testing_features)  # Default prediction

    # compute prior probability for each class
    class_priors = {
        label: max(count, 1) / max(total_examples, 1)  # Avoid division by zero
        for label, count in class_counts.items()
    }

    # compute mean & variance for each feature, per class
    feature_stats_by_class = {}
    for label, feature_lists in data_by_class.items():
        statistics = []
        for values in feature_lists:
            if not values:  # If no values for this feature in this class
                average_value = 0.0
                variance_value = 1e-9
            else:
                average_value = sum(values) / len(values)
                variance_value = sum((v - average_value) ** 2 for v in values) / len(values)
                if variance_value == 0:
                    variance_value = 1e-9
            statistics.append((average_value, variance_value))
        feature_stats_by_class[label] = statistics
        
    # Ensure both classes have feature statistics
    for label in ['yes', 'no']:
        if label not in feature_stats_by_class:
            # Create default statistics if class is missing
            feature_stats_by_class[label] = [(0.0, 1e-9) for _ in range(number_of_features)]

    def gaussian_pdf(x, mean_value, variance_value):
        coefficient   = 1 / math.sqrt(2 * math.pi * variance_value)
        exponent_term = math.exp(-((x - mean_value) ** 2) / (2 * variance_value))
        return coefficient * exponent_term

    predicted_labels = []
    for test_example in testing_features:
        log_prob_by_class = {}
        for label in ['yes', 'no']:  # Always consider both classes
            # start with log prior
            log_probability = math.log(class_priors[label])
            # add log likelihood of each feature
            for index, feature_value in enumerate(test_example):
                mean_value, variance_value = feature_stats_by_class[label][index]
                likelihood = gaussian_pdf(feature_value, mean_value, variance_value)
                
                if likelihood > 0:
                    log_probability += math.log(likelihood)
            log_prob_by_class[label] = log_probability

        # choose label with higher log-probability, tie → "yes"
        if log_prob_by_class['yes'] >= log_prob_by_class['no']:
            predicted_labels.append('yes')
        else:
            predicted_labels.append('no')

    return predicted_labels 