# nb.py

from utils import load_training_data, load_testing_data
import math
from collections import defaultdict, Counter

def classify_nb(training_file, testing_file):
    """
    Gaussian Naive Bayes classifier.

    Args:
      training_file: path to CSV with features + label
      testing_file:  path to CSV with features only

    Returns:
      List[str] of predicted labels (“yes”/“no”) in order.
    """
    training_features, training_labels = load_training_data(training_file)
    testing_features                   = load_testing_data(testing_file)

    number_of_features = len(training_features[0])

    # group feature values by class label
    data_by_class = defaultdict(lambda: [[] for _ in range(number_of_features)])
    class_counts  = Counter(training_labels)
    for feature_vector, label in zip(training_features, training_labels):
        for index, value in enumerate(feature_vector):
            data_by_class[label][index].append(value)

    total_examples = sum(class_counts.values())

    # compute prior probability for each class
    class_priors = {
        label: count / total_examples
        for label, count in class_counts.items()
    }

    # compute mean & variance for each feature, per class
    feature_stats_by_class = {}
    for label, feature_lists in data_by_class.items():
        statistics = []
        for values in feature_lists:
            average_value     = sum(values) / len(values)
            variance_value    = sum((v - average_value) ** 2 for v in values) / len(values)
            if variance_value == 0:
                variance_value = 1e-9
            statistics.append((average_value, variance_value))
        feature_stats_by_class[label] = statistics

    def gaussian_pdf(x, mean_value, variance_value):
        coefficient   = 1 / math.sqrt(2 * math.pi * variance_value)
        exponent_term = math.exp(-((x - mean_value) ** 2) / (2 * variance_value))
        return coefficient * exponent_term

    predicted_labels = []
    for test_example in testing_features:
        log_prob_by_class = {}
        for label in class_counts:
            # start with log prior
            log_probability = math.log(class_priors[label])
            # add log likelihood of each feature
            for index, feature_value in enumerate(test_example):
                mean_value, variance_value = feature_stats_by_class[label][index]
                likelihood = gaussian_pdf(feature_value, mean_value, variance_value)
                # Add a small epsilon to avoid math domain error
                likelihood = max(likelihood, 1e-10)
                log_probability += math.log(likelihood)
            log_prob_by_class[label] = log_probability

        # choose label with higher log-probability, tie → "yes"
        if log_prob_by_class['yes'] >= log_prob_by_class['no']:
            predicted_labels.append('yes')
        else:
            predicted_labels.append('no')

    return predicted_labels
