# knn.py

from utilities import load_training_data, load_testing_data, calculate_euclidean_distance
from collections import Counter

def classify_nn(training_file, testing_file, k):
    """
    k-Nearest Neighbour classifier.

    Args:
      training_file: path to CSV with features + label
      testing_file:  path to CSV with features only
      k:             number of neighbours

    Returns:
      List[str] of predicted labels ("yes"/"no") in order.
    """
    training_features, training_labels = load_training_data(training_file)
    testing_features                   = load_testing_data(testing_file)

    predicted_labels = []
    for test_example in testing_features:
        # compute distance to every training example
        distance_label_pairs = []
        for train_example, train_label in zip(training_features, training_labels):
            distance_label_pairs.append((
                calculate_euclidean_distance(test_example, train_example),
                train_label
            ))

        # pick the k smallest distances
        k_nearest = sorted(distance_label_pairs, key=lambda pair: pair[0])[:k]

        # vote among the k neighbours
        vote_counts = Counter(label for _, label in k_nearest)

        # tie → "yes"
        if vote_counts['yes'] >= vote_counts['no']:
            predicted_labels.append('yes')
        else:
            predicted_labels.append('no')

    return predicted_labels 