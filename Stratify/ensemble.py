# ensemble.py

from knn import classify_nn
from nb import classify_nb
from collections import Counter

def classify_ensemble(training_file, testing_file, k1, k2):
    """
    Ensemble classifier combining KNN (with k1 and k2) and Naive Bayes.

    Args:
      training_file: path to CSV with features + label
      testing_file:  path to CSV with features only
      k1:            first k value for KNN
      k2:            second k value for KNN

    Returns:
      List[str] of predicted labels ("yes"/"no") in order.
    """
    # Run all three classifiers
    knn1_predictions = classify_nn(training_file, testing_file, k1)
    knn2_predictions = classify_nn(training_file, testing_file, k2)
    nb_predictions = classify_nb(training_file, testing_file)

    # Combine predictions through majority voting
    ensemble_predictions = []
    for i in range(len(knn1_predictions)):
        votes = [knn1_predictions[i], knn2_predictions[i], nb_predictions[i]]
        vote_counts = Counter(votes)
        
        # Majority wins, ties → "yes"
        if vote_counts['yes'] >= vote_counts['no']:
            ensemble_predictions.append('yes')
        else:
            ensemble_predictions.append('no')

    return ensemble_predictions 