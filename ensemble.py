# program.py

import sys
from collections import Counter

# import your scratch implementations
from knn import classify_nn
from nb  import classify_nb


def classify_ensemble(training_file, testing_file, k1, k2):
    """
    Ensemble classifier combining two KNNs and NB.

    Args:
      training_file:  path to CSV with features + label
      testing_file:   path to CSV with features only
      k1:             first k for KNN
      k2:             second k for KNN

    Returns:
      List[str] of “yes”/“no” predictions, in order.
    """
    # 1NN, 2nd KNN, and NB predictions
    preds1 = classify_nn(training_file, testing_file, k1)
    preds2 = classify_nn(training_file, testing_file, k2)
    preds3 = classify_nb(training_file, testing_file)

    # majority-vote (ties → “yes”)
    ensemble = []
    for a, b, c in zip(preds1, preds2, preds3):
        votes = Counter((a, b, c))
        ensemble.append("yes" if votes["yes"] >= votes["no"] else "no")

    return ensemble


# alias for ED test scripts that import classify_ens
classify_ens = classify_ensemble


def main():
    if len(sys.argv) != 5:
        print("Usage: python program.py <training_csv> <testing_csv> <k1> <k2>")
        sys.exit(1)

    training_csv = sys.argv[1]
    testing_csv  = sys.argv[2]
    k1           = int(sys.argv[3])
    k2           = int(sys.argv[4])

    for label in classify_ensemble(training_csv, testing_csv, k1, k2):
        print(label)


if __name__ == "__main__":
    main()
