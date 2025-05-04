from knn import classify_nn
from nb import classify_nb
import os

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

    # Combine predictions through majority voting without Counter
    ensemble_predictions = []
    for p1, p2, p3 in zip(knn1_predictions, knn2_predictions, nb_predictions):
        # Count votes
        yes_votes = [p1, p2, p3].count('yes')
        no_votes = 3 - yes_votes  # total votes minus yes votes

        # Majority wins, ties → 'yes'
        if yes_votes >= no_votes:
            ensemble_predictions.append('yes')
        else:
            ensemble_predictions.append('no')

    return ensemble_predictions
  

def main():
    print("Starting ensemble classification...")
    
    # Get predictions from ensemble classifier
    print("Running ensemble classifier with k1=3, k2=5...")
    predictions = classify_ensemble('data/test/occupancy.csv', 'data/test/pima.csv', 3, 5)
    print(f"Generated {len(predictions)} predictions")
    
    # Create directory if it doesn't exist
    output_dir = 'data/test/occupancy'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'predictions.txt')
    print(f"Saving predictions to {output_file}")
    
    # Save predictions to file
    with open(output_file, "w+") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print("Predictions saved successfully")
    
if __name__ == "__main__":
    main()
