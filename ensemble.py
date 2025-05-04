from knn import classify_nn
from nb import classify_nb
from collections import Counter
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
  
  
def main():
    print("Starting ensemble classification...")
    
    # Get predictions from ensemble classifier
    print("Running ensemble classifier with k1=3, k2=5...")
    predictions = classify_ensemble('data/test/occupancy.csv', 'data/test/pima.csv', 3, 5)
    print(f"Generated {len(predictions)} predictions")
    
    # Create directory if it doesn't exist
    os.makedirs('data/test/occupancy', exist_ok=True)
    print(f"Saving predictions to data/test/occupancy/predicitons.txt")
    
    # Save predictions to file
    with open('data/test/occupancy/predicitons.txt', "w+") as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")
    print("Predictions saved successfully")
    
if __name__ == "__main__":
  main()
    