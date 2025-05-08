import os
import sys
import re
import numpy as np

def load_folds(folds_file):
    """
    Load folds from a fold file created by 10_fold_stratification.py
    
    Returns:
        List of 10 folds, where each fold is a list of data rows
    """
    folds = []
    current_fold = []
    
    with open(folds_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check if this is a fold header
            if line.startswith('fold'):
                if current_fold:  # If we have data from a previous fold
                    folds.append(current_fold)
                    current_fold = []
                continue
                
            # Skip empty lines
            if not line:
                continue
                
            # Add data row to current fold
            current_fold.append(line.split(','))
    
    # Add the last fold if it has data
    if current_fold:
        folds.append(current_fold)
    
    # Debug information
    print(f"Loaded {len(folds)} folds from {folds_file}")
    for i, fold in enumerate(folds):
        print(f"  Fold {i+1}: {len(fold)} instances")
        
    return folds

def prepare_train_test_split(folds, test_fold_idx):
    """
    Prepare training and testing data for a specific fold
    
    Args:
        folds: List of all folds
        test_fold_idx: Index of the fold to use as test data (0-9)
        
    Returns:
        (train_X, train_y, test_X, test_y) where:
            train_X: List of training instances (features only)
            train_y: List of training labels
            test_X: List of testing instances (features only)
            test_y: List of testing labels
    """
    test_data = folds[test_fold_idx]
    
    # Combine all other folds for training data
    train_data = []
    for i, fold in enumerate(folds):
        if i != test_fold_idx:
            train_data.extend(fold)
    
    # Split into features (X) and labels (y)
    train_X = [instance[:-1] for instance in train_data]
    train_y = [instance[-1] for instance in train_data]
    
    test_X = [instance[:-1] for instance in test_data]
    test_y = [instance[-1] for instance in test_data]
    
    return train_X, train_y, test_X, test_y

def run_classifier(classifier_name, train_X, train_y, test_X, args=None):
    """
    Run a specific classifier and get predictions
    
    Args:
        classifier_name: Name of classifier ('knn', 'bayes', or 'ensemble')
        train_X: Training features
        train_y: Training labels
        test_X: Testing features
        args: Additional arguments for the classifier
        
    Returns:
        List of predicted labels for test data
    """
    if classifier_name == 'knn':
        from knn import classify_nn
        k = args.get('k', 1)
        
        # Create temporary train file
        temp_train_file = 'temp_train.csv'
        with open(temp_train_file, 'w') as f:
            for features, label in zip(train_X, train_y):
                f.write(','.join(features) + ',' + label + '\n')
        
        # Create temporary test file
        temp_test_file = 'temp_test.csv'
        with open(temp_test_file, 'w') as f:
            for features in test_X:
                f.write(','.join(features) + '\n')
        
        # Run KNN classifier
        predictions = classify_nn(temp_train_file, temp_test_file, k)
        
        # Clean up temp files
        try:
            os.remove(temp_train_file)
            os.remove(temp_test_file)
        except:
            pass
            
        return predictions
    
    elif classifier_name == 'bayes':
        from nb import classify_nb
        
        # Create temporary train file
        temp_train_file = 'temp_train.csv'
        with open(temp_train_file, 'w') as f:
            for features, label in zip(train_X, train_y):
                f.write(','.join(features) + ',' + label + '\n')
        
        # Create temporary test file
        temp_test_file = 'temp_test.csv'
        with open(temp_test_file, 'w') as f:
            for features in test_X:
                f.write(','.join(features) + '\n')
        
        # Run Naive Bayes classifier
        predictions = classify_nb(temp_train_file, temp_test_file)
        
        # Clean up temp files
        try:
            os.remove(temp_train_file)
            os.remove(temp_test_file)
        except:
            pass
            
        return predictions
    
    elif classifier_name == 'ensemble':
        from ensemble import classify_ensemble
        
        # Create temporary train file
        temp_train_file = 'temp_train.csv'
        with open(temp_train_file, 'w') as f:
            for features, label in zip(train_X, train_y):
                f.write(','.join(features) + ',' + label + '\n')
        
        # Create temporary test file
        temp_test_file = 'temp_test.csv'
        with open(temp_test_file, 'w') as f:
            for features in test_X:
                f.write(','.join(features) + '\n')
        
        # Run Ensemble classifier with k1=3 and k2=7
        predictions = classify_ensemble(temp_train_file, temp_test_file, 3, 7)
        
        # Clean up temp files
        try:
            os.remove(temp_train_file)
            os.remove(temp_test_file)
        except:
            pass
            
        return predictions
    
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

def evaluate_predictions(predictions, actual):
    """
    Evaluate classifier predictions against actual labels
    
    Args:
        predictions: List of predicted labels
        actual: List of actual labels
        
    Returns:
        Dictionary with accuracy and confusion matrix metrics
    """
    if len(predictions) != len(actual):
        raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match number of actual labels ({len(actual)})")
    
    correct = sum(1 for p, a in zip(predictions, actual) if p == a)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate confusion matrix (assuming binary classification with "yes"/"no")
    true_pos = sum(1 for p, a in zip(predictions, actual) if p == "yes" and a == "yes")
    false_pos = sum(1 for p, a in zip(predictions, actual) if p == "yes" and a == "no")
    true_neg = sum(1 for p, a in zip(predictions, actual) if p == "no" and a == "no")
    false_neg = sum(1 for p, a in zip(predictions, actual) if p == "no" and a == "yes")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "true_pos": true_pos,
        "false_pos": false_pos,
        "true_neg": true_neg,
        "false_neg": false_neg
    }

def run_cross_validation(folds_file, classifier_name, classifier_args=None):
    """
    Run full 10-fold cross-validation for a classifier
    
    Args:
        folds_file: Path to the file containing the 10 folds
        classifier_name: Name of classifier to use
        classifier_args: Additional arguments for the classifier
        
    Returns:
        Dictionary with average metrics across all folds
    """
    # Load all folds
    folds = load_folds(folds_file)
    
    if len(folds) != 10:
        print(f"Warning: Expected 10 folds, but got {len(folds)}")
    
    # Initialize results storage
    all_metrics = []
    
    # Run evaluation for each fold
    for fold_idx in range(len(folds)):
        # Prepare data split for this fold
        train_X, train_y, test_X, test_y = prepare_train_test_split(folds, fold_idx)
        
        # Run classifier on this split
        predictions = run_classifier(classifier_name, train_X, train_y, test_X, classifier_args)
        
        # Evaluate predictions
        metrics = evaluate_predictions(predictions, test_y)
        metrics['fold'] = fold_idx + 1  # Store fold number (1-based)
        
        # Store metrics for this fold
        all_metrics.append(metrics)
        
        print(f"Fold {fold_idx+1}: Accuracy = {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    # Calculate average metrics
    avg_metrics = {
        "accuracy": np.mean([m['accuracy'] for m in all_metrics]),
        "true_pos": np.mean([m['true_pos'] for m in all_metrics]),
        "false_pos": np.mean([m['false_pos'] for m in all_metrics]),
        "true_neg": np.mean([m['true_neg'] for m in all_metrics]),
        "false_neg": np.mean([m['false_neg'] for m in all_metrics])
    }
    
    # Calculate standard deviation for accuracy
    avg_metrics["accuracy_std"] = np.std([m['accuracy'] for m in all_metrics])
    
    return avg_metrics, all_metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cross-validation evaluation')
    parser.add_argument('folds_file', help='Path to the file containing 10 folds')
    parser.add_argument('--classifier', choices=['knn', 'bayes', 'ensemble'], default='knn',
                        help='Classifier to use (default: knn)')
    parser.add_argument('--k', type=int, default=1, help='k value for KNN classifier (default: 1)')
    
    args = parser.parse_args()
    
    classifier_args = {'k': args.k} if args.classifier == 'knn' else {}
    
    print(f"Running 10-fold cross-validation with {args.classifier} classifier")
    if args.classifier == 'knn':
        print(f"Using k={args.k}")
    
    avg_metrics, fold_metrics = run_cross_validation(
        args.folds_file, 
        args.classifier, 
        classifier_args
    )
    
    print("\nCross-validation complete!")
    print(f"Average accuracy: {avg_metrics['accuracy']:.4f} (std: {avg_metrics['accuracy_std']:.4f})")
    print(f"Average confusion matrix:")
    print(f"  True Positive: {avg_metrics['true_pos']:.2f}")
    print(f"  False Positive: {avg_metrics['false_pos']:.2f}")
    print(f"  True Negative: {avg_metrics['true_neg']:.2f}")
    print(f"  False Negative: {avg_metrics['false_neg']:.2f}")

if __name__ == "__main__":
    main() 