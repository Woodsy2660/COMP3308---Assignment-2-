import sys
import os
import subprocess
from extract_fold import extract_fold

def run_knn_test(train_file, test_file, k):
    """Run k-NN test with the specified k value."""
    print(f"\nRunning k-NN (k={k}) test...")
    cmd = f"python main_knn.py {train_file} {test_file} {k}"
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Save results to file
    output_file = f"results_knn_k{k}_fold{fold_num}.txt"
    with open(output_file, 'w') as f:
        f.write(result.stdout)
    
    print(f"Results saved to {output_file}")
    return result.stdout

def run_nb_test(train_file, test_file):
    """Run Naive Bayes test."""
    print(f"\nRunning Naive Bayes test...")
    cmd = f"python main_nb.py {train_file} {test_file}"
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Save results to file
    output_file = f"results_nb_fold{fold_num}.txt"
    with open(output_file, 'w') as f:
        f.write(result.stdout)
    
    print(f"Results saved to {output_file}")
    return result.stdout

def run_ensemble_test(train_file, test_file, k1, k2):
    """Run Ensemble test with specified k values."""
    print(f"\nRunning Ensemble (k1={k1}, k2={k2}) test...")
    cmd = f"python ensemble.py {train_file} {test_file} {k1} {k2}"
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Save results to file
    output_file = f"results_ensemble_k{k1}_{k2}_fold{fold_num}.txt"
    with open(output_file, 'w') as f:
        f.write(result.stdout)
    
    print(f"Results saved to {output_file}")
    return result.stdout

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_fold_tests.py <fold_file> <fold_number>")
        print("Example: python run_fold_tests.py data/stratification/pima-folds.csv 1")
        sys.exit(1)
    
    fold_file = sys.argv[1]
    fold_num = int(sys.argv[2])
    
    # Extract train/test data for this fold
    train_file, test_file = extract_fold(fold_file, fold_num)
    
    # Run all tests
    # 1. k-NN with k=1
    knn_k1_results = run_knn_test(train_file, test_file, 1)
    
    # 2. k-NN with k=7
    knn_k7_results = run_knn_test(train_file, test_file, 7)
    
    # 3. Naive Bayes
    nb_results = run_nb_test(train_file, test_file)
    
    # 4. Ensemble with k1=1, k2=7
    ensemble_results = run_ensemble_test(train_file, test_file, 1, 7)
    
    print("\nAll tests completed. Results saved to separate files.") 