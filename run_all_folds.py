import subprocess
import sys

def run_all_folds_for_dataset(dataset_name):
    """Run tests on all 10 folds for a dataset."""
    print(f"\n=== Running all tests for {dataset_name} dataset ===")
    
    fold_file = f"data/stratification/{dataset_name}-folds.csv"
    
    for fold_num in range(1, 11):  # 10 folds
        print(f"\n--- Processing fold {fold_num} ---")
        cmd = f"python run_fold_tests.py {fold_file} {fold_num}"
        
        try:
            result = subprocess.run(cmd, shell=True, text=True)
            if result.returncode != 0:
                print(f"Error running tests for fold {fold_num}.")
        except Exception as e:
            print(f"Exception occurred while processing fold {fold_num}: {e}")
    
    print(f"\n=== Completed all tests for {dataset_name} dataset ===")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run for specific dataset
        dataset_name = sys.argv[1].lower()
        if dataset_name in ['pima', 'occupancy']:
            run_all_folds_for_dataset(dataset_name)
        else:
            print("Invalid dataset name. Please use 'pima' or 'occupancy'.")
    else:
        # Run for both datasets
        run_all_folds_for_dataset('pima')
        run_all_folds_for_dataset('occupancy') 