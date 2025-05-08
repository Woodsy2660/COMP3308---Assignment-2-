import sys
import pandas as pd

def extract_fold(fold_file, fold_number):
    """
    Extracts a specific fold as test data and uses the rest as training data.
    
    Args:
        fold_file: Path to the fold file (e.g., 'data/stratification/pima-folds.csv')
        fold_number: The fold number to use as test (1-10)
    
    Returns:
        train_file: Path to the generated training CSV
        test_file: Path to the generated test CSV
    """
    # Read the fold file content
    with open(fold_file, 'r') as f:
        content = f.read()
    
    # Split by 'fold' markers
    parts = content.split('\nfold')
    
    # First part will be "fold1" so we need to handle it separately
    parts[0] = parts[0].replace('fold1\n', '')
    
    # Ensure fold_number is within valid range
    if fold_number < 1 or fold_number > len(parts):
        print(f"Invalid fold number. Must be between 1 and {len(parts)}.")
        sys.exit(1)
        
    # Extract test fold (fold_number) and combine remaining folds for training
    test_data = parts[fold_number - 1].strip()
    
    # Create a list of all folds except the test fold
    train_parts = [parts[i].strip() for i in range(len(parts)) if i != fold_number - 1 and parts[i].strip()]
    train_data = '\n'.join(train_parts)
    
    # Get the base name from the fold file
    base_name = fold_file.split('/')[-1].split('-folds')[0]
    
    # Write to files
    train_file = f"{base_name}_fold{fold_number}_train.csv"
    test_file = f"{base_name}_fold{fold_number}_test.csv"
    
    with open(train_file, 'w') as f:
        f.write(train_data)
    
    with open(test_file, 'w') as f:
        f.write(test_data)
        
    print(f"Created {train_file} and {test_file}")
    return train_file, test_file

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_fold.py <fold_file> <fold_number>")
        sys.exit(1)
        
    fold_file = sys.argv[1]
    fold_number = int(sys.argv[2])
    
    extract_fold(fold_file, fold_number) 