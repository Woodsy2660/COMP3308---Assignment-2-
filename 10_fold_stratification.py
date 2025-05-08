# create_pima_folds.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import csv


def get_stratified_fold_indices(y, n_splits=10, shuffle=True, seed=42):
    """
    Partition indices 0…len(y)-1 into n_splits folds so that
    each fold has (approximately) the same class distribution as y.
    
    Args:
        y: Target labels
        n_splits: Number of folds
        shuffle: Whether to shuffle before splitting
        seed: Random seed
        
    Returns:
        List of arrays containing indices for each fold
    """
    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    
    # Create array of indices
    indices = np.arange(len(y))
    
    # Generate folds that maintain class distribution
    folds = []
    for _, test_idx in skf.split(indices, y):
        folds.append(test_idx)
        
    # Print fold statistics if needed
    for i, fold_idx in enumerate(folds):
        fold_y = [y[idx] for idx in fold_idx]
        yes_count = sum(1 for label in fold_y if label == 'yes')
        no_count = sum(1 for label in fold_y if label == 'no')
        total = len(fold_idx)
        print(f"Fold {i+1}: Total={total}, Yes={yes_count} ({yes_count/total*100:.1f}%), No={no_count} ({no_count/total*100:.1f}%)")
    
    return folds


def write_folds(data, folds, output_path):
    """
    Write folds to a single CSV in the form:
      fold1
      <csv rows>

      fold2
      <csv rows>

      …
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, "w", newline='') as out:
        for i, fold_indices in enumerate(folds, start=1):
            # Write fold header
            out.write(f"fold{i}\n")
            
            # Write rows for this fold
            for idx in fold_indices:
                out.write(','.join(data[idx]) + '\n')
            
            # Add blank line between folds
            out.write("\n")
    
    print(f"Wrote stratified folds to {output_path}")


def read_dataset(input_file):
    """
    Read dataset with flexible delimiter detection.
    
    Args:
        input_file: Path to input file
        
    Returns:
        Tuple of (data, labels) where data is a list of rows and labels is a list
    """
    data = []
    labels = []
    
    # Try to determine the delimiter by examining the first few lines
    with open(input_file, 'r') as f:
        first_line = f.readline().strip()
        
    # Check possible delimiters
    for delimiter in [',', '\t', ' ']:
        if delimiter in first_line:
            print(f"Detected delimiter: '{delimiter}'")
            break
    
    # Read the file
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split the line using the detected delimiter
            parts = line.split(delimiter)
            # Clean parts by removing empty strings
            parts = [p.strip() for p in parts if p.strip()]
            
            if not parts:
                continue
                
            # The last part is the label
            label = parts[-1].strip()
            # The rest are features
            features = parts[:-1]
            
            data.append(parts)  # Keep the whole row
            labels.append(label)
    
    return data, labels


def generate_folds(input_file, output_file, n_splits=10):
    """
    Generate stratified folds from input dataset.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file with folds
        n_splits: Number of folds
    """
    print(f"Generating {n_splits} stratified folds from {input_file}...")
    
    # Load the dataset
    try:
        data, labels = read_dataset(input_file)
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return
    
    # Get class distribution
    yes_count = sum(1 for label in labels if label == 'yes')
    no_count = sum(1 for label in labels if label == 'no')
    total = len(labels)
    
    print(f"Dataset statistics: Total={total}, Yes={yes_count} ({yes_count/total*100:.1f}%), No={no_count} ({no_count/total*100:.1f}%)")
    
    # Generate stratified folds
    folds = get_stratified_fold_indices(labels, n_splits=n_splits, shuffle=True, seed=42)
    
    # Write folds to output file
    write_folds(data, folds, output_file)
    
    print(f"Successfully created {n_splits}-fold stratified cross-validation file.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create stratified k-fold cross-validation file')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('output_file', help='Path to output folds CSV file')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds (default: 10)')
    
    args = parser.parse_args()
    
    generate_folds(args.input_file, args.output_file, args.folds)


if __name__ == "__main__":
    main()
