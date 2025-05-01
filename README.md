# COMP3308 Assignment 2: Machine Learning Classifiers

This repository contains the implementation of three machine learning classifiers from scratch:

1. **K-Nearest Neighbour** (`knn.py`)
   - Function signature: `classify_nn(training_file, testing_file, k)`
   - Majority vote with ties resolved to "yes"

2. **Gaussian Naive Bayes** (`nb.py`)
   - Function signature: `classify_nb(training_file, testing_file)`
   - Normal PDF for numeric attributes

3. **Ensemble** (`ensemble.py`)
   - Function signature: `classify_ensemble(training_file, testing_file, k1, k2)`
   - Combines KNN (k1), KNN (k2), and Naive Bayes predictions

## Usage

Each classifier has its own entry point:

```bash
# KNN classifier
python main_knn.py <training_csv> <testing_csv> <k>

# Naive Bayes classifier
python main_nb.py <training_csv> <testing_csv>

# Ensemble classifier
python main_ensemble.py <training_csv> <testing_csv> <k1> <k2>
```

## Datasets

- `pima.csv`: Pima Indians Diabetes dataset
- `occupancy.csv`: Room occupancy detection dataset 
