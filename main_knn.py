# main_knn.py

import sys
from knn import classify_nn

def main():
    if len(sys.argv) != 4:
        print("Usage: python main_knn.py <training_csv> <testing_csv> <k>")
        sys.exit(1)

    training_path   = sys.argv[1]
    testing_path    = sys.argv[2]
    neighbour_count = int(sys.argv[3])

    predictions = classify_nn(training_path, testing_path, neighbour_count)
    for label in predictions:
        print(label)

if __name__ == '__main__':
    main() 