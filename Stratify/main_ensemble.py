# main_ensemble.py

import sys
from ensemble import classify_ensemble

def main():
    if len(sys.argv) != 5:
        print("Usage: python main_ensemble.py <training_csv> <testing_csv> <k1> <k2>")
        sys.exit(1)

    training_path = sys.argv[1]
    testing_path  = sys.argv[2]
    k1            = int(sys.argv[3])
    k2            = int(sys.argv[4])

    predictions = classify_ensemble(training_path, testing_path, k1, k2)
    for label in predictions:
        print(label)

if __name__ == '__main__':
    main() 