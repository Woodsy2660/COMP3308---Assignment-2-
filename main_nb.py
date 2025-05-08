# main_nb.py

import sys
from nb import classify_nb

def main():
    if len(sys.argv) != 3:
        print("Usage: python main_nb.py <training_csv> <testing_csv>")
        sys.exit(1)

    training_path = sys.argv[1]
    testing_path  = sys.argv[2]

    predictions = classify_nb(training_path, testing_path)
    for label in predictions:
        print(label)

if __name__ == '__main__':
    main()
