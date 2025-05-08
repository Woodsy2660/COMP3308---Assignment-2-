# create_pima_folds.py

import pandas as pd
import numpy as np

# Specify input and output paths directly in the code\ nINPUT_CSV = 'pima.csv'
OUTPUT_CSV = 'pima-folds.csv'


def get_stratified_fold_indices(y, n_splits=10, shuffle=True, seed=42):
    """
    Partition indices 0…len(y)-1 into n_splits folds so that
    each fold has (approximately) the same class distribution as y.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    # collect indices per class
    cls2idx = {cls: np.where(y == cls)[0] for cls in classes}
    if shuffle:
        for idxs in cls2idx.values():
            rng.shuffle(idxs)
    # round-robin assignment
    folds = [[] for _ in range(n_splits)]
    for cls in classes:
        for i, idx in enumerate(cls2idx[cls]):
            folds[i % n_splits].append(idx)
    return [np.array(fold, dtype=int) for fold in folds]


def write_folds(df, folds, output_path):
    """
    Write folds to a single CSV in the form:
      fold1
      <csv rows>

      fold2
      <csv rows>

      …
    """
    with open(output_path, "w") as out:
        for i, idxs in enumerate(folds, start=1):
            out.write(f"fold{i}\n")
            df.iloc[idxs].to_csv(out, header=False, index=False)
            out.write("\n")


def genereate_folds(input):
    # load the Pima dataset; assumes last column is the binary target (yes/no or 1/0)
    df = pd.read_csv(input)
    y = df.iloc[:, -1].values
    folds = get_stratified_fold_indices(y, n_splits=10, shuffle=True, seed=42)
    write_folds(df, folds, input + "-folds.csv")
    print(f"Wrote 10 stratified folds to '{OUTPUT_CSV}'")
    
def main():
    genereate_folds("")
    genereate_folds("")


if __name__ == "__main__":
    main()
