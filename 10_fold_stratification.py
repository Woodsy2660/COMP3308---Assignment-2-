import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold

def create_stratified_folds(
    input_csv: str,
    output_csv: str,
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
    header: int | None = None
) -> None:
    """
    Reads the Pima CSV, splits it into stratified folds, and writes them out in the
    ED-friendly format (fold1...foldN headings, rows, blank line).

    This optimized version uses pandas.to_csv to speed up writing.
    """
    # Measure total operation time
    total_start = time.time()

    # Load data
    read_start = time.time()
    df = pd.read_csv(input_csv, header=header)
    print(f"Loaded data from '{input_csv}' with shape {df.shape} in {time.time() - read_start:.2f}s", flush=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:,  -1]

    # Prepare stratified splitter
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )
    print(f"Initialized StratifiedKFold with n_splits={n_splits}, shuffle={shuffle}, random_state={random_state}", flush=True)

    # Write out each fold using pandas.to_csv for speed
    with open(output_csv, 'w') as out:
        for fold_num, (_, test_idx) in enumerate(skf.split(X, y), start=1):
            fold_start = time.time()
            fold_df = df.iloc[test_idx]
            print(f"Processing fold {fold_num} with {len(test_idx)} samples...", flush=True)

            # Write fold header
            out.write(f'fold{fold_num}\n')
            # Write all rows for this fold at once
            fold_df.to_csv(out, header=False, index=False)
            out.write('\n')

            print(f"  -> Fold {fold_num} written in {time.time() - fold_start:.2f}s", flush=True)

    print(f"All folds written to '{output_csv}' in {time.time() - total_start:.2f}s", flush=True)


if __name__ == "__main__":
    print("Starting stratified folds creation...", flush=True)
    create_stratified_folds(input_csv='data/train/occupancy.csv',output_csv='data/stratification/occupancy-folds.csv',n_splits=10,shuffle=True,random_state=42,header=None)
