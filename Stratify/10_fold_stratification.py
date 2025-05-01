import pandas as pd
from sklearn.model_selection import StratifiedKFold

def generate_stratified_folds(df, output_path, join_char=','):
    """
    Generate and write stratified K-folds to a file.
    - df: DataFrame with last column as class labels.
    - output_path: path to write the folds file.
    - join_char: character to join fields (',' for CSV, '\t' for TSV).
    """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    lines = []
    for fold_idx, (_, test_idx) in enumerate(skf.split(X, y), start=1):
        lines.append(f'fold{fold_idx}')
        for row in df.iloc[test_idx].itertuples(index=False):
            lines.append(join_char.join(map(str, row)))
        lines.append('')  # blank line between folds

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Wrote stratified folds to: {output_path}')

if __name__ == '__main__':
    # Generate pima-folds.csv (comma-separated)
    df_pima = pd.read_csv('pima.csv', header=None)
    generate_stratified_folds(df_pima, 'pima-folds.csv', join_char=',')

    # Generate occupancy-folds.csv (tab-separated)
    df_occ = pd.read_csv('occupancy.csv', header=None, sep='\t')
    generate_stratified_folds(df_occ, 'occupancy-folds.csv', join_char='\t')