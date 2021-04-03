import pandas as pd


def gen_folds(data, nr):
    if isinstance(data, pd.DataFrame):
        # Split into roughly equal folds (some might be 1 row smaller than the rest)
        fold_len = int(len(shuffled_df)/nr)
        folds = [df.iloc[i*fold_len/(i+1)*fold_len] for i in range(nr)]
    else:
        folds = []
        X, Y = data
        fold_len = int(len(X)/nr)
        folds = [
            (X[i*fold_len:(i+1)*fold_len,:],
            Y[i*fold_len:(i+1)*fold_len,:])
        for i in range(nr)]

    return folds

def select_col(folds, col):
    return [list(fold[col]) for fold in folds]

def merge_folds(folds):
    if type(folds[0]) == list:
        merged = []
        for fold in folds:
            merged = merged + fold
        return merged
    else:
        return pd.concat(folds)
