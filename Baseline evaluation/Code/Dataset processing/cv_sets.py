from pathlib import Path

def apply_names(value, val, test):
    if value == test:
        return "test"
    if value == val:
        return "val"
    return "train"
    
def create_simple_cv_sets(df, method_name="data", col_filename="filename", col_folds="fold"):
    """
    Creates cross-validation sets from the given data that contains a folds column.
    The last fold is used as the fixed test set, and the other folds create CV sets with
    different validation sets. In each CV set, only one fold is used for the test set,
    and only one fold is used for the validation set.  
    
    For example, 5 k-folds result in 4 CV sets being made:  
    1      2      3      4      5  
    train  train  train  val    test   
    train  train  val    train  test   
    train  val    train  train  test   
    val    train  train  train  test   
    
        Parameters:
            df: The input dataframe.
            col_folds: The column with fold IDs in the input dataframe.
            method_name: Used for the output dictionary keys.
        Returns:
            cvs: A dictionary in the format
            { method_name-val_fold: { file_name: 'train'|'val'|'test' } }
    """
    
    result = {}
    folds = df[col_folds].unique()
    folds.sort()
    for f in folds[:-1]:
        cv = df[[col_filename, col_folds]].copy()
        cv[col_folds] = cv[col_folds].apply(lambda x: apply_names(x, f, folds[-1]))
        cv.rename(columns={col_folds: "split"}, inplace=True)
        result[f"{method_name}-{f}"] = cv
    return result

def create_nested_cv_sets(df, method_name="data", col_filename="filename", col_folds="fold"):
    """
    Creates nested cross-validation sets from the given data that contains a folds column.
    Every combination of exactly 1 test fold, 1 val fold, n-2 train folds is created.
    Multiple folds aren't used to form a val or test set in one CV set.
    5 k-folds result in 20 total CV sets, i.e. 5 outer CV sets (with differing
    test sets), each made out of 4 inner CV sets (with differing val sets).
        Parameters:
            df: The input dataframe.
            col_folds: The column with fold IDs in the input dataframe.
            method_name: Used for the output dictionary keys.
        Returns:
            cvs: A dictionary in the format
            { method_name-test_fold-val_fold: { file_name: 'train'|'val'|'test' } }
    """
    result = {}
    folds = df[col_folds].unique()
    folds.sort()
    for tf in folds:
        for vf in folds:
            if tf == vf:
                continue
            cv = df[[col_filename, col_folds]].copy()
            cv[col_folds] = cv[col_folds].apply(lambda x: apply_names(x, vf, tf))
            cv.rename(columns={col_filename: "cloud_name", col_folds: "split"}, inplace=True)
            result[f"{method_name}-t{tf}-v{vf}"] = cv
    return result

def save_cvs(cv_dict, out_path, sep=","):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    for k, v in cv_dict.items():
        v.to_csv(out_path / f"{k}.csv", sep=sep, index=False)












