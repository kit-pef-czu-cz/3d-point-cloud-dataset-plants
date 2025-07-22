import random

def search_for_folds(df, categories, n_folds=5, col_combinations="combination", col_weights=None, col_folds="fold", start_seed=0, iterations=1):
    """
    Adds a 'fold' column to a new copy of the provided dataframe that balances
    the data into n_folds k-folds so that each unique value from the col_combinations
    column is represented in each k-fold evenly. Each sample can be weighted, e.g.
    by the number of plants it contains.
        Parameters:
            df: The input dataframe.
            categories: List of column names used for categories. 
            n_folds: Number of output k-folds.
            col_combinations: Name of the input dataframe's column to balance the data around. If None, do a simple random distribution.
            col_weights: Name of the input dataframe's column to weigh each sample by. If None, all have the weight of 1.
            col_folds: Name of the column to be created with fold IDs
            start_seed: Starting random seed. Each iteration increments the seed by 1.
            iterations: Number of search iterations.
        Returns:
            data: A processed pandas DataFrame with the lowest MAE.
    """
    random.seed(start_seed)
    worst_mae = None
    best_mae = None
    best_data = None
    for i_seed in range(start_seed, start_seed+iterations):
        i_data = df.sample(frac=1, random_state=i_seed).reset_index(drop=True)
        i_data = create_folds(i_data, n_folds, col_combinations, col_weights, col_folds, start_seed)
        mae = get_mae(i_data, categories, n_folds, col_weights, col_folds)
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_data = i_data
        if worst_mae is None or mae > worst_mae:
            worst_mae = mae
    print(f"best: {best_mae:.4f}\nworse: {worst_mae:.4f}")
    return best_data


def create_folds(df, n_folds=5, col_combinations="combination", col_weights="weights", col_folds="fold", random_seed=0):
    """
    Adds a 'fold' column to a new copy of the provided dataframe that balances
    the data into n_folds k-folds so that each unique value from the col_combinations
    column is represented in each k-fold evenly. Each sample can be weighted, e.g.
    by the number of plants it contains.
        Parameters:
            df: The input dataframe.
            n_folds: Number of output k-folds.
            col_combinations: Name of the input dataframe's column to balance the data around. If None, do a simple random distribution.
            col_weights: Name of the input dataframe's column to weigh each sample by. If None, all have the weight of 1.
        Returns:
            data: A processed pandas DataFrame
    """
    random.seed(random_seed)
    df = df.copy()
    df[col_folds] = 0
    ratios = [1 / n_folds] * n_folds
    if col_combinations is not None:
        # The cooler splitting
        for combo_name in df[col_combinations].unique():
            # Go combination by combination
            combo = df[df[col_combinations] == combo_name]
            fold_combo(df, combo, ratios, col_weights, col_folds)
    else:
        # Boring random shuffling
        fold_combo(df, df, ratios, col_weights, col_folds)
    return df


def fold_combo(df, combo, ratios, col_weights, col_folds):
    """
    Fold creation substep - not for users!
    Assigns each row a number so that their distribution is balanced
    as much as possible according to the values in ratios.
        Parameters:
            df: The input dataframe, edited in-place.
            combo: The view of the group to balance from df.
            ratios: A list of numbers adding up to 1, desired ratios of the individual k-folds. Usually all are identical.
            col_weights: Name of the input dataframe's column to weigh each sample by. If None, all have the weight of 1.
            col_folds: Name of the input dataframe's column to write k-fold IDs from 0 to len(ratios)-1 to.
        Returns:
            None
    """
    total_sum = combo[col_weights].sum() if col_weights else len(combo)
    target_sums = [r * total_sum for r in ratios]
    current_sums = [0] * len(ratios)
    # Haters will say iterrows() is slow. Too bad!
    for row in combo.iterrows():
        diff = [target_sums[i] - current_sums[i] for i in range(len(ratios))]
        index = random.choice([i for i, val in enumerate(diff) if val == max(diff)])
        df.at[row[0], col_folds] = index
        sum_add = row[1][col_weights] if col_weights else 1
        current_sums[index] += sum_add
        

def get_mae(data, categories, num_splits, col_weights, col_folds):
    from sklearn.metrics import mean_absolute_error
    totals_dict = {c: {
        val: data[data[c] == val][col_weights].sum() for val in data[c]
        } for c in categories
    }
    target_ratios = [1 / num_splits] * (sum([len(x) for x in totals_dict.values()]) * num_splits)
    split_names = data[col_folds].unique()
    i_ratios = []
    for cl in categories:
        for val in data[cl].unique():
            grouped = data[data[cl] == val].groupby(col_folds)[col_weights].sum()
            for split in split_names:
                if not split in grouped:
                    i_ratios.append(0)
                    continue
                i_ratios.append(grouped[split] / totals_dict[cl][val])
    return mean_absolute_error(target_ratios, i_ratios)