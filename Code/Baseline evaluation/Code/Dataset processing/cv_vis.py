import numpy as np
import matplotlib.pyplot as plt

# FOLDS

def plot_folds(data, label, categories, col_weights):
    plot_dict = get_presplit_plot_dict(data, categories, col_weights)
    fig, axes = plt.subplots(ncols=len(categories), nrows=1, figsize=[5*len(categories),5], dpi=144)
    if len(categories) == 1:
        axes = [axes]
    fig.suptitle("Percentage of each category value in k-folds" + f" ({label})" if label else "", y=1.0)
    for c, ax in zip(categories, axes):
        ax.set_title(c)
        ax.axhline(y=20)
        ax.set_ylim([10, 30])
        ax.boxplot(plot_dict[c].values(), labels=plot_dict[c].keys(), showcaps=False, showbox=False, showfliers=False, whis=0.0, medianprops={'alpha': 0.0})
        for i, kv in enumerate(plot_dict[c].items()):
            k = kv[0]
            v = kv[1]
            d = 1 / len(v) / 8
            x = [i+1+d*(l-len(v)/2) for l in range(len(v))]
            ax.plot(x, v, 'ko', label=k, alpha=0.5)
        ax.set_ylabel("Percentage of total")
        

def get_presplit_plot_dict(df, categories, col_weights):
    totals = {x: {} for x in categories}
    for c in categories:
        for val in df[c].unique():
            grouped = df[df[c] == val].groupby("fold")[col_weights].sum()
            totals[c][val] = grouped.sum()
    
    cvs = {c: {
        v: [] for v in np.sort(df[c].unique())
    } for c in categories}
    
    for c in categories:
        for val in df[c].unique():
            grouped = df[df[c] == val].groupby("fold")[col_weights].sum()
            for i in grouped.index:
                cvs[c][val].append(grouped[i] / totals[c][val] * 100.0)
    
    return cvs


# MATRIX 

def make_matrix(df, split_column, classes, col_weights):
    split_names = df[split_column].unique()
    if "train" in split_names and "val" in split_names and "test" in split_names:
        split_names = ["train", "val", "test"]
    else:
        split_names = np.sort(split_names)
    matrix = [[" "] + list(split_names)]
    for cl in classes:
        for val in np.sort(df[cl].unique()):
            line = []
            line.append(val)
            grouped = df[df[cl] == val].groupby(split_column)[col_weights].sum()
            total = grouped.sum()
            for split in split_names:
                if split in grouped:
                    line.append(f"{grouped[split]} ({(grouped[split] / total * 100):.1f}%)")
                else:
                    line.append("0 (0%)")
            matrix.append(line)
        matrix.append(["---"] * (len(split_names)+1))
    line = ["total"]
    total = df[col_weights].sum()
    for split in split_names:
        split_count = df[df[split_column] == split].sum(numeric_only=True)[col_weights]
        line.append(f"{split_count:.0f} ({(split_count / total * 100):.1f}%)")
    matrix.append(line)
    return matrix

def print_table(data) -> None:
    def print_hr(lengths) -> None:
        print("-" * (sum(lengths) + 3*len(lengths) + 1))
    lengths = [0 for x in data[0]]
    for row in data:
        i = 0
        for cell in row:
            lengths[i] = max(lengths[i], len(str(cell)))
            i += 1
    i = 0
    print_hr(lengths)
    for row in data:
        for cell, l in zip(row, lengths):
            print("| " + str(cell).rjust(l), end=" ")
        print("|")
        if i == 0:
            print_hr(lengths)
        i += 1
    print_hr(lengths)