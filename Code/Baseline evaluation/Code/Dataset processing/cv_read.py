import numpy as np
import pandas as pd
import functools
import os
from datetime import datetime

def read_csv(path, sep=",", decimal="."):
    df = pd.read_csv(path, sep=sep, decimal=decimal, usecols=[
        "Specie",
        "Date time",
        "File name",
        "Exp. num."
    ])
    df.rename(columns={
        "Specie": "species",
        "Date time": "date",
        "File name": "file_name",
        "Exp. num.": "exp_num",
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df['species'] = df['species'].fillna("N/A")
    df = df.groupby('file_name').agg({
        'species': 'first',
        'date': 'first',
        'exp_num': 'first',
        'file_name': 'size'
    }).rename(columns={'file_name': 'num_plants'}).reset_index()
    df["ones"] = 1
    return df

def read_excel(path):
    """
    Reads the Annotation figure IDs Excel file and does basic processing to it.
        Parameters:
            path: Path to the file.
        Returns:
            df: A processed pandas DataFrame
    """
    df = pd.read_excel(path, skiprows=1, usecols=[
        "Specie",
        "Labeling status",
        "Date time",
        "file_name",
        "Dataset",
        "Exp. num."
        ])
    df.rename(columns={
        "Date time": "date",
        "Specie": "species",
        "Dataset": "dataset",
        "Exp. num.": "exp_num",
        "Labeling status": "status"
    }, inplace=True)
    df = df[df["status"] == "double-checked"]
    df["date"] = pd.to_datetime(df["date"])
    df['species'] = df['species'].fillna("N/A")
    df = df.groupby('file_name').agg({
        'species': 'first',
        'date': 'first',
        'dataset': 'first',
        'exp_num': 'first',
        'file_name': 'size'
    }).rename(columns={'file_name': 'num_plants'}).reset_index()
    df["ones"] = 1
    return df

# ------------------
# MODIFIER FUNCTIONS
# ------------------

# Main

def replace_filenames(df, path):
    csv = pd.read_csv(path)
    replace_dict = {
         row[1]['Annotation figure IDs file name']: row[1]['common_database_double-checked file name ']
         for row in csv.iterrows()
    }
    return df.replace(replace_dict)

def format_column(df, column, format_string):
    def formatter(value):
        return format_string.format(value=value)
    df[column] = df[column].apply(formatter)
    return df

def add_file(df, path, left_on, right_on, cols=None, csv_sep=",", csv_dec="."):
    if path.endswith("xlsx"):
        file = pd.read_excel(path)
    else:
        file = pd.read_csv(path, sep=csv_sep, decimal=csv_dec)
    if cols is not None:
        file = file[cols + [right_on]]
    df = df.merge(file, left_on=left_on, right_on=right_on)
    if left_on != right_on:
        df = df.drop(right_on, axis=1)
    return df

def add_quantile_bins(df, column, bins=None, quantiles=None):
    if bins is not None and quantiles is not None:
        raise ValueError("Only one of 'bins' or 'quantiles' may be specified.")
    if bins is None and quantiles is None:
        raise ValueError("Either 'bins' or 'quantiles' must be specified.")
    arr = df[column]
    if bins is not None:
        quantiles = [x/bins for x in range(1, bins+1)]
    q_values = np.quantile(arr, quantiles)
        
    q_values[-1] += 1
    new_col = np.array(arr, dtype=int)
    for i, inp in enumerate(arr):
        for j, q in enumerate(q_values):
            if inp < q:
                new_col[i] = j
                break
    col_name = column + "_binned"
    df[col_name] = new_col
    q_values[-1] -= 1
    df[col_name] = df[col_name].apply(lambda x: f"<{q_values[x]:.2f} ({quantiles[x]*100:.1f}%)")
    return df

def add_combinations(df, categories):
    df["combination"] = ""
    for cl in categories:
        df["combination"] += "_" + df[cl].apply(str)
    return df

def drop_samples(df, samples):
    return df[~df['file_name'].isin(samples)].reset_index(drop=True)

# Legacy

def add_days_since_seeding(df):
    seeding_dates = {51: '2022-06-18', 50: '2022-05-18', 59: '2023-07-12'}
    def calculate_days_from_seeding(row):
        seeding_date = datetime.strptime(seeding_dates[row['exp_num']], '%Y-%m-%d')
        return (row['date'] - seeding_date).days
    df['days'] = df.apply(calculate_days_from_seeding, axis=1)
    return df

def add_age_category(df):
    def time_category(days):
        if days <= 7:
            return "00-07 days"
        elif 8 <= days <= 18:
            return "08-18 days"
        else:
            return "18+ days"
    df['age_category'] = df['days'].apply(time_category)
    return df
    
def add_count_category(df):
    def num_category(num):
        if num <= 4:
            return "1-4 plants"
        elif 5 <= num <= 7:
            return "5-7 plants"
        else:
            return "8+ plants"
    df["num_category"] = df["num_plants"].apply(num_category)
    return df


# Deprecated

def add_oi_index(df, path):
    csv = pd.read_csv(path, header=None, names=["file_name", "oi"])
    csv.file_name = csv.file_name.apply(lambda x: f"{x}.pcd")
    df = df.merge(csv, on="file_name")
    return df

def add_point_counts(df, pcd_folder_path):
    """
    Adds a column 'points' with number of points of each point cloud calculated
    by reading each file in the given directory.
    """
    import open3d as o3d
    points = []
    for f in df.file_name:
        pcd = o3d.io.read_point_cloud(os.path.join(pcd_folder_path, f))
        points.append(len(pcd.points))
    df['points'] = points
    return df























    