import os, argparse, sys
import numpy as np
import pandas as pd
import open3d as o3d

CFG_CLASS_REPLACES = [
    ("SegmentationPlant", "Plant"),
]
CFG_SAMPLE_DIGITS = 6

NAME_MAPPER_CSV_FILENAME = "common_database_double_checked_creation.txt"

PATH_SETS = "ImageSets"
PATH_POINTS = "points"
PATH_LABELS = "labels"

LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

def main_console():
    args = create_parser().parse_args()
    return main(args)
    
def main_notebook(data, csv, output, augmentations=None, augment_val=False, colors=False, luma=False):
    raw_args = [data, csv, output]
    if augmentations:
        raw_args.append("--augs")
        raw_args.extend(augmentations)
    if augment_val:
        raw_args.append("-v")
    if colors:
        raw_args.append("-c")
    if luma:
        raw_args.append("-l")
    args = create_parser().parse_args(raw_args)
    return main(args)
    
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str,
                        help="Input folder with point clouds in PCD format and labels in TXT.")
    parser.add_argument("csv", type=str,
                        help="Input CSV file from the cross-val script.")
    parser.add_argument("output", type=str,
                        help="Output folder path. The folder will contain a CSV file mapping original names to IDs in addition to populated folders 'points', 'labels', and 'ImageSets'.")
    parser.add_argument("--augs", type=str, nargs="+", default=None,
                        help="List of augmentations suffixes to look for.")
    parser.add_argument("--augment_val", "-v", action="store_true",
                        help="Whether to use augmentations for validation split samples as well. Train set samples are always augmented, test set samples are never augmented.")
    parser.add_argument("--colors", "--colours", "-c", action="store_true",
                        help="Whether to encode colour information in resulting point files. Make sure the input point files actually contain colour information!")
    parser.add_argument("--luma", "-l", action="store_true",
                        help="Whether to encode greyscale information in resulting point files converted from RGB colour data to be used as 'intensity'. Ignored if -c is set. Make sure the input point files actually contain colour information!")
    return parser

def main(args):
    inputs = read_and_validate(args)
    if inputs is None:
        print("No valid input files found.")
        return
    process_files(args, inputs)

def read_and_validate(args):
    df = check_paths_and_read_csv(args)
    if df is None:
        return None
    for path in [PATH_POINTS, PATH_LABELS, PATH_SETS]:
        if not make_and_check_output_path(args, path):
            return None
    return make_full_df(args, df)

def check_paths_and_read_csv(args):
    # arg path check
    if not os.path.exists(args.data):
        print("ERROR: Given input dataset folder doesn't exist.")
        return None
    try:
        return pd.read_csv(args.csv)
    except FileNotFoundError:
        print("ERROR: Given input CSV file doesn't exist.")
        return None
    except Exception as e:
        print(f"ERROR: Can't load CSV file: {repr(e)}")
        return None

def make_and_check_output_path(args, folder):
    path = os.path.join(args.output, folder)
    os.makedirs(path, exist_ok=True)
    if not os.path.exists(path):
        print("ERROR: Couldn't create output paths.")
        return False
    return True

def make_full_df(args, df):
    existing_files = os.listdir(args.data)
    if NAME_MAPPER_CSV_FILENAME in existing_files:
        df = replace_doublechecked_filenames(args, df)
    result = pd.DataFrame(columns=["cloud", "label", "split", "aug"])
    print(f"Processing {args.csv}...")
    successes = 0
    for _, row in df.iterrows():
        samples = get_sample_names(args, row, existing_files)
        if samples is None:
            continue
        successes += 1
        result = pd.concat([result, samples])
    print("Validation done. " +
          f"Validated {successes}/{len(df)} samples, {len(result)} incl. augmentations.")
    if len(result) == 0:
        return None
    return result

def replace_doublechecked_filenames(args, df):
    csv_map = pd.read_csv(os.path.join(args.data, NAME_MAPPER_CSV_FILENAME))
    # whyyyy these column names and with a bonus whitespace at the end?
    replace_dict = {
         row[1]['Annotation figure IDs file name']: row[1]['common_database_double-checked file name ']
         for row in csv_map.iterrows()
    }
    return df.replace(replace_dict)

def get_sample_names(args, row, files):
    base_cloud_name, base_cloud_ext = os.path.splitext(row.cloud_name)
    if row.cloud_name not in files:
        print(f"Cloud (base) not found: {row.cloud_name}")
        return None
    base_label_name = base_cloud_name + ".txt" 
    if base_label_name not in files:
        base_label_name = row.cloud_name + ".txt" 
    if base_label_name not in files:
        print(f"Label (base) not found for {base_cloud_name}")
        return None
    result = pd.DataFrame(columns=["cloud", "label", "split", "aug"])
    result.loc[len(result)] = [row.cloud_name, base_label_name, row.split, "None (base)"]
    if not args.augs or row.split == "test":
        return result
    if row.split == "val" and not args.augment_val:
        return result
    for aug in args.augs:
        aug_cloud_name = f"{base_cloud_name}{aug}.pcd"
        if aug_cloud_name not in files:
            print(f"Cloud (augm. {aug}) not found: {aug_cloud_name}")
            return None
        aug_label_name = f"{base_cloud_name}{aug}.txt"
        if not aug_label_name in files:
            print(f"Label (augm. {aug}) not found: {aug_label_name}")
            return None
        aug_split = f"{row.split}_{aug}"
        result.loc[len(result)] = [aug_cloud_name, aug_label_name, aug_split, aug]
    return result
    
def process_files(args, df):
    # print("Beginning file processing...")
    name_map = pd.DataFrame(columns=["id", "name", "augmentation"])
    imagesets = {}
    file_counter = 0
    for _, row in df.iterrows():
        cloud, label = read_sample(args.data, row.cloud, row.label)
        if not cloud:
            continue
        label = replace_class_names(label)
        points = np.asarray(cloud.points)
        if args.colors:
            colors = np.asarray(cloud.colors)
            points = np.concatenate((points, colors), axis=1)
        elif args.luminance:
            luma = np.asarray(cloud.colors).transpose()
            luma = LUMA_WEIGHTS.dot(luma)
            points = np.concatenate((points, luma), axis=1)
        id_name = str(file_counter).zfill(CFG_SAMPLE_DIGITS)
        save_data(points, label, args, id_name)
        name_map.loc[len(name_map)] = {"id": id_name, "name": row.cloud, "augmentation": row.aug}
        update_image_sets(imagesets, id_name, row.split)
        file_counter += 1
    name_map.to_csv(os.path.join(args.output, 'name_id_map.csv'), index=False)
    save_imagesets(args, imagesets)
    print(f"Processing done. Processed {len(name_map)}/{len(df)} samples.")

def read_sample(folder, cloud_name, label_name):
    try:
        pcd = o3d.io.read_point_cloud(os.path.join(folder, cloud_name))
        with open(os.path.join(folder, label_name), "r", encoding="utf-8") as fp:
            labels = fp.readlines()
        if len(labels) == 0:
            raise Exception(f"Label file is empty.")
        return pcd, labels
    except Exception as e:
        print(f"Error when reading sample {cloud_name}: {e}")
        return None, None

def replace_class_names(sample_labels: list[str]):
    for i, object_label in enumerate(sample_labels):
        for pair in CFG_CLASS_REPLACES:
            object_label = object_label.replace(pair[0], pair[1])
            sample_labels[i] = object_label
    return sample_labels

def save_data(pcd, labels, args, name):
    np.save(os.path.join(args.output, PATH_POINTS, f"{name}.npy"), pcd)
    label_out = "".join(labels)
    with open(os.path.join(args.output, PATH_LABELS, f"{name}.txt"), "w+", encoding="utf-8") as fp:
        fp.write(label_out)

def update_image_sets(sets, id_name, split):
    if not split in sets:
        sets.update({split: [id_name]})
    else:
        sets[split].append(id_name)
    
def save_imagesets(args, imagesets):
    for split, ids in imagesets.items():
        joined = "\n".join(ids)
        with open(os.path.join(args.output, PATH_SETS, split + ".txt"), "w+") as fp:
            fp.writelines(joined)
            
if __name__ == "__main__":
    # debug_result = main_notebook(
    #     "./data",
    #     "./crossval.csv",
    #     "./output",
    #     colors=False
    # )
    debug_result = main_console()
