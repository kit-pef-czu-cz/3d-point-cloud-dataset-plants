These scripts provide splitting the dataset into k-folds for cross-validation and train/val/test splits and converting the dataset into the folder and file structure expected by OpenPCDet. For the latter, the OpenPCDet code patches and additions from `/Code/OpenPCDet` are also necessary.

# Preparing the dataset
* **Prerequisities**
	* The dataset: Point clouds, generated cuboid annotations, and `Annotation data.csv` (if not using the provided baseline splits)
	* OpenPCDet (see installation instructions in `Code/OpenPCDet/README.md`
	* Pandas, Open3D
		* Tested with `python=3.11.8 pandas=2.2.3 open3d=0.18.0`
		* If skipping the first step, scikit-learn is not necessary
	* Your work folder should look like this:
```
work/
├─Annotation data.csv
├─create_baseline_cvs.py
├─crossval2pcdet.py
├─cv_folds.py
├─cv_read.py
├─cv_sets.py
├─cv_vis.py
├─nested_crossval2pcdet.sh
└─dataset/
  ├─50-27-0_0-20220524T041419.pcd
  ├─50-27-0_0-20220524T041419.txt
  └─...
```

* **Creating nested cross-validation splits**
	* If using the provided splits in `/Baseline Splits`, this step can be skipped: Move the provided CSVs to a `./cvs_repo/` subfolder
	* Run `python create_baseline_cvs.py`
	* This creates shuffled nested k-fold cross-validation sets (CVs) in the form of 20 .csv files. Each one contains train/val/test split assignments for each sample. The output will be in `./cvs_repo/`.
	* `dataset_scripts/cv_*.py` files contain functions for custom data splitting methods, non-nested k-folds, and visualisation.
		* Fold search requires `scikit-learn`, plotting requires `matplotlib`
* **Converting to OpenPCDet structure**
	* Run `./nested_crossval2pcdet.sh ./dataset ./cvs_repo repo_rnd_0 ./data_out`
	* This runs the `crossval2pcdet.py` script for each .csv file with the required parameters
	* Output folders are in the newly created `./data_out` folder. Move its contents to `OpenPCDet/data`
* **Generating OpenPCDet custom infos**
	* Under `OpenPCDet`, run `./nested_custom_infos.sh ./tools/cfgs/base_dat_second.yaml repo_rnd_0`
	* This runs the OpenPCDet script for creating its required .pkl annotation files and optional ground truth samples
	* If not using nested CVs, run the Python script directly with the necessary parameters (use `-h` for help):
		* `python ./pcdet/datasets/custom/custom_dataset.py`
	* The dataset is now ready for training and evaluation. Your `OpenPCDet/data` folder should look like this:
```
data
├─repo_rnd_0-t0-v1/
│ ├─ImageSets/
│ │ ├─test.txt
│ │ ├─train.txt
│ │ └─val.txt
│ ├─custom_dbinfos_train.pkl
│ ├─custom_infos_test.pkl
│ ├─custom_infos_train.pkl
│ ├─custom_infos_val.pkl
│ ├─gt_database/
│ │ ├─000000_Embryonic_leaf_0.bin
│ │ └─...
│ ├─labels/
│ │ ├─000000.txt
│ │ └─...
│ ├─name_id_map.csv
│ └─points/
│   ├─000000.npy
│   └─...
├─repo_rnd_0-t0-v2/
│ └─...
└─...
```