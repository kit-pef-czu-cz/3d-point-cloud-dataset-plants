The following files are modified versions of those from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), commit [8caccce](https://github.com/open-mmlab/OpenPCDet/commit/8cacccec11db6f59bf6934600c9a175dae254806), which is available under the Apache license (see the LICENSE file). The following changes were made:

## Files description
**datasets/custom/custom_dataset.py**
* Previously hardcoded limit for using a "train" and "test" set is removed - any filename can be used
	* This enables the use of a validation set as well
* Added support for augmentations
* Dataset config YAMLs now use the following extra options:
	* CULS_AUGMENTS: [str]
		* Names of augmentations to use
	* CULS_AUGMENTS_VAL: bool
		* Whether to use augmentations in the validation set as well

**datasets/processor/data_processor.py**
* Fix for [issue #313](https://github.com/open-mmlab/OpenPCDet/issues/313), sourced from the issue thread

**models/dense_heads/anchor_head_single.py**  
**models/dense_heads/point_head_box.py**  
**models/roi_heads/pointrcnn_head.py**  

* Enable generating targets during evaluation to allow calculating validation loss. See [issue #1626](https://github.com/open-mmlab/OpenPCDet/issues/1626#issuecomment-2282274279)

**models/detectors/detector3d_template.py**
* Added `map_location` argument to `load_params_from_file` to allow loading a checkpoint trained on a GPU with a different ID to another GPU, replacing the previous `to_cpu` bool argument
	* e.g. A model trained on GPU #4 can be moved to GPU #1 for evaluation

---
The following files don't come from OpenPCDet:  
**nested_custom_infos.sh**  
* Runs `datasets/custom/custom_dataset.py` for nested CV datasets in the format `<name>-t<t>-v<v>`.
* Takes two mandatory arguments: path to the desired dataset config file, usually under `tools/cfgs`, and the base name of the dataset.

## Installing OpenPCDet
* Install [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) according to the [instructions](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md).
* Tested with OpenPCDet commit 8caccce (Apr 17, 2024; currently latest), Python 3.11.8, PyTorch 2.3.1, pytorch-cuda 12.1, CUDA 12.4, NumPy 1.26.4
	* Do NOT use NumPy 2! Installation will fail.
	* Untested on Windows - especially CUDA and some dependencies cause severe installation issues and failures.
* Merge this repo's `OpenPCDet` folder into the OpenPCDet installation folder
* Evaluation scripts additionally require SciPy and scikit-learn
	* Tested with SciPy 1.14.1 and scikit-learn 1.5.2

## Training and evaluation
* You can use the `run_train.py` and `run_test.py` scripts directly with command line arguments. Check the `-h` help argument!
* `sh-train-nested.sh` is provided for easier training of nested CVs on multiple GPUs. It runs one outer CV at once (4 training sessions). If there are fewer than 4 GPUs, two or more training sessions will run at once on a given GPU.
* `sh-07-12_baseline.sh` runs training sessions for the baseline results for both SECONDNet and PointRCNN, saves results under `results/baseline_second` and `results/baseline_pointrcnn`, and shuts down the computer.
* By default, `run_train.py` automatically does per-epoch evaluation on both the train and validation sets and post-train evaluation on the test set; `run_test.py` isn't strictly necessary.
	* Loading a checkpoint file trained using PointRCNN doesn't appear to work at the moment.
* Results are saved into `log_train.csv`, `log_val.csv` and `log_test.csv`. `metrics.pkl` contains detailed metrics (precision, recall, F-score, RMSE, MAE, and R^2 on all confidence thresholds levels between 0.01 and 1.0).