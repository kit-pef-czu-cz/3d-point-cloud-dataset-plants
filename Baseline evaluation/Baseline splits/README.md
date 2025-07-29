The dataset has been randomly shuffled and split into 5 evenly sized k-folds. Each sample name - fold assignment can be found in `repo_rnd_0-fold_numbers.csv`.

In each resulting dataset, one k-fold is used as the test set, one as the validation set, and three as the train set, for a 60/20/20 percent split. Every possible combination of these is created. The base name is repo_rnd_0, followed by the ID of the k-fold used as the test set, followed by the ID of the fold used as the validation set, e.g. repo_rnd_0-t0-v1.csv uses folds 0 and 1 as the test and validation sets, respectively.

This distribution was created by `/Code/Dataset splitting/make_baseline_cvs.py` and was used for the baseline training results.