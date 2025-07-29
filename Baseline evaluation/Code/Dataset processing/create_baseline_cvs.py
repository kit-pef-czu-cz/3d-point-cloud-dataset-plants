import cv_read, cv_folds, cv_sets, cv_vis

df = cv_read.read_csv("./Annotation data.csv")
df = cv_read.format_column(df, "file_name", "{value}.pcd")

folds = cv_folds.create_folds(df, col_weights=None, col_combinations=None, random_seed=0)
#folds[['file_name', 'split']].to_csv("./repo_rnd_0-fold_numbers.csv", index=False)
cvs = cv_sets.create_nested_cv_sets(folds, method_name="repo_rnd_0", col_filename="file_name")
cv_sets.save_cvs(cvs, "./cvs_repo/")
