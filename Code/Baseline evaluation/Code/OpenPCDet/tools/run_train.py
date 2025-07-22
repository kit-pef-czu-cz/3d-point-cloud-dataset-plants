#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, logging
import culs_defs_loading as culs_loading
import culs_defs_train as culs_training
import culs_defs_callbacks as callbacks
import culs_defs_eval as culs_eval
import warnings, numba
import scipy.stats
import pcdet.config
import torch, numpy, random

def main(cfg_path, data_path, epochs, out_path, es_patience, es_warmup, es_obj, batch_size, augs, aug_val, epochs_per_eval):
    culs_loading.ensure_determinism()

    # disable some warnings
    warnings.filterwarnings("ignore", category=UserWarning) # PyTorch
    warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning) # PyTorch
    warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)
    
    # load config
    cfg = culs_loading.load_config(cfg_path, epochs)
    
    if data_path is not None:
        cfg.DATA_CONFIG.DATA_PATH = data_path        

    if len(augs) > 0 and augs[0] != "__NONE":
        cfg.DATA_CONFIG.CULS_AUGMENTS = augs
        cfg.DATA_CONFIG.CULS_AUGMENTS_VAL = aug_val
    
    # create logger
    logger = culs_loading.create_console_logger()
    
    # load dataset splits
    train_data = culs_loading.create_loaders(cfg, logger, "train", epochs=epochs, batch_size=batch_size)
    val_data = culs_loading.create_loaders(cfg, logger, "val", epochs=epochs)
    test_data = culs_loading.create_loaders(cfg, logger, "test", epochs=epochs)
    
    # build the model
    model, optimizer, lr_sched, lr_wu_sched = culs_loading.build_model(cfg, train_data)
    
    # create callbacks
    epoch_callbacks, end_callbacks = [], []
    if es_patience > 0:
        epoch_callbacks.append(
            callbacks.EarlyStopCallback(epochs, es_obj[0], es_obj[1], es_patience, es_warmup, out_path))
    epoch_callbacks.append(
        callbacks.CsvMetricsCallback(out_path, append=False))
    epoch_callbacks.append(
        callbacks.SaveBestCheckpointsCallback(
            [("ap_sum_0.3", "max")],
            out_path
        )
    )
    end_callbacks.append(
        callbacks.CsvMetricsCallback(out_path, append=False))
    end_callbacks.append(
        callbacks.SaveLastCheckpointCallback(out_path + "/latest"))
    
    # Pickle metrics callback
    pickle_cb = callbacks.PickleMetricsCallback(out_path)
    epoch_callbacks.append(pickle_cb)
    end_callbacks.append(pickle_cb)
    
    # train the model
    result = culs_training.train_model(
        cfg, model, optimizer, lr_sched, lr_wu_sched, logger,
        train_data, val_data, test_data,
        culs_eval.evaluate_model_scores, culs_eval.evaluate_model_scores, culs_eval.evaluate_model_scores,
        epoch_callbacks, end_callbacks, epochs_per_eval)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str,
                        help="Path to the model YAML config file.")
    parser.add_argument("output", type=str,
                        help="Output folder path.")
    parser.add_argument("--epochs", "-e", type=int, default=None, nargs="?",
                        help="Number of epochs to train for. If None, use epochs from the config.")
    parser.add_argument("--dataset", "-d", type=str, default=None, metavar="PATH",
                        help="Alternative dataset path. Does not affect the rest of the dataset configuration - the alt. set must be compatible!")
    parser.add_argument("--gpu", "-g", type=int, default=0, metavar="ID",
                        help="ID of the GPU to use.")
    parser.add_argument("--es-patience", type=int, default=0, metavar="EPOCHS",
                        help="Patience in epochs for the early stopping mechanism. If 0, don't use early stopping. (default: %(default)s)")
    parser.add_argument("--es-warmup", type=int, default=0, metavar="EPOCHS",
                        help="Warm-up period in epochs for the early stopping mechanism during which patience isn't checked for. (default: %(default)s)")
    parser.add_argument("--es-objective",type=str, default=["loss", "min"], nargs="*",
                        help="Two strings: The metric name to check for, e.g. 'loss', and either 'min' or 'max' for its objective.")
    parser.add_argument("--batch-size", type=int, default=1, metavar="BZ")
    parser.add_argument("--augs", type=str, nargs="*", default="",
                        help="List of augmentations to use.")
    parser.add_argument("--aug-val", action='store_true',
                        help="If set, use augmentations with the validation set as well.")
    parser.add_argument("--epochs-per-eval", type=int, default=1, metavar="EPOCHS",
                        help="Only make evaluations every n-th epoch.")
    args = parser.parse_args()
    if len(args.es_objective) != 2:
        raise ValueError("--es-objective must have two values specified when used.")
    if args.es_objective[1] not in ["min", "max"]:
        raise ValueError("--es-objective 2nd value must be 'min' or 'max'.")
    if len(args.augs) == 1 and "@" in args.augs[0]:
        args.augs = args.augs[0].split("@")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.gpu >= 0:
        gpu_id = args.gpu % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)
        if args.gpu >= torch.cuda.device_count():
            print(f"GPU ID {args.gpu} out of range {torch.cuda.device_count()}; rolling over to {gpu_id}.")
    else:
        print("Invalid GPU ID. Using 0.")
    main(args.config, args.dataset, args.epochs, args.output, args.es_patience, args.es_warmup, args.es_objective, args.batch_size, args.augs, args.aug_val, args.epochs_per_eval)
