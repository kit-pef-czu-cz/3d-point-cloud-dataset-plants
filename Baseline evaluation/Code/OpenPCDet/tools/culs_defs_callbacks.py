#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from pathlib import Path
import train_utils.train_utils as train_utils
from easydict import EasyDict

class CulsCallback:
    def __init__(self):
        pass
    
    def on_epoch_done(self, cfg, checkpoint, epoch, train_predictions, train_metrics, val_predictions, val_metrics, logger):
        raise NotImplementedError()
    
    def on_train_done(self, cfg, checkpoint, epoch, predictions, metrics, logger):
        raise NotImplementedError()
        
    def init_type_check(self, variables, types, names):
        for v, t, n in zip(variables, types, names):
            if isinstance(t, list) or isinstance(t, tuple):
                good = False
                for tt in t:
                    if isinstance(v, tt):
                        good = True
                        break
                if not good:
                    raise TypeError(f"Variable {n} is not any of {[x.__name__ for x in t]}")
                continue
            if not isinstance(v, t):
                raise TypeError(f"Variable {n} is not a {t.__name__}")

class EarlyStopCallback(CulsCallback):
    def __init__(self, total_epochs, metric, objective, patience, warmup=0, path=None):
        self.init_type_check(
            [total_epochs, metric, objective, patience, warmup, path],
            [int, str, str, int, int, [str, Path]],
            ["total_epochs", "metric", "objective", "patience", "warmup", "path"])
        self.total_epochs = total_epochs
        self.metric = metric
        self.objective = objective
        self.patience = patience
        self.best_value = None
        self.wait = 0
        self.warmup = warmup
        if path is None:
            self.path = None
        else:
            self.path = Path(path)
        self.init_dict()
        
    
    def init_dict(self):
        self.dict = EasyDict({
            "metric": self.metric,
            "objective": self.objective,
            "patience": self.patience,
            "warmup": self.warmup,
        })
            
        
    def on_epoch_done(self, cfg, checkpoint, epoch, train_predictions, train_metrics, val_predictions, val_metrics, logger):
        if self.warmup >= epoch:
            self.save_dict()
            return False
        if not self.metric in val_metrics:
            logger.warning(f"EarlyStopCallback: Metric {self.metric} not found in metrics.")
            self.save_dict()
            return False
        metric = val_metrics[self.metric]
        if self.best_value is None:
            logger.debug(f"EarlyStopCallback: New metric {self.metric} with value {metric:.2f}.")
            self.update_best(checkpoint, epoch, val_predictions, val_metrics)
            self.save_dict()
            return False
        if self.is_metric_better(metric):
            self.update_best(checkpoint, epoch, val_predictions, val_metrics)
            logger.info(f"EarlyStopCallback: New best metric value {metric:.2f}.")
            self.save_dict()
            return False
        self.wait += 1
        stop = False
        if (self.wait >= self.patience):
            logger.info(f"EarlyStopCallback: Patience reached, restoring checkpoint from epoch {self.best_epoch}.")
            stop = True
        if epoch == self.total_epochs and not self.is_metric_better(metric):
            logger.info(f"EarlyStopCallback: Maximum initial epochs reached, restoring checkpoint from epoch {self.best_epoch}")
            stop = True
        if stop:
            checkpoint.clear()
            checkpoint.update(self.best_checkpoint)
            self.save_dict()
            return True
        logger.debug(f"EarlyStopCallback: No improvement. Waiting for {self.wait} epochs.")
        self.save_dict()
        return False
    
    def is_metric_better(self, metric):
        if self.objective == "min" and self.best_value > metric:
            return True
        if self.objective == "max" and self.best_value < metric:
            return True
        return False
    
    def update_best(self, checkpoint, epoch, predictions, metrics):
        self.best_value = metrics[self.metric]
        self.best_metrics = metrics
        self.best_predictions = predictions
        self.best_epoch = epoch
        self.best_checkpoint = checkpoint
        self.wait = 0
        self.dict.best_epoch = epoch
        self.dict.best_value = self.best_value
        
    def save_dict(self):
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "early_stop.pkl").touch()
        with open(self.path / "early_stop.pkl", "wb+") as fp:
            pickle.dump(self.dict, fp)
  
      
class PickleMetricsCallback(CulsCallback):
    def __init__(self, path):
        self.path = Path(path)
        self.output = {
            "train": {},
            "val": {}
        }
        
    def on_epoch_done(self, cfg, checkpoint, epoch, train_predictions, train_metrics, val_predictions, val_metrics, logger):
        self.output['train'][epoch] = train_metrics
        self.output['val'][epoch] = val_metrics
        self.save_file()
        
    def on_train_done(self, cfg, checkpoint, epoch, predictions, metrics, logger):
        self.output['test'] = metrics
        self.save_file()
        
    def save_file(self):
        with open(self.path / "metrics.pkl", "wb+") as fp:
            pickle.dump(self.output, fp)

class CsvMetricsCallback(CulsCallback):
    def __init__(self, path, sep=",", append=True, configs=None):
        self.path = Path(path)
        self.written_epoch = False
        self.written_test = False
        self.sep = sep
        self.append = append
        
    def on_epoch_done(self, cfg, checkpoint, epoch, train_predictions, train_metrics, val_predictions, val_metrics, logger):
        self.update_file(epoch, train_metrics, "log_train.csv", True, self.written_epoch)
        self.update_file(epoch, val_metrics, "log_val.csv", True, self.written_epoch)
        self.written_epoch = True
    
    def on_train_done(self, cfg, checkpoint, epoch, predictions, metrics, logger):
        self.update_file(epoch, metrics, "log_test.csv", False, self.written_test)
        self.written_test = True
    
    def update_file(self, epoch, metrics, filename, include_epoch, written):
        if not written:
            self.init_file(metrics, include_epoch, filename)
        with open(self.path / filename, "a") as fp:
            formatted_values = [("n/a" if x is None else f"{x:.5f}") for x in metrics.values() if not isinstance(x, dict)]
            if include_epoch:
                line = str(epoch) + self.sep + self.sep.join(formatted_values) + "\n"
            else:
                line = self.sep.join(formatted_values) + "\n"
            fp.write(line)
    
    def init_file(self, metrics, include_epoch, filename):
        header = ""
        if include_epoch:
            header += "epoch" + self.sep
        for k, v in metrics.items():
            if isinstance(v, dict):
                continue
            header += k + self.sep
        header = header[:-1] + "\n"
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / filename).touch()
        if self.append:
            with open(self.path, "r+") as fp:
                lines = fp.readlines()
                if len(lines) == 0:
                    fp.write(header)
                    return
                if lines[0] != header:
                    raise IOError("CsvMetricsCallback: The file already exists and has a different header!")
        else:
            with open(self.path / filename, "w+") as fp:
                fp.write(header)

class SaveLastCheckpointCallback(CulsCallback):
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
    def on_train_done(self, cfg, checkpoint, epoch, predictions, metrics, logger):
        train_utils.save_checkpoint(checkpoint, self.path)


class SaveBestCheckpointsCallback(CulsCallback):
    def __init__(self, metrics_to_use, path, save_checkpoints=True, save_epoch_pkl=True):
        metrics_error_msg = "SaveBestCheckpointsCallback: metrics_to_use is in incorrect format.\nIt must be an iterable of string pairs with metric names and metric objectives, e.g.: [('val_loss', 'min'), ('recall_0.5', 'max')]"
        try:
            # Is this SERIOUSLY the best way to check if an object is iterable?
            # Apparently yes. Python's use of exceptions really is exceptional.
            iter(metrics_to_use)
        except:
            raise TypeError(metrics_error_msg)
        for metric in metrics_to_use:
            if len(metric) != 2:
                raise TypeError(metrics_error_msg)
            if not isinstance(metric[0], str) or not isinstance(metric[1], str):
                raise TypeError(metrics_error_msg)
            if metric[1] not in ["min", "max"]:
                raise TypeError(metrics_error_msg)
        self.metrics_cfg = [{"name": x[0], "goal": x[1]} for x in metrics_to_use]
        self.best_values = {x[0]: {"val": None, "epoch": None} for x in metrics_to_use}
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save_checkpoints = save_checkpoints
        self.save_epoch_pkl = save_epoch_pkl
        
    def on_epoch_done(self, cfg, checkpoint, epoch, train_predictions, train_metrics, val_predictions, val_metrics, logger):
        for metric_cfg in self.metrics_cfg:
            name = metric_cfg['name']
            if not name in val_metrics:
                continue
            current_value = val_metrics[name]
            if current_value is None:
                continue
            best_value = self.best_values[name]['val']
            if best_value is None:
                self.save_best(name, checkpoint, current_value, epoch)
                continue
            if ((metric_cfg['goal'] == "min" and current_value < best_value) or
                (metric_cfg['goal'] == "max" and current_value > best_value)):
                self.save_best(name, checkpoint, current_value, epoch)
                continue
        if self.save_epoch_pkl:
            with open(self.path / "bests.pkl", 'wb') as f:
                pickle.dump(self.best_values, f)
        return False
    
    def save_best(self, metric_name, checkpoint, metric_value, epoch):
        self.best_values[metric_name]['val'] = metric_value
        self.best_values[metric_name]['epoch'] = epoch
        filename = self.path / f"best_{metric_name}"
        if self.save_checkpoints:
            train_utils.save_checkpoint(checkpoint, filename)
