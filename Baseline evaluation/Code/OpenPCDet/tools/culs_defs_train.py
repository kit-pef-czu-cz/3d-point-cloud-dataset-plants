import torch, time, os, pickle
import numpy as np
from dataclasses import dataclass
import pcdet.models
import culs_defs_callbacks as callbacks
import train_utils.train_utils as train_utils


def train_model(cfg, model, optimizer, lr_normal_scheduler, lr_warmup_scheduler,
          logger, train_data, val_data=None, test_data=None,
          train_eval_func=None, val_eval_func=None, test_eval_func=None,
          epoch_callbacks=None, train_callbacks=None):
    if epoch_callbacks is None:
        epoch_callbacks = []
    if train_callbacks is None:
        train_callbacks = []
    type_check(callbacks.CulsCallback, epoch_callbacks)
    type_check(callbacks.CulsCallback, train_callbacks)
    train_func = pcdet.models.model_fn_decorator() # I have no clue
    accumulated_iter = 0
    # predictions, metrics = [], []
    train_metrics, val_metrics = [], []
    for current_epoch in range(cfg.OPTIMIZATION.NUM_EPOCHS):
        # set-up
        shifted_epoch = current_epoch + 1
        start_time = time.perf_counter()
        if train_data.sampler is not None:
            train_data.sampler.set_epoch(current_epoch)
        lr_scheduler = choose_scheduler(
            lr_normal_scheduler, lr_warmup_scheduler, cfg, current_epoch)
        
        # train and train loss
        accumulated_iter, train_loss = train_epoch(
            model, optimizer, train_data.loader, train_func, lr_scheduler,
            accumulated_iter, cfg, current_epoch)
        checkpoint = train_utils.checkpoint_state(model, optimizer, current_epoch, accumulated_iter)
        tr_time = time.perf_counter()
        tr_dur = tr_time - start_time
        
        # val eval
        if train_eval_func:
            epoch_train_result, epoch_train_metrics = train_eval_func(cfg, model, train_data.loader)
            epoch_train_metrics["epoch_time"] = tr_dur
        else:
            epoch_train_result, epoch_train_metrics = {}, {}
        if val_eval_func:
            epoch_val_result, epoch_val_metrics = val_eval_func(cfg, model, val_data.loader)
            epoch_val_metrics["epoch_time"] = tr_dur
        else:
            epoch_val_result, epoch_val_metrics = {}, {}
        
        ev_time = time.perf_counter()
        ev_dur = ev_time - tr_time
            
        # predictions.append(epoch_result)
        train_metrics.append(epoch_train_metrics)
        val_metrics.append(epoch_val_metrics)
        stop_early = False
        for cb in epoch_callbacks:
            if cb.on_epoch_done(cfg, checkpoint, shifted_epoch,
                                epoch_train_result, epoch_train_metrics,
                                epoch_val_result, epoch_val_metrics, logger):
                stop_early = True
        cb_time = time.perf_counter()
        cb_dur = cb_time - ev_time
        
        logger.info(f"Epoch {str(shifted_epoch).zfill(3)}  " \
                    "Time: " \
                    f"{tr_dur:.1f}/{ev_dur:.1f}/{cb_dur:.1f} ({tr_dur+ev_dur+cb_dur:.1f})")
        if stop_early:
            break
    if test_eval_func:
        test_result, test_metrics = test_eval_func(cfg, model, test_data.loader)
    else:
        test_result, test_metrics = {}, {}
    for cb in train_callbacks:
        cb.on_train_done(cfg, checkpoint, shifted_epoch, test_result, test_metrics, logger)
    return {
        "checkpoint": checkpoint,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }


def type_check(parent, children):
    if not isinstance(children, list):
        raise TypeError(f"{type(children).__name__} is not a list.")
    for child in children:
        if not isinstance(child, parent):
            raise TypeError(f"{type(child).__name__} is not a {parent.__name__} type.")


def choose_scheduler(normal_lr, warmup_lr, cfg, epoch):
    if warmup_lr is not None and epoch < cfg.OPTIMIZATION.WARMUP_EPOCH:
        return warmup_lr
    else:
        return normal_lr
    
    
def train_epoch(model, optimizer, train_loader, model_func, lr_scheduler,
                accumulated_iter, cfg, cur_epoch):
    model.train()
    total_it_each_epoch = len(train_loader)
    dataloader_iter = iter(train_loader)
    optim_cfg = cfg.OPTIMIZATION
    start_it = accumulated_iter % total_it_each_epoch
    scaler = torch.cuda.amp.GradScaler(
        enabled=False, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))
    losses = []
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        lr_scheduler.step(accumulated_iter, cur_epoch)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            loss, tb_dict, disp_dict = model_func(model, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        accumulated_iter += 1
    return accumulated_iter, np.average(losses)

