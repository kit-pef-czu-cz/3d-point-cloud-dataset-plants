import argparse, logging, pickle, warnings
import scipy, torch
import culs_defs_eval as culs_eval
import culs_defs_loading as culs_loading
from pcdet.models import build_network
from pathlib import Path
from easydict import EasyDict

def main(args):
    warnings.filterwarnings("ignore", category=UserWarning) # PyTorch
    warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning)
    torch.no_grad()
    torch.cuda.empty_cache()
    
    logger = culs_loading.create_console_logger()
    intro_print(args, logger)
    cfg = culs_loading.load_config(args.cfg, 0)
    
    if args.data is not None:
        cfg.DATA_CONFIG.DATA_PATH = args.data
    
    if args.study:
        with open(args.study, "rb") as fp:
            study = pickle.load(fp)
        culs_loading.load_trial_params(study, cfg)
    
    data = culs_loading.create_loaders(cfg, logger, "test")
    model = culs_loading.build_model(cfg, data)
    model = build_model(cfg, args, data.set)
    predictions, metrics = culs_eval.evaluate_model_scores(cfg, model, data.loader)
    if args.out is not None:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        if not args.no_predictions:
            with open(Path(args.out) / "predictions.pkl", 'wb') as f:
                pickle.dump(predictions, f)
        if not args.no_metrics:
            with open(Path(args.out) / "metrics.pkl", 'wb') as f:
                pickle.dump(metrics, f)
    return metrics, predictions

def print_metrics(metrics):
    keys = list(metrics.keys())
    keys.sort()
    key_length = max(len(x) for x in keys)
    for k in keys:
        if isinstance(metrics[k], dict) or isinstance(metrics[k], list):
            continue
        print(f"{k.ljust(key_length)}: {metrics[k]:.3f}")
        
def print_two_metrics(a, b):
    keys = list(a.keys())
    keys.sort()
    key_length = max(len(x) for x in keys)
    for k in keys:
        if isinstance(a[k], dict) or isinstance(a[k], list):
            continue
        print(f"{k.ljust(key_length)}: {a[k]:.3f} {b[k]:.3f} {'!!!' if a[k] != b[k] else ''}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('out', type=str, default=None)
    parser.add_argument('--data', '-d', type=str, default=None)
    parser.add_argument('--study', '-s', type=str, default=None)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--workers', '-w', type=int, default=1)
    parser.add_argument('--no-metrics', action="store_true")
    parser.add_argument('--no-predictions', action="store_true")
    return parser.parse_args()

def intro_print(args, logger):
    logger.info("Eval script provided arguments")
    logger.info(f"Config path:     {args.cfg}")
    logger.info(f"Checkpoint path: {args.checkpoint}")
    logger.info(f"Output path:     {args.out}")
    logger.info(f"Dataset path:    {args.data}")
    logger.info(f"Study path:      {args.study}")
    logger.info(f"Batch size:      {args.batch_size}")
    logger.info(f"Workers:         {args.workers}")

def build_model(cfg, args, dataset):
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )
    model.load_params_from_file(
        filename=args.checkpoint,
        logger=logging.getLogger(__name__),
        map_location={'cuda:1': 'cuda:0'},
        pre_trained_path=args.checkpoint
    )
    model.cuda()
    return model


if __name__ == "__main__":
    args = parse_args()
    eval_metrics, eval_predictions = main(args)
