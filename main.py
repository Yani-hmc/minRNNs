"""
Training and evaluation script for sequence learning tasks.
Handles configuration loading, model training, checkpointing,
and evaluation on validation and test sets.
"""

import os
import os.path as osp
import pprint
import time
import collections
import collections.abc

from attrdict import AttrDict

# Fix for Python 3.10+ compatibility with older library 
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

import torch
import yaml
from tqdm import tqdm

from arguments import parse_args
from models import Model
from utils import (compute_outs, gen_evalset, get_eval_path, get_loss,
                        get_metric, get_sampler, sample_data)
from utils import RunningAverage, get_logger
from utils import results_path

def main():
    args = parse_args()

    # Retrieve default arguments
    with open(f"configs/default.yaml", "r") as f:
        cfg = AttrDict(yaml.safe_load(f))

    # Retrieve task-specific arguments
    with open(f"configs/{args.task}.yaml", "r") as f:
        task_cfg = AttrDict(yaml.safe_load(f))

    # Override the default arguments with task-level arguments
    for key, val in dict(task_cfg).items():
        cfg[key] = val

    # Override the arguments with custom command-line arguments
    for key, val in vars(args).items():
        if val is not None:
            if key in cfg:
                print(f"Overriding argument {key}: {val}")
            cfg[key] = val

    cfg.root = osp.join(results_path, cfg.task, cfg.expid)

    for key in list(cfg.keys()):
        if cfg[key] == 'None':
            cfg[key] = None
        elif cfg[key] == 'False':
            cfg[key] = False
        elif cfg[key] == 'True':
            cfg[key] = True

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "test":
        eval(cfg, mode='test')

def train(cfg):
    pprint.pp(cfg)

    model = get_model(cfg)
    model.train()

    print(model)

    if osp.exists(cfg.root + "/ckpt.tar"):
        if cfg.check_exists:
            raise FileExistsError(cfg.root)
    else:
        os.makedirs(cfg.root, exist_ok=True)

    with open(osp.join(cfg.root, "config.yaml"), "w") as f:
        yaml.dump(dict(cfg), f)

    torch.manual_seed(cfg.train_seed)
    torch.cuda.manual_seed(cfg.train_seed)

    sampler = get_sampler(cfg, mode='train')

    # Create parameter groups for optimization
    params_with_decay = []
    params_without_decay = []

    for name, param in model.named_parameters():
        if 'to_hidden_and_gate' in name:  # Don't apply weight decay to RNN transitions
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    optimizer_grouped_parameters = [
        {"params": params_with_decay, "weight_decay": cfg.wd},
        {"params": params_without_decay, "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=cfg.lr)

    loss = torch.nn.CrossEntropyLoss()

    logfilename = os.path.join(
        cfg.root, f'train_{time.strftime("%Y%m%d-%H%M")}.log')

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    best_metric = 0

    logger.info(f"Experiment: {cfg.expid}")
    logger.info(
        f"Total number of parameters: {sum(p.numel() for p in model.parameters())}\n")

    optimizer.zero_grad()
    for step in range(cfg.num_steps + 1):
        for _ in range(cfg.accum_iter):
            batch = sample_data(cfg, sampler, mode='train')

            y_pred = model(batch.x)
            outs = compute_outs(cfg, y_pred, batch, loss)

            # Accumulate gradients over multiple iterations [cite: 341]
            (outs.loss / cfg.accum_iter).backward()

            for key, val in outs.items():
                ravg.update(key, val)

        if cfg.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)

        if step % cfg.print_freq == 0:
            line = f"{cfg.model}:{cfg.expid} step {step} "
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train] "
            line += ravg.info()
            logger.info(line)
            ravg.reset()

        optimizer.step()
        optimizer.zero_grad()

        if (step % cfg.save_freq == 0) or (step % cfg.eval_freq == 0):
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            ckpt.best_metric = best_metric

            if (step % cfg.eval_freq == 0):
                val_metric, val_line = eval(cfg, model, mode='val')
                logger.info(val_line + "\n")

                if val_metric >= best_metric:
                    best_metric = val_metric
                    logger.info("Early Stopping!")
                    ckpt.best_metric = best_metric
                    torch.save(ckpt, os.path.join(cfg.root, "best_ckpt.tar"))

                    eval_logfile = os.path.join(cfg.root, f"eval.log")
                    with open(eval_logfile, "w") as f:
                        f.write(val_line)

                model.train()

            torch.save(ckpt, os.path.join(cfg.root, "ckpt.tar"))

    # Final evaluation on test set
    _, test_line = eval(cfg, model, mode='test')
    
    eval_logfile = os.path.join(cfg.root, f"eval.log")
    with open(eval_logfile, "w") as f:
        f.write(test_line)
    logger.info(test_line + "\n")

def eval(cfg, model=None, mode=None):
    if model is None:
        model = get_model(cfg)
        # Fix for PyTorch 2.6+ security settings
        ckpt = torch.load(os.path.join(
            cfg.root, "best_ckpt.tar"), map_location="cuda", weights_only=False)
        model.load_state_dict(ckpt.model)

    model.eval()

    if cfg.task in ['selective_copy', 'parity_check', 'even_pairs', 'cycle_nav', 'bucket_sort', 'majority', 'majority_count', 'missing_duplicate']:
        path, filename = get_eval_path(cfg, mode)
        if not osp.isfile(osp.join(path, filename)):
            print(f"Generating evaluation sets... {mode}")
            gen_evalset(cfg, mode)

        # Fix for PyTorch 2.6+ security settings loading AttrDict data
        eval_batches = torch.load(osp.join(path, filename), weights_only=False)

        def make_loader(batches):
            for data in batches:
                batch = AttrDict()
                for key, val in data.items():
                    batch[key] = val.cuda()
                yield batch

        eval_loader = make_loader(eval_batches)
    else:
        raise ValueError

    ravg = RunningAverage()
    loss = get_loss(cfg)

    with torch.no_grad():
        for batch in tqdm(eval_loader, ascii=True):
            y_pred = model(batch.x)
            outs = compute_outs(cfg, y_pred, batch, loss)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f"Evaluating on {mode}" + "\n"
    line += f"{cfg.model}: {cfg.expid}"
    line += f"[{mode}] "
    line += ravg.info()

    metric = get_metric(cfg, ravg)
    return metric, line

def get_model(cfg):
    rnn_config = {
        'dim': cfg.d_model,
        'expansion_factor': cfg.expand_dim,
        'forget_bias_init_scale': cfg.forget_bias_init_scale,
        'use_coeff_norm': cfg.use_coeff_norm,
        'use_init_hidden_state': cfg.use_init_hidden_state}
    model = Model(
        module=cfg.model,
        num_tokens=cfg.vocab_size,
        d_in=cfg.d_in,
        d_out=cfg.d_out,
        dim=cfg.d_model,
        depth=cfg.num_layers,
        dropout=cfg.dropout,
        ff_mult=cfg.ff_mult,
        conv_kernel_size=cfg.kernel_size,
        enable_conv=(cfg.kernel_size is not None) and (cfg.kernel_size != 0),
        enable_ff=cfg.enable_ff,
        norm_type=cfg.norm_type,
        rnn_config=rnn_config).cuda()
    return model

if __name__ == "__main__":
    main()