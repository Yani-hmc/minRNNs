"""
Utility functions for training and evaluation.
Includes logging, running averages, data sampling,
evaluation set generation, loss/metric handling,
and helper functions for sequence modeling tasks.
"""

import os
import os.path as osp
import random
import time
import logging
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import samplers
from attrdict import AttrDict

ROOT = '.'
eval_data_dir = osp.join(ROOT, 'eval_datasets')
results_path = osp.join(ROOT, 'results')


# Associative scan

def associative_scan_log(log_coeffs, log_values, return_log=False):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star

    if return_log:
        return log_h
    else:
        return log_h.exp()


# Helpers

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


# Misc

def exists(v):
    return v is not None


# Log

def get_logger(filename, mode='a'):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger

class RunningAverage(object):
    def __init__(self, *keys):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()
        for key in keys:
            self.sum[key] = 0
            self.cnt[key] = 0

    def update(self, key, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if self.sum.get(key, None) is None:
            self.sum[key] = val
            self.cnt[key] = 1
        else:
            self.sum[key] = self.sum[key] + val
            self.cnt[key] += 1

    def reset(self):
        for key in self.sum.keys():
            self.sum[key] = 0
            self.cnt[key] = 0
        self.clock = time.time()

    def clear(self):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()

    def keys(self):
        return self.sum.keys()

    def get(self, key):
        assert (self.sum.get(key, None) is not None)
        return self.sum[key] / self.cnt[key]

    def info(self, show_et=True):
        line = ''
        for key in self.sum.keys():
            val = self.sum[key] / self.cnt[key]
            if type(val) == float:
                line += f'{key} {val:.4f} '
            else:
                line += f'{key} {val} '.format(key, val)
        if show_et:
            line += f'({time.time()-self.clock:.3f} secs)'
        return line


# Data

def get_sampler(cfg: AttrDict, mode: str):
    if cfg.task == 'selective_copy':
        sampler = samplers[cfg.task](
            sequence_length=cfg.sequence_length,
            num_tokens_memorize=cfg.num_tokens_memorize,
            seed=cfg[f'{mode}_seed']
        )
    elif cfg.task in ['parity_check', 'even_pairs', 'cycle_nav', 'bucket_sort', 'majority', 'majority_count', 'missing_duplicate']:
        sampler = samplers[cfg.task](
            seed=cfg[f'{mode}_seed'],
        )
    else:
        raise ValueError
    return sampler

def get_eval_path(cfg: AttrDict, mode: str):
    path = osp.join(eval_data_dir, cfg.task)

    if cfg.task == 'selective_copy':
        filename = f"seq_len-{cfg.sequence_length}_"
    else:
        filename = ""

    seed = cfg[f'{mode}_seed']
    filename += f"seed-{seed}_{mode}"
    filename += ".tar"
    return path, filename

def get_loss(cfg):
    loss_types = {
        'selective_copy': torch.nn.CrossEntropyLoss(),
        'parity_check': torch.nn.CrossEntropyLoss(),
        'even_pairs': torch.nn.CrossEntropyLoss(),
        'cycle_nav': torch.nn.CrossEntropyLoss(),
        'bucket_sort': torch.nn.CrossEntropyLoss(),
        'majority': torch.nn.CrossEntropyLoss(),
        'majority_count': torch.nn.CrossEntropyLoss(),
        'missing_duplicate': torch.nn.CrossEntropyLoss(),
    }
    return loss_types[cfg.task]

def get_metric(cfg, ravg):
    """
        ravg: RunningAverage (see utils/log.py)
    """
    metric_names = {
        'selective_copy': 'acc',
        'parity_check': 'acc',
        'even_pairs': 'acc',
        'cycle_nav': 'acc',
        'bucket_sort': 'acc',
        'majority': 'acc',
        'majority_count': 'acc',
        'missing_duplicate': 'acc',
    }
    metric_name = metric_names[cfg.task]

    metric = ravg.get(
        metric_name
    )
    return metric

def gen_evalset(cfg: AttrDict, mode: str):
    print(f"Generating Evaluation Sets")

    sampler = get_sampler(cfg, mode=mode)
    batches = []
    for i in tqdm(range(cfg.eval_num_batches), ascii=True):
        batches.append(sample_data(cfg, sampler, mode))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(cfg, mode)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))

def compute_outs(cfg, y_pred, batch, loss):
    outs = AttrDict()
    if cfg.task == 'selective_copy':
        y_pred = y_pred[:, -cfg.num_tokens_memorize:]
    elif cfg.task in ['parity_check', 'even_pairs', 'cycle_nav', 'majority', 'majority_count', 'missing_duplicate']:
        y_pred = y_pred[:, -1:]
    elif cfg.task in ['bucket_sort']:
        length = y_pred.shape[1] // 2
        y_pred = y_pred[:, -length:]
    else:
        raise ValueError

    y_pred = y_pred.flatten(0, 1)
    y_target = batch.y.flatten(0, 1)
    outs.loss = loss(y_pred, y_target)
    outs.acc = (y_pred.argmax(-1) == y_target).float().mean()

    return outs

def sample_data(cfg, sampler, mode, device='cuda'):
    if mode == 'train':
        batch_size = cfg.train_batch_size // cfg.accum_iter
    else:  # Eval
        batch_size = cfg.eval_batch_size

    if cfg.task == 'selective_copy':
        batch = sampler.sample(batch_size=batch_size, device=device)
    elif cfg.task in ['parity_check', 'even_pairs', 'cycle_nav', 'bucket_sort', 'majority', 'majority_count', 'missing_duplicate']:
        if mode == 'train':  # Train
            length = random.randint(1, 40)
        else:  # Eval
            length = random.randint(40, 256)  # following xLSTM's setup

        batch = sampler.sample(batch_size=batch_size,
                               length=length, device=device)
    else:
        raise ValueError
    return batch
