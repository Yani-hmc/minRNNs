"""
Command-line argument parser for configuring experiments,
including training, evaluation, task selection, and model settings.
"""

import argparse

def add_arguments(parser):
    # Task Specific Arguments
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--num_tokens_memorize", type=int, default=None)
    return parser

def parse_args():
    # Parser for arguments that override the default ones in configs/default.yaml and configs/{task}.yaml
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument("--mode", default="train",
                        choices=[
                            'train',
                            'test',
                        ])
    parser.add_argument("--expid", type=str, default="default")
    parser.add_argument("--check_exists", type=str, default=None)
    parser.add_argument("--task", type=str, default="selective_copy",
                        choices=[
                            'selective_copy',
                            'parity_check',
                            'even_pairs',
                            'cycle_nav',
                            'bucket_sort',
                            'majority',
                            'majority_count',
                            'missing_duplicate',
                        ])

    # Train
    parser.add_argument("--train_seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--accum_iter", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=None)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--print_freq", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=None)

    # Eval
    parser.add_argument("--val_seed", type=int, default=0)
    parser.add_argument("--test_seed", type=int, default=1)
    parser.add_argument("--eval_num_batches", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)

    # Module Specifics
    parser.add_argument("--model", type=str, default='minLSTM',
                        choices=[
                            'minGRU',
                            'minLSTM',
                            'gru',
                            'lstm',
                            'transformer',
                            'mamba'])
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--ff_mult", type=int, default=None)
    parser.add_argument("--expand_dim", type=float, default=None)
    parser.add_argument("--kernel_size", type=int, default=None)
    parser.add_argument("--enable_ff", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--forget_bias_init_scale", type=int, default=None)
    parser.add_argument("--use_coeff_norm", type=str, default=None)
    parser.add_argument("--use_init_hidden_state", type=str, default=None)
    parser.add_argument("--norm_type", type=str, default=None)

    # parse arguments
    args, unknown_args = parser.parse_known_args()

    # get task-specific arguments
    if args.task == 'selective_copy':
        parser = add_arguments(parser)
    else:
        print(f"No specialized Arguments for {args.task}" + "\n\n")

    # parse remaining arguments
    args = parser.parse_args()

    return args
