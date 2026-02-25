"""
Depth ablation experiment on the Majority task.
"""

import torch
import matplotlib.pyplot as plt
import yaml
import os
from main import get_model
from attrdict import AttrDict
from utils import get_sampler, compute_outs

# Allow AttrDict objects when loading checkpoints
torch.serialization.add_safe_globals([AttrDict])

def run_depth_ablation():

    device = "cuda"

    # Depth values tested in the ablation
    depths = [1,5]

    # Experiment IDs for each model
    experiments = {
        "minGRU": ["minGRU_depth1","minGRU_depth5"],
        "minLSTM": ["minLSTM_depth1","minLSTM_depth5"],
        "GRU": ["GRU_depth1","GRU_depth5"],
        "LSTM": ["LSTM_depth1","LSTM_depth5"]}

    # Plot styles per model
    styles = {
        'minGRU':  {'c': '#1f77b4', 'm': 'o', 'ls': '-',  'label': 'minGRU'},
        'minLSTM': {'c': '#d62728', 'm': 's', 'ls': '-',  'label': 'minLSTM'},
        'GRU':     {'c': '#17becf', 'm': 'v', 'ls': '--', 'label': 'GRU'},
        'LSTM':    {'c': '#ff7f0e', 'm': '^', 'ls': '--', 'label': 'LSTM'}}

    # Prepare figure
    plt.figure(figsize=(10,6))

    for model_key, exp_list in experiments.items():

        accuracies = []

        print(f"\nEvaluating {model_key}")

        for exp_id in exp_list:

            # Paths to config and checkpoint
            config_path = f"results/majority/{exp_id}/config.yaml"
            ckpt_path = f"results/majority/{exp_id}/best_ckpt.tar"

            # Skip missing experiments
            if not os.path.exists(ckpt_path):
                print(f"Skipping {exp_id}")
                accuracies.append(None)
                continue

            # Load experiment config
            with open(config_path) as f:
                cfg = AttrDict(yaml.safe_load(f))

            # Rebuild model
            model = get_model(cfg).to(device)

            # Load trained weights
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])

            model.eval()

            # Sampler for majority test task
            sampler = get_sampler(cfg, mode='test')

            batch_accs = []

            # Evaluate over multiple batches
            for _ in range(50):

                batch = sampler.sample(batch_size=64, length=128, device=device)

                with torch.no_grad():

                    y_pred = model(batch.x)

                    outs = compute_outs(
                        cfg,
                        y_pred,
                        batch,
                        torch.nn.CrossEntropyLoss())

                    batch_accs.append(outs.acc.item())

            # Mean accuracy for experiment
            acc = sum(batch_accs)/len(batch_accs)

            print(f"{exp_id}: {acc:.4f}")

            accuracies.append(acc)

        # Plot accuracy vs depth
        plt.plot(
            depths,
            accuracies,
            label=styles[model_key]['label'],
            color=styles[model_key]['c'],
            marker=styles[model_key]['m'],
            linestyle=styles[model_key]['ls'],
            linewidth=2)

    # Figure formatting
    plt.title("Depth Ablation on Majority Task")
    plt.xlabel("Number of Layers")
    plt.ylabel("Accuracy")
    plt.xticks(depths)
    plt.ylim(0,1.05)
    plt.grid(True,alpha=0.2)
    plt.legend()
    plt.savefig("depth_ablation_comparison.png")
    plt.show()

run_depth_ablation()