"""
Benchmark script evaluating length generalization of trained models on the
majority task by measuring accuracy across increasing sequence lengths.
"""

import torch
import matplotlib.pyplot as plt
import yaml
import os
from main import get_model
from attrdict import AttrDict
from utils import get_sampler, compute_outs

# Allow AttrDict for security-compliant loading
torch.serialization.add_safe_globals([AttrDict])

def run_full_benchmark():
    device = "cuda"
    # The evaluation lengths used in the paper for stress-testing
    lengths = [20, 40, 80, 160, 320, 640]
    
    # Matching the experiment IDs from your training commands
    experiments = [
        "minGRU_Maj_GenBase", 
        "minLSTM_Maj_GenBase", 
        "GRU_Maj_GenBase", 
        "LSTM_Maj_GenBase", 
        "Transformer_Maj_GenBase"]
    
    styles = {
        'minGRU':      {'c': '#1f77b4', 'm': 'o', 'ls': '-',  'label': 'minGRU'},
        'minLSTM':     {'c': '#d62728', 'm': 's', 'ls': '-',  'label': 'minLSTM'},
        'GRU':         {'c': '#17becf', 'm': 'v', 'ls': '--', 'label': 'Standard GRU'},
        'LSTM':        {'c': '#ff7f0e', 'm': '^', 'ls': '--', 'label': 'Standard LSTM'},
        'Transformer': {'c': '#2ca02c', 'm': 'x', 'ls': '-.', 'label': 'Transformer'}}

    plt.figure(figsize=(10, 6))

    for exp_id in experiments:
        model_key = next((k for k in styles if k.lower() in exp_id.lower()), None)
        if not model_key: continue

        config_path = f"results/majority/{exp_id}/config.yaml"
        ckpt_path = f"results/majority/{exp_id}/best_ckpt.tar"
        
        if not os.path.exists(ckpt_path):
            print(f"Skipping {exp_id}: Checkpoint not found.")
            continue

        with open(config_path, 'r') as f:
            cfg = AttrDict(yaml.safe_load(f))
        
        # Initialize and load
        model = get_model(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()

        sampler = get_sampler(cfg, mode='test')
        accuracies = []

        print(f"Benchmarking {model_key}...")
        for L in lengths:
            batch_accs = []
            for _ in range(50): # 50 batches per length for statistical significance
                batch = sampler.sample(batch_size=64, length=L, device=device)
                with torch.no_grad():
                    y_pred = model(batch.x)
                    outs = compute_outs(cfg, y_pred, batch, torch.nn.CrossEntropyLoss())
                    batch_accs.append(outs.acc.item())
            accuracies.append(sum(batch_accs) / len(batch_accs))

        plt.plot(lengths, accuracies, label=styles[model_key]['label'], 
                 color=styles[model_key]['c'], marker=styles[model_key]['m'], 
                 linestyle=styles[model_key]['ls'], linewidth=2)

    # Thresholds and markers
    plt.axvline(x=40, color='gray', linestyle=':', label='Training Limit')
    plt.axhline(y=1/63, color='black', linestyle='--', alpha=0.3, label='Random Chance')
    
    plt.xscale('log', base=2)
    plt.xticks(lengths, [str(l) for l in lengths])
    plt.title("Length Generalization: minRNNs vs Standard Baselines", fontsize=14)
    plt.xlabel("Sequence Length", fontsize=12)
    plt.ylabel("Majority Task Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.2)
    plt.savefig("generalization_comparison.png")
    plt.show()

if __name__ == "__main__":
    run_full_benchmark()