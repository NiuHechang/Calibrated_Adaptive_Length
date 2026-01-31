"""
Implementation of the Oracle Peak analysis for Diffusion Language Models.
This script evaluates how the first-step denoising confidence $\Phi(L)$ 
varies with the infilling length $L$. (See Section 3: Analysis in the paper).
"""

import torch
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from benchmark.humaneval.human_eval_infilling.data import read_problems


def get_bias(l):
    """
    Length Bias function B(L) fitted via double-exponential decay.
    Parameters are aligned with Appendix A of the paper.
    """
    # Parameters: a=1.00, b=1.77, c=0.56, d=0.06, e=0.24
    a, b, c, d, e = 1.00, 1.77, 0.56, 0.06, 0.24
    return a * np.exp(-b * l) + c * np.exp(-d * l) + e

@torch.no_grad()
def get_conf_at_len(model, prefix_ids, suffix_ids, l, mask_id=126336):
    """
    Computes the average first-step denoising confidence Phi(L).
    See Section 2: Preliminaries for the definition of Phi(L).
    """
    device = model.device
    prefix_len = prefix_ids.shape[1]
    suffix_len = suffix_ids.shape[1]
    
    # Construct the fully masked sequence x_T = [P; [MASK]^L; S]
    x_t = torch.full((1, prefix_len + l + suffix_len), mask_id, dtype=torch.long, device=device)
    x_t[0, :prefix_len] = prefix_ids
    if suffix_ids is not None:
        x_t[0, prefix_len + l:] = suffix_ids
    
    att = torch.ones((1, x_t.shape[1]), dtype=torch.long, device=device)
    
    # First-step prediction (Step 1 of the diffusion process)
    logits = model(x_t, attention_mask=att).logits
    p = torch.softmax(logits, dim=-1)
    
    # Get the max probability for each masked position
    x0_t = torch.argmax(logits, dim=-1)
    x0_p_t = torch.gather(p, dim=-1, index=x0_t.unsqueeze(-1)).squeeze(-1)
    
    # Phi(L): Average confidence over the masked region
    mask_conf = x0_p_t[0, prefix_len : prefix_len + l].mean().item()
    return mask_conf

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Use general device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'GSAI-ML/LLaDA-8B-Base' 
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    problems = read_problems("single-line")
    results = []
    
    # Scan range: offset from Oracle length L* from -15 to +15
    offsets = range(-15, 16)
    
    problem_list = list(problems.values())
    num_samples = min(len(problem_list), 100)
    sampled_problems = random.sample(problem_list, num_samples)
    
    pbar = tqdm(sampled_problems, desc="Analyzing Oracle Peak (Raw vs. Normalized)") 
    for prob in pbar:
        gt_solution = prob['canonical_solution']
        l_oracle = len(tokenizer.encode(gt_solution, add_special_tokens=False))
        
        prefix_ids = tokenizer(prob['prompt'], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        suffix_ids = tokenizer(prob['suffix'], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        for delta in offsets:
            l_curr = l_oracle + delta
            if l_curr <= 0: continue
            
            # Raw confidence Phi(L)
            conf_raw = get_conf_at_len(model, prefix_ids, suffix_ids, l_curr)
            
            # Calibrated confidence: Phi_norm(L) = Phi(L) / B(L)
            bias = get_bias(l_curr)
            conf_norm = conf_raw / bias
            
            results.append({
                "delta": delta,
                "l_curr": l_curr,
                "conf_raw": conf_raw,
                "conf_norm": conf_norm
            })

    # --- Plotting (Optimized for Paper Style) ---
    delta_to_raw = {}
    delta_to_norm = {}
    
    for r in results:
        d = r['delta']
        if d not in delta_to_raw:
            delta_to_raw[d], delta_to_norm[d] = [], []
        delta_to_raw[d].append(r['conf_raw'])
        delta_to_norm[d].append(r['conf_norm'])
    
    sorted_deltas = sorted(delta_to_raw.keys())
    avg_raw = [np.mean(delta_to_raw[d]) for d in sorted_deltas]
    avg_norm = [np.mean(delta_to_norm[d]) for d in sorted_deltas]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
        'font.size': 22,
        'axes.labelsize': 26,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 16,
        'axes.linewidth': 2
    })

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel('Offset from Oracle Length ($\Delta L$)', fontweight='bold')
    ax1.set_ylabel('Raw Confidence', color='#1f77b4', fontweight='bold')
    ax1.plot(sorted_deltas, avg_raw, marker='o', color='#1f77b4', linewidth=3, markersize=10, 
             label='Raw Confidence (Biased)')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Normalized Confidence', color='#d62728', fontweight='bold')
    ax2.plot(sorted_deltas, avg_norm, marker='s', color='#d62728', linewidth=3, markersize=10, 
             label='Normalized Confidence (Corrected)')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    plt.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Oracle Length')
    
    ax1.grid(True, linestyle='--', alpha=0.4)
    plt.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.15)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    # Save to a generic path
    plot_path = "oracle_peak_analysis.pdf"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Figure saved to {plot_path}")
    
    output_path = "oracle_peak_results.jsonl"
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
