"""
Length Bias Calibration: Fitting the systematic confidence decay B(L).
This script implements the double-exponential fitting used to decouple 
length bias from semantic signals. (See Appendix A of the paper).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from transformers import AutoTokenizer, AutoModel
from benchmark.humaneval.human_eval_infilling.data import read_problems

# Import shared confidence calculation function
from oracle_peak import get_conf_at_len


def bias_func(L, a, b, c, d, e):
    """
    Double-exponential decay model: B(L) = a*exp(-b*L) + c*exp(-d*L) + e.
    Designed to capture both rapid local dilution and slow asymptotic stabilization.
    """
    return a * np.exp(-b * L) + c * np.exp(-d * L) + e

def main():
    # Use general device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'GSAI-ML/LLaDA-8B-Base'
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    problems = read_problems("demo")
    test_lengths = [1, 2, 4, 6, 12, 16, 24, 32, 48, 64, 96, 128]
    
    all_data_points = []
    problem_list = list(problems.values())
    
    pbar = tqdm(problem_list, desc="Fitting Length Bias (Oracle-Excluded)")
    for prob in pbar:
        # Identify Oracle length for exclusion to avoid semantic contamination
        gt_solution = prob['canonical_solution']
        l_oracle = len(tokenizer.encode(gt_solution, add_special_tokens=False))
        
        prefix_ids = tokenizer(prob['prompt'], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        suffix_ids = tokenizer(prob['suffix'], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        for l in test_lengths:
            # Oracle Exclusion Strategy: Discard points near the ground truth
            if l_oracle - 4 <= l <= l_oracle + 4:
                continue
                
            conf = get_conf_at_len(model, prefix_ids, suffix_ids, l)
            all_data_points.append((l, conf))

    # Aggregate results for fitting
    len_to_confs_final = {}
    for l, conf in all_data_points:
        if l not in len_to_confs_final:
            len_to_confs_final[l] = []
        len_to_confs_final[l].append(conf)
    
    x_data = []
    y_data = []
    for l in sorted(len_to_confs_final.keys()):
        avg_conf = np.mean(len_to_confs_final[l])
        x_data.append(l)
        y_data.append(avg_conf)
        print(f"Length {l}: Avg Conf = {avg_conf:.4f} (Samples: {len(len_to_confs_final[l])})")

    # Perform robust non-linear least squares fitting
    try:
        # Initial guess: [a, b, c, d, e]
        p0 = [1.0, 1.8, 0.6, 0.05, 0.3]
        
        # Weighted fitting based on sample size
        sigmas = [1.0 / np.sqrt(len(len_to_confs_final[l])) for l in x_data]
        
        popt, _ = curve_fit(
            bias_func, 
            x_data, 
            y_data, 
            p0=p0, 
            sigma=sigmas,
            maxfev=10000, 
            bounds=(0, [1.0, 2.0, 1.0, 0.5, 0.5])
        )
        
        a, b, c, d, e_val = popt
        print("\n" + "="*30)
        print("Fitting Success! Parameters for B(L):")
        print(f"B(L) = {a:.4f}*exp(-{b:.4f}*L) + {c:.4f}*exp(-{d:.4f}*L) + {e_val:.4f}")
        print("="*30)
        
        # --- Plotting (Optimized for Paper Style) ---
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

        plt.figure(figsize=(12, 7))
        
        # Data points (semantic-agnostic)
        plt.scatter(x_data, y_data, color='#d62728', s=100, alpha=1.0, label='Agnostic Data (Oracle Excluded)')
        
        # Fitted curve
        x_fit = np.linspace(1, 128, 100)
        y_fit = bias_func(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='#1f77b4', linewidth=4, label=f'Fitted Length Bias Trend')
        
        plt.xlabel('Generation Length ($L$)', fontweight='bold')
        plt.ylabel('Mean First-Step Confidence', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(frameon=True, loc='upper right', facecolor='white', framealpha=0.9)
        
        plt.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.15)
        
        plot_path = "length_bias_calibration.pdf"
        plt.savefig(plot_path, bbox_inches='tight') 
        print(f"Figure saved to {plot_path}")
        
    except Exception as e:
        print(f"Fitting Failed: {e}")

if __name__ == "__main__":
    main()
