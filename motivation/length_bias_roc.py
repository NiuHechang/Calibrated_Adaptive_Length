"""
In-domain Length Bias fitting on ROCStories.

Fits the double-exponential B(L) on 1,000 demo samples (ROCStories_demo.jsonl),
which are strictly excluded from the evaluation set. Used to compare with the
code-fitted B(L) and to demonstrate cross-domain consistency of the slow-decay
component d.

Outputs:
  - motivation/bias_params_roc.json       : Fitted parameters (consumed by
    text-infilling scripts when --bias_source=indomain).
  - motivation/length_bias_roc_{ts}.pdf   : Visualization of the fitted curve.

See Appendix B and Appendix E.2 of the paper.
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from scipy.optimize import curve_fit
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from oracle_peak import get_conf_at_len   # Reuse the single-pass confidence routine.


def bias_func(L, a, b, c, d, e):
    return a * np.exp(-b * L) + c * np.exp(-d * L) + e


def main():
    # ── Configuration ────────────────────────────────────────────────────
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name  = 'GSAI-ML/LLaDA-8B-Base'
    demo_path   = 'benchmark/ROCstories/ROCStories_demo.jsonl'
    output_json = 'motivation/bias_params_roc.json'

    # Text spans concentrate in 2-10 tokens; densely sample within that range.
    test_lengths = [1, 2, 4, 6, 8, 12]
    # Smaller exclusion window for short text spans (code uses +/-4).
    oracle_excl_window = 1
    # ─────────────────────────────────────────────────────────────────────

    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load all 1,000 demo samples.
    problems = []
    with open(demo_path) as f:
        for line in f:
            problems.append(json.loads(line))
    print(f"Loaded {len(problems)} demo samples for fitting.")

    all_data_points = []

    pbar = tqdm(problems, desc="ROC In-domain Bias Fitting")
    for prob in pbar:
        # Token length of the ground-truth span.
        l_oracle = len(tokenizer.encode(prob["answer"], add_special_tokens=False))

        prefix_ids = tokenizer(
            prob["prompt"], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        suffix_ids = tokenizer(
            prob["suffix"], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        for l in test_lengths:
            # Exclude lengths near the Oracle to prevent the Oracle Peak
            # from contaminating the bias curve.
            if abs(l - l_oracle) <= oracle_excl_window:
                continue
            conf = get_conf_at_len(model, prefix_ids, suffix_ids, l)
            all_data_points.append((l, conf))

    # Aggregate by length.
    len_to_confs = {}
    for l, conf in all_data_points:
        len_to_confs.setdefault(l, []).append(conf)

    x_data, y_data = [], []
    print("\n-- Average confidence per length --")
    for l in sorted(len_to_confs):
        avg = np.mean(len_to_confs[l])
        x_data.append(l)
        y_data.append(avg)
        print(f"  L={l:3d}: {avg:.4f}  (n={len(len_to_confs[l])})")

    # ── Double-exponential fitting ────────────────────────────────────────
    # Text confidence is generally higher than code (shorter spans);
    # adjust the initial guess accordingly.
    p0     = [1.0, 1.5, 0.6, 0.08, 0.35]
    sigmas = [1.0 / np.sqrt(len(len_to_confs[l])) for l in x_data]

    try:
        popt, _ = curve_fit(
            bias_func, x_data, y_data,
            p0=p0, sigma=sigmas,
            maxfev=10000,
            bounds=(0, [1.5, 3.0, 1.5, 0.5, 0.8])
        )
        a, b, c, d, e_val = popt
        print(f"\nFitted: B(L) = {a:.4f}*exp(-{b:.4f}*L) + {c:.4f}*exp(-{d:.4f}*L) + {e_val:.4f}")

        # Save parameters.
        params = {"a": float(a), "b": float(b), "c": float(c),
                  "d": float(d), "e": float(e_val), "domain": "ROCStories"}
        with open(output_json, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Parameters saved to {output_json}")

        # ── Plot ──────────────────────────────────────────────────────────
        # Overlay the code-fitted curve for comparison.
        def code_bias(l):
            return 1.00 * np.exp(-1.77 * l) + 0.56 * np.exp(-0.06 * l) + 0.24

        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
            'font.size': 22, 'axes.labelsize': 26,
            'xtick.labelsize': 20, 'ytick.labelsize': 20,
            'legend.fontsize': 16, 'axes.linewidth': 2
        })
        plt.figure(figsize=(12, 7))

        plt.scatter(x_data, y_data, color='#d62728', s=100, alpha=1.0,
                    label='Observed Confidence (ROCStories)')

        x_fit = np.linspace(1, 16, 200)
        plt.plot(x_fit, bias_func(x_fit, *popt), color='#1f77b4', linewidth=4,
                 label='In-domain $B(L)$ (ROCStories)')
        plt.plot(x_fit, [code_bias(l) for l in x_fit], color='#ff7f0e',
                 linewidth=3, linestyle='--', label='Code-fitted $B(L)$ (HumanEval)')

        plt.xlabel('Infilling Length ($L$)', fontweight='bold')
        plt.ylabel('Mean First-Step Confidence', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(frameon=True, loc='upper right', facecolor='white', framealpha=0.9)
        plt.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.15)

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        plot_path = f"motivation/length_bias_roc_{timestamp}.pdf"
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Comparison plot saved to {plot_path}")

    except Exception as ex:
        print(f"Fitting failed: {ex}")


if __name__ == "__main__":
    main()
