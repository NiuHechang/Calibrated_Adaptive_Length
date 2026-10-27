"""
Functional Form Comparison for Length Bias B(L).

Fits four candidate forms (double-exponential, single-exponential,
single-exponential without offset, and linear) to the observed first-step
confidence on HumanEval-Infilling, computes R^2 / AIC / BIC, and saves the
fitted parameters to JSON for use in downstream Pass@1 ablation.

See Appendix B.1 of the paper.
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from scipy.optimize import curve_fit
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from human_eval_infilling.data import stream_jsonl
from oracle_peak import get_conf_at_len


# ── Four functional forms ───────────────────────────────────────────────────

def bias_double_exp(L, a, b, c, d, e):
    return a * np.exp(-b * L) + c * np.exp(-d * L) + e

def bias_single_exp(L, c, d, e):
    return c * np.exp(-d * L) + e

def bias_single_exp_no_offset(L, c, d):
    return c * np.exp(-d * L)

def bias_linear_raw(L, alpha, beta):
    """Used only for curve_fit; clip to eps at evaluation time."""
    return alpha - beta * L

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot

def compute_aic_bic(y_true, y_pred, k):
    """AIC and BIC under the normal-error / least-squares assumption."""
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    log_lik = -n / 2 * np.log(rss / n)   # up to a constant
    aic = -2 * log_lik + 2 * k
    bic = -2 * log_lik + k * np.log(n)
    return float(aic), float(bic)

def param_ci(pcov, alpha=0.05):
    """Return 95% half-widths from the covariance matrix."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return z * np.sqrt(np.diag(pcov))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", default="data/HumanEval-SingleLineInfilling-Demo.jsonl.gz")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'GSAI-ML/LLaDA-8B-Base'

    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Demo file: 100 problems used for fitting (same as length_bias.py)
    problems = {p["task_id"]: p for p in stream_jsonl(args.data_file)}
    test_lengths = [1, 2, 4, 6, 12, 16, 24, 32, 48, 64, 96, 128]

    # ── Step 1: Collect confidence measurements ──────────────────────────────
    all_data_points = []

    for prob in tqdm(problems.values(), desc="Collecting confidence data"):
        gt_solution = prob['canonical_solution']
        l_oracle = len(tokenizer.encode(gt_solution, add_special_tokens=False))

        prefix_ids = tokenizer(
            prob['prompt'], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        suffix_ids = tokenizer(
            prob['suffix'], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        for l in test_lengths:
            # Exclude lengths near Oracle (same exclusion window as length_bias.py)
            if l_oracle - 4 <= l <= l_oracle + 4:
                continue
            conf = get_conf_at_len(model, prefix_ids, suffix_ids, l)
            all_data_points.append((l, conf))

    # Aggregate by length
    len_to_confs = {}
    for l, conf in all_data_points:
        len_to_confs.setdefault(l, []).append(conf)

    x_data = np.array(sorted(len_to_confs.keys()), dtype=float)
    y_data = np.array([np.mean(len_to_confs[l]) for l in x_data])
    sigmas = np.array([1.0 / np.sqrt(len(len_to_confs[l])) for l in x_data])

    print("\nObserved data (length -> avg confidence):")
    for l, y, n in zip(x_data, y_data, [len(len_to_confs[l]) for l in x_data]):
        print(f"  L={int(l):3d}: {y:.4f}  (n={n})")

    # ── Step 2: Fit four functional forms ───────────────────────────────────

    # Double-exp (paper's form)
    popt_d, pcov_d = curve_fit(
        bias_double_exp, x_data, y_data,
        p0=[1.0, 1.8, 0.6, 0.05, 0.3], sigma=sigmas,
        maxfev=10000, bounds=(0, [1.0, 2.0, 1.0, 0.5, 0.5])
    )
    y_pred_d = bias_double_exp(x_data, *popt_d)
    r2_d = r_squared(y_data, y_pred_d)
    aic_d, bic_d = compute_aic_bic(y_data, y_pred_d, k=5)
    ci_d = param_ci(pcov_d)

    # Single-exp
    popt_s, pcov_s = curve_fit(
        bias_single_exp, x_data, y_data,
        p0=[0.6, 0.05, 0.3], sigma=sigmas,
        maxfev=10000, bounds=(0, [2.0, 0.5, 0.5])
    )
    y_pred_s = bias_single_exp(x_data, *popt_s)
    r2_s = r_squared(y_data, y_pred_s)
    aic_s, bic_s = compute_aic_bic(y_data, y_pred_s, k=3)
    ci_s = param_ci(pcov_s)

    # Single-exp without offset
    popt_sn, pcov_sn = curve_fit(
        bias_single_exp_no_offset, x_data, y_data,
        p0=[0.6, 0.05], sigma=sigmas,
        maxfev=10000, bounds=(0, [2.0, 0.5])
    )
    y_pred_sn = bias_single_exp_no_offset(x_data, *popt_sn)
    r2_sn = r_squared(y_data, y_pred_sn)
    aic_sn, bic_sn = compute_aic_bic(y_data, y_pred_sn, k=2)
    ci_sn = param_ci(pcov_sn)

    # Linear (alpha - beta*L, clipped to eps at evaluation)
    popt_l, pcov_l = curve_fit(
        bias_linear_raw, x_data, y_data,
        p0=[0.8, 0.004], sigma=sigmas,
        maxfev=10000, bounds=([0.0, 0.0], [2.0, 0.1])
    )
    y_pred_l = np.maximum(popt_l[0] - popt_l[1] * x_data, 1e-6)
    r2_l = r_squared(y_data, y_pred_l)
    aic_l, bic_l = compute_aic_bic(y_data, y_pred_l, k=2)
    ci_l = param_ci(pcov_l)

    # Compute Delta-AIC / Delta-BIC relative to best model (double-exp)
    delta_aic = {
        "double_exp": 0.0,
        "single_exp": round(aic_s - aic_d, 2),
        "single_exp_no_offset": round(aic_sn - aic_d, 2),
        "linear": round(aic_l - aic_d, 2),
    }
    delta_bic = {
        "double_exp": 0.0,
        "single_exp": round(bic_s - bic_d, 2),
        "single_exp_no_offset": round(bic_sn - bic_d, 2),
        "linear": round(bic_l - bic_d, 2),
    }

    print("\n-- Fitting results --")
    print(f"Double-exp      : a={popt_d[0]:.4f}+/-{ci_d[0]:.4f}, b={popt_d[1]:.4f}+/-{ci_d[1]:.4f}, "
          f"c={popt_d[2]:.4f}+/-{ci_d[2]:.4f}, d={popt_d[3]:.4f}+/-{ci_d[3]:.4f}, e={popt_d[4]:.4f}+/-{ci_d[4]:.4f}  "
          f"R^2={r2_d:.4f}  AIC={aic_d:.2f}  BIC={bic_d:.2f}  dAIC=0  dBIC=0")
    print(f"Single-exp      : c={popt_s[0]:.4f}+/-{ci_s[0]:.4f}, d={popt_s[1]:.4f}+/-{ci_s[1]:.4f}, "
          f"e={popt_s[2]:.4f}+/-{ci_s[2]:.4f}  "
          f"R^2={r2_s:.4f}  AIC={aic_s:.2f}  BIC={bic_s:.2f}  dAIC={delta_aic['single_exp']}  dBIC={delta_bic['single_exp']}")
    print(f"Single-exp (no offset): c={popt_sn[0]:.4f}+/-{ci_sn[0]:.4f}, d={popt_sn[1]:.4f}+/-{ci_sn[1]:.4f}  "
          f"R^2={r2_sn:.4f}  AIC={aic_sn:.2f}  BIC={bic_sn:.2f}  dAIC={delta_aic['single_exp_no_offset']}  dBIC={delta_bic['single_exp_no_offset']}")
    print(f"Linear          : alpha={popt_l[0]:.4f}+/-{ci_l[0]:.4f}, beta={popt_l[1]:.4f}+/-{ci_l[1]:.4f}  "
          f"R^2={r2_l:.4f}  AIC={aic_l:.2f}  BIC={bic_l:.2f}  dAIC={delta_aic['linear']}  dBIC={delta_bic['linear']}")

    # ── Step 3: Save params JSON for downstream inference ────────────────────
    params = {
        "double_exp": {
            "a": float(popt_d[0]), "a_ci95": float(ci_d[0]),
            "b": float(popt_d[1]), "b_ci95": float(ci_d[1]),
            "c": float(popt_d[2]), "c_ci95": float(ci_d[2]),
            "d": float(popt_d[3]), "d_ci95": float(ci_d[3]),
            "e": float(popt_d[4]), "e_ci95": float(ci_d[4]),
            "r2": float(r2_d), "aic": float(aic_d), "bic": float(bic_d),
            "delta_aic": 0.0, "delta_bic": 0.0,
        },
        "single_exp": {
            "c": float(popt_s[0]), "c_ci95": float(ci_s[0]),
            "d": float(popt_s[1]), "d_ci95": float(ci_s[1]),
            "e": float(popt_s[2]), "e_ci95": float(ci_s[2]),
            "r2": float(r2_s), "aic": float(aic_s), "bic": float(bic_s),
            "delta_aic": delta_aic["single_exp"], "delta_bic": delta_bic["single_exp"],
        },
        "single_exp_no_offset": {
            "c": float(popt_sn[0]), "c_ci95": float(ci_sn[0]),
            "d": float(popt_sn[1]), "d_ci95": float(ci_sn[1]),
            "r2": float(r2_sn), "aic": float(aic_sn), "bic": float(bic_sn),
            "delta_aic": delta_aic["single_exp_no_offset"], "delta_bic": delta_bic["single_exp_no_offset"],
        },
        "linear": {
            "alpha": float(popt_l[0]), "alpha_ci95": float(ci_l[0]),
            "beta":  float(popt_l[1]), "beta_ci95":  float(ci_l[1]),
            "r2": float(r2_l), "aic": float(aic_l), "bic": float(bic_l),
            "delta_aic": delta_aic["linear"], "delta_bic": delta_bic["linear"],
        },
    }

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    params_path = os.path.join(os.path.dirname(__file__), f"bias_compare_params_{timestamp}.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nParams saved to {params_path}")

    # ── Step 4: Plot overlaid curves ─────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
        'font.size': 22, 'axes.labelsize': 26,
        'xtick.labelsize': 20, 'ytick.labelsize': 20,
        'legend.fontsize': 16, 'axes.linewidth': 2
    })
    plt.figure(figsize=(12, 7))

    x_fit = np.linspace(1, 128, 300)

    plt.scatter(x_data, y_data, color='#d62728', s=120, alpha=1.0, zorder=5,
                label='Observed Confidence (Non-Oracle)')
    plt.plot(x_fit, bias_double_exp(x_fit, *popt_d),
             color='#1f77b4', linewidth=3.5,
             label=f'Double-Exp $B(L)$ (R^2={r2_d:.3f})')
    plt.plot(x_fit, bias_single_exp(x_fit, *popt_s),
             color='#2ca02c', linewidth=3.5, linestyle='--',
             label=f'Single-Exp (R^2={r2_s:.3f})')
    plt.plot(x_fit, bias_single_exp_no_offset(x_fit, *popt_sn),
             color='#9467bd', linewidth=3.5, linestyle='-.',
             label=f'Single-Exp (no offset) (R^2={r2_sn:.3f})')
    plt.plot(x_fit, np.maximum(popt_l[0] - popt_l[1] * x_fit, 1e-6),
             color='#ff7f0e', linewidth=3.5, linestyle=':',
             label=f'Linear (R^2={r2_l:.3f})')

    plt.xlabel('Infilling Length ($L$)', fontweight='bold')
    plt.ylabel('Mean First-Step Confidence', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, loc='upper right', facecolor='white', framealpha=0.9)
    plt.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.15)

    plot_path = os.path.join(os.path.dirname(__file__), f"bias_compare_{timestamp}.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
