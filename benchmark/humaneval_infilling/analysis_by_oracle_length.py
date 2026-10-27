"""
Stratified analysis of code infilling results by Oracle length.

For each (task_type, L_init), compares Fixed vs CAL Pass@1 stratified by
Oracle-length bins for the extended code-infilling analysis.

By default, reads result JSONL files from
    <repo_root>/llada_cal/code_results/
which is where llada_cal/llada_code_infilling.py writes its outputs.

Override with --base_dir if your results live elsewhere, and update
FILE_PAIRS below with relative paths such as <run>/samples_results.jsonl.
"""

import argparse
import json
import os
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BASE_DIR = os.path.join(REPO_ROOT, "llada_cal", "code_results")

# ── File mapping: (task, L_init) -> (fixed_file, cal_file) ──────────────
# Replace these placeholders with <run>/samples_results.jsonl paths.
FILE_PAIRS = {
    ("single", 4):  ("FIXED_SINGLE_L4_TS",  "CAL_SINGLE_L4_TS"),
    ("single", 8):  ("FIXED_SINGLE_L8_TS",  "CAL_SINGLE_L8_TS"),
    ("single", 16): ("FIXED_SINGLE_L16_TS", "CAL_SINGLE_L16_TS"),
    ("single", 32): ("FIXED_SINGLE_L32_TS", "CAL_SINGLE_L32_TS"),
    ("multi", 4):   ("FIXED_MULTI_L4_TS",   "CAL_MULTI_L4_TS"),
    ("multi", 8):   ("FIXED_MULTI_L8_TS",   "CAL_MULTI_L8_TS"),
    ("multi", 16):  ("FIXED_MULTI_L16_TS",  "CAL_MULTI_L16_TS"),
    ("multi", 32):  ("FIXED_MULTI_L32_TS",  "CAL_MULTI_L32_TS"),
}

# ── Bin definitions ─────────────────────────────────────────────────────
BINS_SINGLE = [(1, 5), (6, 10), (11, 15), (16, 44)]
BINS_MULTI  = [(1, 15), (16, 30), (31, 50), (51, 80), (81, 128), (129, 999)]


def load_results(base_dir, filename):
    """Load results JSONL, return list of dicts."""
    path = os.path.join(base_dir, filename)
    with open(path) as f:
        return [json.loads(line) for line in f]


def bin_label(lo, hi):
    if hi >= 999:
        return f"{lo}+"
    return f"{lo}-{hi}"


def assign_bin(true_len, bins):
    for lo, hi in bins:
        if lo <= true_len <= hi:
            return (lo, hi)
    return None


def stratified_analysis(task, l_init, bins, base_dir):
    """Compare Fixed vs CAL Pass@1 stratified by Oracle length bins."""
    fixed_file, cal_file = FILE_PAIRS[(task, l_init)]
    fixed_data = load_results(base_dir, fixed_file)
    cal_data = load_results(base_dir, cal_file)
    assert len(fixed_data) == len(cal_data)

    # Accumulate per-bin stats
    stats = {b: {"n": 0,
                 "fixed_pass": 0, "cal_pass": 0,
                 "fixed_lenerr_sum": 0, "cal_lenerr_sum": 0}
             for b in bins}

    for fd, cd in zip(fixed_data, cal_data):
        assert fd["task_id"] == cd["task_id"]
        true_len = fd["true_len"]
        b = assign_bin(true_len, bins)
        if b is None:
            continue
        s = stats[b]
        s["n"] += 1
        s["fixed_pass"] += int(fd["passed"])
        s["cal_pass"] += int(cd["passed"])
        s["fixed_lenerr_sum"] += fd["len_error"]
        s["cal_lenerr_sum"] += cd["len_error"]

    return stats


def print_table(task, l_init, bins, stats):
    """Print a formatted table for one (task, L_init) setting."""
    header = (f"{'Oracle Len':>12} | {'N':>5} | "
              f"{'Fixed Pass@1':>12} | {'CAL Pass@1':>12} | {'Δ':>7} | "
              f"{'Fixed LenErr':>12} | {'CAL LenErr':>12}")
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"  Task: {task}-line  |  L_init = {l_init}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    total_n = 0
    total_fp = 0
    total_cp = 0

    for b in bins:
        s = stats[b]
        n = s["n"]
        if n == 0:
            continue
        total_n += n
        fp = s["fixed_pass"] / n * 100
        cp = s["cal_pass"] / n * 100
        total_fp += s["fixed_pass"]
        total_cp += s["cal_pass"]
        delta = cp - fp
        fle = s["fixed_lenerr_sum"] / n
        cle = s["cal_lenerr_sum"] / n
        label = bin_label(*b)
        print(f"{label:>12} | {n:>5} | {fp:>11.1f}% | {cp:>11.1f}% | "
              f"{delta:>+6.1f} | {fle:>12.2f} | {cle:>12.2f}")

    # Total row
    if total_n > 0:
        fp_all = total_fp / total_n * 100
        cp_all = total_cp / total_n * 100
        print(sep)
        print(f"{'Total':>12} | {total_n:>5} | {fp_all:>11.1f}% | "
              f"{cp_all:>11.1f}% | {cp_all - fp_all:>+6.1f} |")


def print_markdown_table(task, l_init, bins, stats):
    """Print a markdown-formatted table for one (task, L_init) setting."""
    print(f"\n**{task.capitalize()}-line, L_init = {l_init}**\n")
    print("| Oracle Len | N | Fixed Pass@1 | CAL Pass@1 | Δ | Fixed LenErr | CAL LenErr |")
    print("|------------|---|-------------|-----------|---|-------------|-----------|")

    total_n = total_fp = total_cp = 0
    for b in bins:
        s = stats[b]
        n = s["n"]
        if n == 0:
            continue
        total_n += n
        fp = s["fixed_pass"] / n * 100
        cp = s["cal_pass"] / n * 100
        total_fp += s["fixed_pass"]
        total_cp += s["cal_pass"]
        delta = cp - fp
        fle = s["fixed_lenerr_sum"] / n
        cle = s["cal_lenerr_sum"] / n
        label = bin_label(*b)
        print(f"| {label} | {n} | {fp:.1f}% | {cp:.1f}% | "
              f"{delta:+.1f} | {fle:.2f} | {cle:.2f} |")

    if total_n > 0:
        fp_all = total_fp / total_n * 100
        cp_all = total_cp / total_n * 100
        print(f"| **Total** | {total_n} | {fp_all:.1f}% | {cp_all:.1f}% | "
              f"{cp_all - fp_all:+.1f} | — | — |")


def collect_all_rows(base_dir):
    """Run all analyses and return a list of flat row dicts for TSV output."""
    rows = []
    for task, bins in [("single", BINS_SINGLE), ("multi", BINS_MULTI)]:
        for l_init in [4, 8, 16, 32]:
            stats = stratified_analysis(task, l_init, bins, base_dir)
            total_n = total_fp = total_cp = 0
            for b in bins:
                s = stats[b]
                n = s["n"]
                if n == 0:
                    continue
                total_n += n
                fp = s["fixed_pass"] / n * 100
                cp = s["cal_pass"] / n * 100
                total_fp += s["fixed_pass"]
                total_cp += s["cal_pass"]
                fle = s["fixed_lenerr_sum"] / n
                cle = s["cal_lenerr_sum"] / n
                rows.append({
                    "Task": f"{task}-line",
                    "L_init": l_init,
                    "Oracle_Len": bin_label(*b),
                    "N": n,
                    "Fixed_Pass@1": f"{fp:.1f}%",
                    "CAL_Pass@1": f"{cp:.1f}%",
                    "Delta": f"{cp - fp:+.1f}",
                    "Fixed_LenErr": f"{fle:.2f}",
                    "CAL_LenErr": f"{cle:.2f}",
                })
            # Total row
            if total_n > 0:
                fp_all = total_fp / total_n * 100
                cp_all = total_cp / total_n * 100
                rows.append({
                    "Task": f"{task}-line",
                    "L_init": l_init,
                    "Oracle_Len": "Total",
                    "N": total_n,
                    "Fixed_Pass@1": f"{fp_all:.1f}%",
                    "CAL_Pass@1": f"{cp_all:.1f}%",
                    "Delta": f"{cp_all - fp_all:+.1f}",
                    "Fixed_LenErr": "",
                    "CAL_LenErr": "",
                })
    return rows


def write_tsv(rows, path):
    """Write rows to a TSV file."""
    cols = ["Task", "L_init", "Oracle_Len", "N",
            "Fixed_Pass@1", "CAL_Pass@1", "Delta",
            "Fixed_LenErr", "CAL_LenErr"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\nTSV saved to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR,
                        help="Directory containing the result JSONL files.")
    parser.add_argument("--markdown", action="store_true",
                        help="Output in markdown format")
    parser.add_argument("--output", "-o", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "stratified_by_oracle_length.tsv"),
                        help="Output TSV path")
    args = parser.parse_args()

    printer = print_markdown_table if args.markdown else print_table

    print("=" * 60)
    print("  Stratified Analysis by Oracle Length")
    print("  Model: LLaDA-8B-Base  |  Benchmark: HumanEval-Infilling")
    print(f"  Reading results from: {args.base_dir}")
    print("=" * 60)

    # Single-line
    for l_init in [4, 8, 16, 32]:
        bins = BINS_SINGLE
        stats = stratified_analysis("single", l_init, bins, args.base_dir)
        printer("single", l_init, bins, stats)

    # Multi-line
    for l_init in [4, 8, 16, 32]:
        bins = BINS_MULTI
        stats = stratified_analysis("multi", l_init, bins, args.base_dir)
        printer("multi", l_init, bins, stats)

    # Write TSV
    rows = collect_all_rows(args.base_dir)
    write_tsv(rows, args.output)


if __name__ == "__main__":
    main()
