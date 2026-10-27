"""
Hyperparameter sweep over the DAEDAL low-confidence threshold.

Reproduces Appendix F.1, Table `tab:daedal_ablation` of the paper.
Runs daedal_code_infilling.py for every (initial_gen_length, low_conf) pair
in `sweep_configs`.

Usage:
    python daedal_cal/daedal_sweep.py --data_file data/HumanEval-SingleLineInfilling-Rest.jsonl.gz
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_sweep(args):
    # --- 1. Define the sweep grid ---
    # Each entry below corresponds to one row in Table `tab:daedal_ablation`.
    sweep_configs = [
        {"initial_gen_length": 4,  "low_conf": 0.1},
        {"initial_gen_length": 8,  "low_conf": 0.1},
        {"initial_gen_length": 16, "low_conf": 0.1},
        {"initial_gen_length": 32, "low_conf": 0.1},
        {"initial_gen_length": 4,  "low_conf": 0.2},
        {"initial_gen_length": 8,  "low_conf": 0.2},
        {"initial_gen_length": 16, "low_conf": 0.2},
        {"initial_gen_length": 32, "low_conf": 0.2},
        {"initial_gen_length": 4,  "low_conf": 0.3},
        {"initial_gen_length": 8,  "low_conf": 0.3},
        {"initial_gen_length": 16, "low_conf": 0.3},
        {"initial_gen_length": 32, "low_conf": 0.3},
        {"initial_gen_length": 4,  "low_conf": 0.4},
        {"initial_gen_length": 8,  "low_conf": 0.4},
        {"initial_gen_length": 16, "low_conf": 0.4},
        {"initial_gen_length": 32, "low_conf": 0.4},
        {"initial_gen_length": 4,  "low_conf": 0.5},
        {"initial_gen_length": 8,  "low_conf": 0.5},
        {"initial_gen_length": 16, "low_conf": 0.5},
        {"initial_gen_length": 32, "low_conf": 0.5},
    ]

    # --- 2. GPU configuration ---
    # Set this to the IDs of the GPUs you want to use, e.g. ["0", "1"].
    gpu_list = args.gpu_ids

    print(f"Starting DAEDAL sweep on GPUs: {' '.join(gpu_list)}")

    # --- 3. Run experiments ---
    for i, config in enumerate(sweep_configs):
        print(f"\n>>> [{i + 1}/{len(sweep_configs)}] Config: {config}")

        cmd = [
            sys.executable, str(Path(__file__).with_name("daedal_code_infilling.py")),
            "--data_file", args.data_file,
            "--benchmark_name", args.benchmark_name,
            "--gpu_ids", *gpu_list,
        ]
        for param, value in config.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{param}")
            else:
                cmd.extend([f"--{param}", str(value)])

        subprocess.run(cmd, check=True)

    print("\nAll sweep runs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--benchmark_name", default="single-line")
    parser.add_argument("--gpu_ids", nargs="+", default=["0"])
    run_sweep(parser.parse_args())
