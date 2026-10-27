"""Fit model-specific caption length bias on calibration examples only."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from common import add_model_arguments, load_scorer, read_jsonl, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--exclusion-radius", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fit_parameters(grouped):
    import numpy as np
    from scipy.optimize import curve_fit

    def curve(length, a, b, c, d, e):
        return a * np.exp(-b * length) + c * np.exp(-d * length) + e

    lengths = np.array(sorted(grouped), dtype=float)
    means = np.array([mean(grouped[int(length)]) for length in lengths])
    sigma = np.array([1 / np.sqrt(len(grouped[int(length)])) for length in lengths])
    parameters, _ = curve_fit(
        curve, lengths, means, p0=[1.0, 1.8, 0.6, 0.05, 0.3], sigma=sigma,
        bounds=([0, 0, 0, 0, 0], [1, 2, 1, 0.5, 0.5]), maxfev=10000,
    )
    residual = np.sum((means - curve(lengths, *parameters)) ** 2)
    r_squared = 1 - residual / np.sum((means - np.mean(means)) ** 2)
    return dict(zip(("a", "b", "c", "d", "e"), map(float, parameters))), float(r_squared)


def main():
    args = parse_args()
    import torch

    examples = read_jsonl(args.data_file)[:args.num_examples]
    lengths = list(range(args.min_length, args.max_length + 1))
    torch.cuda.set_device(args.device)
    scorer = load_scorer(args)
    grouped = defaultdict(list)
    for index, example in enumerate(examples, 1):
        context = scorer.encode_context(
            args.image_root / example["image_file"], args.instruction,
        )
        for length in lengths:
            if abs(length - example["reference_length"]) <= args.exclusion_radius:
                continue
            score = scorer.score_length(context, example["prefix_ids"],
                                        example["suffix_ids"], length)
            grouped[length].append(score.confidence)
        print(f"[{index}/{len(examples)}] {example['example_id']}", flush=True)
        del context
        torch.cuda.empty_cache()
    parameters, r_squared = fit_parameters(grouped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json({"model": args.model, "instruction": args.instruction,
                "examples": len(examples), "candidate_lengths": lengths,
                "exclusion_radius": args.exclusion_radius,
                "calibration_example_ids": [item["example_id"] for item in examples],
                "fit_pair_count": sum(map(len, grouped.values())),
                "bias_formula": "a*exp(-b*L) + c*exp(-d*L) + e",
                "parameters": parameters, "r_squared_on_length_means": r_squared},
               args.output)
    print(parameters)


if __name__ == "__main__":
    main()
