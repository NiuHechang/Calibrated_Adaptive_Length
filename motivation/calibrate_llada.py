"""Collect calibration confidence once, then fit E2/E3 curves on CPU.

collect reads HumanEval Demo JSONL/JSONL.GZ and excludes reference lengths +/-4.
fit uses nested seeded subsets and the same weighted double-exponential fit.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

TEST_LENGTHS = [1, 2, 4, 6, 12, 16, 24, 32, 48, 64, 96, 128]
SCORES = ["top1", "margin", "entropy"]


def bias_func(length, a, b, c, d, e):
    return a * np.exp(-b * length) + c * np.exp(-d * length) + e


def collect(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from human_eval_infilling.data import stream_jsonl
    from tqdm import tqdm

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llada_cal"))
    from llada_cal import get_search_confidence

    model = AutoModel.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    problems = list(stream_jsonl(args.data_file))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), open(args.output, "w", encoding="utf-8") as target:
        for problem in tqdm(problems, desc="Calibration probes"):
            reference_length = len(tokenizer.encode(problem["canonical_solution"], add_special_tokens=False))
            prefix = tokenizer(problem["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.to(args.device)
            suffix = tokenizer(problem["suffix"], add_special_tokens=False, return_tensors="pt").input_ids.to(args.device)
            confidence = {score: {} for score in SCORES}
            for length in TEST_LENGTHS:
                if abs(length - reference_length) <= 4:
                    continue
                tokens = torch.full(
                    (1, prefix.shape[1] + length + suffix.shape[1]), 126336,
                    dtype=torch.long, device=model.device,
                )
                tokens[:, :prefix.shape[1]] = prefix
                tokens[:, prefix.shape[1] + length:] = suffix
                logits = model(tokens, attention_mask=torch.ones_like(tokens)).logits
                logits = logits[:, prefix.shape[1]:prefix.shape[1] + length]
                for score in SCORES:
                    confidence[score][str(length)] = get_search_confidence(logits, score).mean().item()
            target.write(json.dumps({
                "task_id": problem["task_id"], "reference_length": reference_length,
                "confidence": confidence, "num_forwards": len(confidence["top1"]),
            }) + "\n")
    print(f"Saved {len(problems)} calibration examples to {args.output}")


def fit_records(records, n, seed, score):
    if n < len(records):
        records = records.copy()
        random.Random(seed).shuffle(records)
        records = records[:n]
    length_to_confidences = {}
    for record in records:
        for length, confidence in record["confidence"][score].items():
            length_to_confidences.setdefault(int(length), []).append(confidence)
    lengths = np.array(sorted(length_to_confidences), dtype=float)
    observations = np.array([np.mean(length_to_confidences[int(length)]) for length in lengths])
    sigmas = np.array([1 / np.sqrt(len(length_to_confidences[int(length)])) for length in lengths])
    offset_upper = 0.5 if score == "top1" else 1.0
    parameters, _ = curve_fit(
        bias_func, lengths, observations, p0=[1.0, 1.8, 0.6, 0.05, 0.3],
        sigma=sigmas, maxfev=10000, bounds=(0, [1.0, 2.0, 1.0, 0.5, offset_upper]),
    )
    fitted = bias_func(lengths, *parameters)
    r2 = 1 - np.sum((observations - fitted) ** 2) / np.sum((observations - observations.mean()) ** 2)
    return {
        "double_exp": {**dict(zip(("a", "b", "c", "d", "e"), map(float, parameters))), "r2": float(r2)},
        "calibration": {
            "score": score, "size": len(records), "seed": seed,
            "task_ids": [record["task_id"] for record in records],
            "num_forwards": sum(record["num_forwards"] for record in records),
            "probe_lengths": TEST_LENGTHS, "oracle_exclusion": 4,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--data_file", required=True)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--model_name", default="GSAI-ML/LLaDA-8B-Base")
    collect_parser.add_argument("--device", default="cuda:0")
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--cache", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--n", type=int, default=100)
    fit_parser.add_argument("--seed", type=int, default=40)
    fit_parser.add_argument("--score", choices=SCORES, default="top1")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        with open(args.cache, encoding="utf-8") as source:
            records = [json.loads(line) for line in source]
        result = fit_records(records, args.n, args.seed, args.score)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as target:
            json.dump(result, target, indent=2)
        print(json.dumps(result["double_exp"], indent=2))


if __name__ == "__main__":
    main()
