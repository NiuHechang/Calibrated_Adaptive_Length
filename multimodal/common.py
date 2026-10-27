"""Shared CAL search and small I/O helpers for caption infilling."""

import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INSTRUCTION = "Provide a one-sentence caption for the provided image."


def add_model_arguments(parser):
    parser.add_argument("--model", choices=["lavida", "llada_v"], required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vision-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)


def load_scorer(args):
    # Each process/environment loads only one of the two upstream llava packages.
    if args.model == "lavida":
        sys.path.insert(0, str(args.upstream_root.resolve()))
        from lavida_cal import LaViDaCALScorer

        scorer_class = LaViDaCALScorer
        vision = args.vision_checkpoint or "google/siglip-so400m-patch14-384"
    else:
        sys.path.insert(0, str((args.upstream_root / "train").resolve()))
        from llada_v_cal import LLaDAVCALScorer

        scorer_class = LLaDAVCALScorer
        vision = args.vision_checkpoint or "google/siglip2-so400m-patch14-384"
    return scorer_class(args.checkpoint, args.device, vision)


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def write_json(data, path):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_tsv(rows, path):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def bias_value(length, parameters):
    return (
        parameters["a"] * math.exp(-parameters["b"] * length)
        + parameters["c"] * math.exp(-parameters["d"] * length)
        + parameters["e"]
    )


def search_length(scorer, context, prefix_ids, suffix_ids, init_length,
                  span, max_length, dstep, parameters):
    curve = []

    def score(length):
        raw = scorer.score_length(context, prefix_ids, suffix_ids, length).confidence
        calibrated = raw / bias_value(length, parameters)
        curve.append({"length": length, "raw_confidence": raw,
                      "calibrated_confidence": calibrated})
        return calibrated

    best_length = init_length
    best_score = score(init_length)
    # Both directions share the best score; D counts failures to improve it.
    for direction in (1, -1):
        length = init_length + direction * span
        failures = 0
        while 1 <= length <= max_length:
            current = score(length)
            if current > best_score:
                best_score, best_length = current, length
                failures = 0
            else:
                failures += 1
                if failures >= dstep:
                    break
            length += direction * span
    return best_length, curve


def calculate_metrics(reference, prediction, rouge):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    bleu2 = sentence_bleu(
        [reference.split()], prediction.split(), weights=(0.5, 0.5),
        smoothing_function=SmoothingFunction().method1,
    )
    rouge_l = rouge.score(reference, prediction)["rougeL"].fmeasure
    return bleu2, rouge_l
