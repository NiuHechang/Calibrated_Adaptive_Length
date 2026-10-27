"""Paired Fixed/CAL evaluation on image-conditioned caption spans."""

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from common import (ROOT, add_model_arguments, calculate_metrics, load_scorer,
                    read_jsonl, search_length, write_json, write_tsv)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--bias-file", type=Path)
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--init-length", type=int, default=8)
    parser.add_argument("--span", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=16)
    parser.add_argument("--dstep", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def decode(scorer, context, example, length, steps, rouge):
    import torch

    torch.cuda.synchronize(scorer.device)
    started = time.time()
    ids = scorer.decode_infilling(
        context, example["prefix_ids"], example["suffix_ids"], length, steps,
    )
    torch.cuda.synchronize(scorer.device)
    seconds = time.time() - started
    prediction = scorer.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    bleu2, rouge_l = calculate_metrics(example["target_text"], prediction, rouge)
    return {"prediction": prediction, "generated_ids": ids[0].tolist(),
            "selected_length": length, "configured_decode_steps": steps,
            "bleu2": bleu2, "rougeL": rouge_l, "decode_seconds": seconds}


def main():
    args = parse_args()
    import torch
    from rouge_score import rouge_scorer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bias_file = args.bias_file or ROOT / "configs" / f"{args.model}_bias.json"
    with bias_file.open(encoding="utf-8") as file:
        bias = json.load(file)
    examples = read_jsonl(args.data_file)[:args.num_examples]
    torch.cuda.set_device(args.device)
    scorer = load_scorer(args)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    records = []
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as output:
        for index, example in enumerate(examples, 1):
            context = scorer.encode_context(
                args.image_root / example["image_file"], args.instruction,
            )
            fixed = decode(scorer, context, example, args.init_length,
                           args.init_length, rouge)
            fixed.update(search_forwards=0, search_seconds=0.0)
            torch.cuda.synchronize(scorer.device)
            started = time.time()
            length, curve = search_length(
                scorer, context, example["prefix_ids"], example["suffix_ids"],
                args.init_length, args.span, args.max_length, args.dstep,
                bias["parameters"],
            )
            torch.cuda.synchronize(scorer.device)
            search_seconds = time.time() - started
            cal = decode(scorer, context, example, length, args.init_length, rouge)
            cal.update(search_forwards=len(curve), search_seconds=search_seconds,
                       search_curve=curve)
            record = {"example_id": example["example_id"],
                      "image_file": example["image_file"],
                      "prefix_text": example["prefix_text"],
                      "target_text": example["target_text"],
                      "suffix_text": example["suffix_text"],
                      "reference_token_length": example["reference_length"],
                      "fixed": fixed, "cal": cal}
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index}/{len(examples)}] {example['example_id']} "
                  f"L={args.init_length}->{length} "
                  f"B2={fixed['bleu2']:.4f}->{cal['bleu2']:.4f} "
                  f"RL={fixed['rougeL']:.4f}->{cal['rougeL']:.4f}", flush=True)
            del context
            torch.cuda.empty_cache()

    summary = []
    for method in ("fixed", "cal"):
        row = {"model": args.model, "method": method, "examples": len(records),
               "init_length": args.init_length}
        for field in ("bleu2", "rougeL", "selected_length", "search_forwards",
                      "search_seconds", "decode_seconds"):
            row[f"avg_{field}"] = mean(record[method][field] for record in records)
        summary.append(row)
    write_tsv(summary, args.output_dir / "summary.tsv")
    write_json({"configuration": {key: str(value) if isinstance(value, Path)
                                  else value for key, value in vars(args).items()},
                "bias": bias,
                "decode_budget": "configured steps = initial length; search is additional",
                "metrics": "sentence BLEU-2 (method1 smoothing); ROUGE-L F1 (stemming)",
                "summary": summary}, args.output_dir / "metadata.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
