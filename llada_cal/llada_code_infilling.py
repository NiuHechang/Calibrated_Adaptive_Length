"""HumanEval-Infilling generation and functional evaluation for CAL.

Use --data_file for the prepared Demo/Rest splits. E4 is selected from Multi-Line
Rest by --min_ref_length 160 --max_ref_length 240. Evaluation executes generated
Python through the upstream HumanEval checker; run it in an isolated environment.
"""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from llada_cal import generate, load_bias_params


def run_on_gpu(rank, args, problems_list, output_dir):
    from transformers import AutoModel, AutoTokenizer

    gpu_id = args.gpu_ids[rank]
    device = f"cuda:{gpu_id}"
    model = AutoModel.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    bias_params = load_bias_params(args.bias_params_file) if args.bias_params_file else None
    samples = []
    for problem in tqdm(problems_list[rank::len(args.gpu_ids)], desc=f"GPU {gpu_id}"):
        prefix = tokenizer(problem["prompt"], add_special_tokens=False, return_tensors="pt").to(device)
        suffix = tokenizer(problem["suffix"], add_special_tokens=False, return_tensors="pt").to(device)
        true_len = len(tokenizer.encode(problem["canonical_solution"], add_special_tokens=False))
        initial_length = true_len if args.oracle else args.initial_gen_length
        output, search_steps, search_seconds, decode_seconds = generate(
            model, prefix_ids=prefix["input_ids"], suffix_ids=suffix["input_ids"],
            attention_mask=prefix["attention_mask"], suffix_attention_mask=suffix["attention_mask"],
            steps=args.steps, gen_length=initial_length, block_length=args.block_length,
            temperature=args.temp, cfg_scale=args.cfg_scale, span=args.span,
            max_gen_length=args.max_gen_length, dstep=-1 if args.oracle else args.dstep,
            use_bias=args.use_bias, bias_params=bias_params, search_score=args.search_score,
        )
        middle = output[0, prefix["input_ids"].shape[1]:output.shape[1] - suffix["input_ids"].shape[1]]
        selected_length = len(middle)
        decode_steps = selected_length if args.steps is None else args.steps
        samples.append({
            "task_id": problem["task_id"],
            "completion": tokenizer.decode(middle, skip_special_tokens=True),
            "gen_len": selected_length,
            "actual_gen_len": selected_length,
            "true_len": true_len,
            "len_error": abs(selected_length - true_len),
            "search_steps": search_steps,
            "decode_steps": decode_steps,
            "total_forwards": search_steps + decode_steps,
            "wall_search": search_seconds,
            "wall_decode": decode_seconds,
            "wall_total": search_seconds + decode_seconds,
        })
    with open(Path(output_dir) / f"shard_{rank}.jsonl", "w", encoding="utf-8") as target:
        for sample in samples:
            target.write(json.dumps(sample) + "\n")


def evaluate_samples(problems, samples, timeout, n_workers):
    from human_eval_infilling.execution import check_correctness

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(check_correctness, problems[sample["task_id"]], sample["completion"], timeout, 0)
            for sample in samples
        ]
        for sample, future in zip(samples, futures):
            result = future.result()
            sample["passed"] = result["passed"]
            sample["result"] = result["result"]
    return sum(sample["passed"] for sample in samples) / len(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--benchmark_name", default="single-line")
    parser.add_argument("--data_file", help="Prepared HumanEval JSONL or JSONL.GZ; avoids patching upstream data loaders")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--initial_gen_length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None, help="Omit for S=selected length; use 16/32 for E1a")
    parser.add_argument("--block_length", type=int, default=None, help="Legacy option; CAL decodes one full selected span")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--span", type=int, default=1)
    parser.add_argument("--max_gen_length", type=int, default=64)
    parser.add_argument("--dstep", type=int, default=4, help="Set -1 to disable length search")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--bias_params_file")
    parser.add_argument("--search_score", choices=["top1", "margin", "entropy"], default="top1")
    parser.add_argument("--min_ref_length", type=int, default=None)
    parser.add_argument("--max_ref_length", type=int, default=None)
    parser.add_argument("--output_dir", default="llada_cal/code_results")
    parser.add_argument("--run_name", default="run")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    from human_eval_infilling.data import read_problems, stream_jsonl, write_jsonl

    if args.data_file:
        problems_list = list(stream_jsonl(args.data_file))
    else:
        problems_list = list(read_problems(args.benchmark_name).values())
    if args.min_ref_length is not None or args.max_ref_length is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        lengths = [len(tokenizer.encode(p["canonical_solution"], add_special_tokens=False)) for p in problems_list]
        problems_list = [
            p for p, length in zip(problems_list, lengths)
            if (args.min_ref_length is None or length >= args.min_ref_length)
            and (args.max_ref_length is None or length <= args.max_ref_length)
        ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(args.output_dir) / f"{args.benchmark_name}_{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as target:
        json.dump({**vars(args), "num_examples": len(problems_list)}, target, indent=2)
    write_jsonl(str(run_dir / "problems.jsonl"), problems_list)
    mp.set_start_method("spawn", force=True)
    mp.spawn(run_on_gpu, args=(args, problems_list, str(run_dir)), nprocs=len(args.gpu_ids), join=True)

    samples = []
    for rank in range(len(args.gpu_ids)):
        samples.extend(stream_jsonl(str(run_dir / f"shard_{rank}.jsonl")))
    order = {problem["task_id"]: i for i, problem in enumerate(problems_list)}
    samples.sort(key=lambda sample: order[sample["task_id"]])
    write_jsonl(str(run_dir / "samples.jsonl"), samples)
    pass_at_1 = ""
    if not args.skip_evaluation:
        problems = {problem["task_id"]: problem for problem in problems_list}
        pass_at_1 = 100 * evaluate_samples(problems, samples, args.timeout, args.n_workers)
        write_jsonl(str(run_dir / "samples_results.jsonl"), samples)

    summary = {
        "model": args.model_name, "benchmark": args.benchmark_name, "n": len(samples),
        "initial_length": args.initial_gen_length, "steps_arg": args.steps,
        "oracle": args.oracle, "dstep": args.dstep, "span": args.span,
        "search_score": args.search_score, "use_bias": args.use_bias,
        "bias_params_file": args.bias_params_file, "pass_at_1": pass_at_1,
    }
    for field in ("gen_len", "len_error", "search_steps", "decode_steps", "total_forwards",
                  "wall_search", "wall_decode", "wall_total"):
        summary[f"avg_{field}"] = sum(sample[field] for sample in samples) / len(samples)
    with open(run_dir / "summary.tsv", "w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(summary), delimiter="\t")
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
