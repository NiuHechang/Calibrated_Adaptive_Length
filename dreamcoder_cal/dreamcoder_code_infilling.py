import os
import torch
import subprocess
import json
import argparse
import re
import torch.multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from dreamcoder_cal import generate_fim
from benchmark.humaneval.human_eval_infilling.data import write_jsonl, read_problems

"""
Evaluation script for DreamCoder code infilling on HumanEval-Infilling.
This script supports the CAL framework (length discovery) and standard 
fixed-length or Oracle-length generation.
See Section 5.1: Experimental Setup - Code Infilling in the paper.
"""

def run_on_gpu(rank, gpu_ids, problems_list, num_samples_per_task, output_dir, model_name, 
    initial_gen_length, steps, temperature, alg,
    span, max_gen_length, dstep, use_oracle, use_bias):
    """
    Worker function for parallel inference on a single GPU.
    """
    gpu_id = gpu_ids[rank]
    world_size = len(gpu_ids)
    device = f'cuda:{gpu_id}'
    
    print(f"Process {rank} using GPU {gpu_id}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # Partition tasks among processes
    my_problems = [problems_list[i] for i in range(len(problems_list)) if i % world_size == rank]
    
    samples = []
    for i in range(num_samples_per_task):
        pbar = tqdm(my_problems, desc=f"Sample {i+1}/{num_samples_per_task}", disable=(rank != 0))
        for problem in pbar:
            task_id = problem["task_id"]
            prefix = problem["prompt"]
            suffix = problem["suffix"]

            # Length configuration logic
            curr_gen_len = initial_gen_length
            curr_dstep = dstep
            if use_oracle:
                gt_solution = problem.get("canonical_solution", "")
                if gt_solution:
                    encoded_gt = tokenizer(gt_solution, add_special_tokens=False)
                    curr_gen_len = len(encoded_gt['input_ids'])
                    curr_dstep = -1 # Disable search if Oracle length is provided
                else:
                    print(f"Warning: No canonical_solution for {task_id}, using default.")

            # Perform generation using CAL framework (Integrated Phase 1 Probing)
            completion, actual_gen_len, s_steps = generate_fim(
                model, 
                tokenizer, 
                prefix=prefix, 
                suffix=suffix, 
                gen_length=curr_gen_len,
                steps=steps, 
                temperature=temperature, 
                alg=alg,
                span=span,
                max_gen_length=max_gen_length,
                dstep=curr_dstep,
                use_bias=use_bias
            )

            samples.append(dict(
                task_id=task_id, 
                completion=completion,
                gen_len=actual_gen_len,
                actual_gen_len=actual_gen_len,
                search_steps=s_steps
            ))

    # Save shard results
    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(os.path.join(output_dir, f"shard_gpu_{gpu_id}.jsonl"), samples)

if __name__ == "__main__":
    # --- Multi-processing configuration ---
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="DreamCoder CAL Evaluation on HumanEval-Infilling")
    parser.add_argument("--model_name", type=str, default="Dream-org/Dream-Coder-v0-Base-7B")
    parser.add_argument("--benchmark_name", type=str, default="single-line")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--initial_gen_length", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--span", type=int, default=1, help="Search step size (Delta L)")
    parser.add_argument("--max_gen_length", type=int, default=64)
    parser.add_argument("--dstep", type=int, default=4, help="Tolerance (D)")
    parser.add_argument("--oracle", action="store_true", help="Use ground-truth length")
    parser.add_argument("--use_bias", action="store_true", default=False, help="Enable Length Bias calibration")
    args = parser.parse_args()

    # --- Setup Output Paths ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_base = "benchmark/humaneval/generated/dreamcoder_cal"
    output_dir = os.path.join(output_base, "temp_shards")
    os.makedirs(output_base, exist_ok=True)
    final_output = os.path.join(output_base, f"{args.benchmark_name}_{timestamp}.jsonl")
    
    # --- Load Problems ---
    raw_problems = read_problems(benchmark_name=args.benchmark_name)
    problems_list = [raw_problems[tid] for tid in raw_problems]
    print(f"Starting DreamCoder-CAL on {len(args.gpu_ids)} GPUs: {args.gpu_ids}")
    
    # --- Run Parallel Inference ---
    mp.spawn(
        run_on_gpu,
        args=(args.gpu_ids, problems_list, 1, output_dir, args.model_name, 
            args.initial_gen_length, args.steps, args.temp, args.alg,
            args.span, args.max_gen_length, args.dstep, args.oracle, args.use_bias),
        nprocs=len(args.gpu_ids),
        join=True
    )

    # --- Merge Shards ---
    all_samples = []
    for gpu_id in args.gpu_ids:
        part_file = os.path.join(output_dir, f"shard_gpu_{gpu_id}.jsonl")
        if os.path.exists(part_file):
            with open(part_file, 'r') as f:
                for line in f:
                    all_samples.append(json.loads(line))
            os.remove(part_file)
    write_jsonl(final_output, all_samples)
    print(f"Merged results saved to {final_output}")

    # --- Evaluation ---
    print(f"Running functional correctness evaluation for {args.benchmark_name}...")
    pass_at_1 = "N/A"
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        benchmark_root = os.path.join(project_root, "benchmark", "humaneval")
        eval_script = os.path.join(benchmark_root, "human_eval_infilling", "evaluate_functional_correctness.py")
        eval_cmd = f"PYTHONPATH={benchmark_root} python {eval_script} {final_output} --benchmark_name={args.benchmark_name}"
        eval_output = subprocess.check_output(eval_cmd, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        match = re.search(r"'pass@1':\s*([\d\.]+)", eval_output)
        if match:
            pass_at_1 = f"{float(match.group(1))*100:.2f}%"
            print(f"Pass@1: {pass_at_1}")
    except Exception as e:
        print(f"Evaluation failed: {e}")

    # --- Logging Metrics ---
    total = len(all_samples)
    avg_len = sum(s.get("gen_len", 0) for s in all_samples) / total if total > 0 else 0
    avg_ss = sum(s.get("search_steps", 0) for s in all_samples) / total if total > 0 else 0
    
    tsv_path = "results_summary_dreamcoder.tsv"
    headers = ["Time", "Model", "Task", "Init_L", "Span", "Max_L", "DStep", "Bias", "Pass@1", "Avg_Len", "Avg_SS"]
    row = [timestamp, args.model_name, args.benchmark_name, args.initial_gen_length, args.span, 
           args.max_gen_length, args.dstep, args.use_bias, pass_at_1, f"{avg_len:.1f}", f"{avg_ss:.1f}"]
    
    file_exists = os.path.isfile(tsv_path)
    with open(tsv_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("\t".join(headers) + "\n")
        f.write("\t".join(map(str, row)) + "\n")
    print(f"Metrics appended to {tsv_path}")
