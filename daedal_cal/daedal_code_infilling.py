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
from daedal_cal import generate
from benchmark.humaneval.human_eval_infilling.data import write_jsonl, read_problems

"""
Evaluation script for the DAEDAL baseline on HumanEval-Infilling.
DAEDAL uses heuristic-based length adaptation during decoding.
See Section 5.1: Experimental Setup - Code Infilling in the paper.
"""

def run_on_gpu(rank, gpu_ids, problems_list, num_samples_per_task, output_dir, model_name, 
    length_strategy, initial_gen_length, enable_stage1, enable_stage2, expansion_factor, low_conf_threshold, 
    high_conf_threshold, eos_confidence_threshold, expand_eos_confidence_threshold, eos_check_tokens):
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
            canonical_solution = problem.get("canonical_solution", "")

            encoded_prefix = tokenizer(prefix, add_special_tokens=False, padding=True, return_tensors="pt")
            encoded_suffix = tokenizer(suffix, add_special_tokens=False, padding=True, return_tensors="pt")

            prefix_ids = encoded_prefix['input_ids'].to(device)
            suffix_ids = encoded_suffix['input_ids'].to(device)
            
            # Determine initial length based on strategy
            curr_initial_len = initial_gen_length
            curr_block_len = 32
            if length_strategy == "canonical" and canonical_solution:
                encoded_gt = tokenizer(canonical_solution, add_special_tokens=False)
                curr_initial_len = len(encoded_gt['input_ids'])
                curr_block_len = curr_initial_len

            # Perform DAEDAL heuristic generation
            out_list, s1_max, s2_max = generate(model, tokenizer, prefix=prefix_ids, suffix=suffix_ids, 
                               enable_stage1=enable_stage1,
                               enable_stage2=enable_stage2,
                               initial_gen_length=curr_initial_len,
                               max_gen_length=128,
                               block_length=curr_block_len,
                               temperature=0., cfg_scale=0.,
                               expansion_factor=expansion_factor,
                               low_conf_threshold=low_conf_threshold,
                               high_conf_threshold=high_conf_threshold,
                               eos_confidence_threshold=eos_confidence_threshold,
                               expand_eos_confidence_threshold=expand_eos_confidence_threshold,
                               eos_check_tokens=eos_check_tokens)

            # Extract generated middle segment
            o = out_list[0]
            prefix_len = prefix_ids.shape[1]
            suffix_len = suffix_ids.shape[1]
            middle_part = o[prefix_len : len(o) - suffix_len]
            completion = tokenizer.decode(middle_part, skip_special_tokens=True)

            samples.append(dict(
                task_id=task_id, 
                completion=completion,
                gen_len=len(middle_part),
                stage1_max=s1_max[0].item(),
                stage2_max=s2_max[0].item()
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
    parser = argparse.ArgumentParser(description="DAEDAL Baseline Evaluation on HumanEval-Infilling")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--benchmark_name", type=str, default="multi-line")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--length_strategy", type=str, choices=["dynamic", "canonical"], default="dynamic")
    parser.add_argument("--initial_gen_length", type=int, default=8)
    parser.add_argument("--enable_stage1", action="store_true")
    parser.add_argument("--disable_stage2", action="store_true")
    parser.add_argument("--expansion_factor", type=int, default=2)
    parser.add_argument("--low_conf", type=float, default=0.3)
    parser.add_argument("--high_conf", type=float, default=0.9)
    parser.add_argument("--eos_conf", type=float, default=0.5)
    parser.add_argument("--exp_eos_conf", type=float, default=0.9)
    parser.add_argument("--check_tokens", type=int, default=8)
    args = parser.parse_args()

    # --- Setup Output Paths ---
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_base = "benchmark/humaneval/generated/daedal"
    output_dir = os.path.join(output_base, "temp_shards")
    os.makedirs(output_base, exist_ok=True)
    final_output = os.path.join(output_base, f"{args.benchmark_name}_{timestamp}.jsonl")
    
    # --- Load Problems ---
    raw_problems = read_problems(benchmark_name=args.benchmark_name)
    problems_list = [raw_problems[tid] for tid in raw_problems]
    print(f"Starting DAEDAL evaluation on {len(args.gpu_ids)} GPUs: {args.gpu_ids}")
    
    # --- Run Parallel Inference ---
    mp.spawn(
        run_on_gpu,
        args=(args.gpu_ids, problems_list, 1, output_dir, args.model_name, args.length_strategy, 
            args.initial_gen_length, args.enable_stage1, not args.disable_stage2, args.expansion_factor, 
            args.low_conf, args.high_conf, args.eos_conf, args.exp_eos_conf, args.check_tokens),
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
    
    tsv_path = "results_summary_daedal.tsv"
    headers = ["Time", "Model", "Task", "Strategy", "Init_L", "Pass@1", "Avg_Len"]
    row = [timestamp, args.model_name, args.benchmark_name, args.length_strategy, args.initial_gen_length, pass_at_1, f"{avg_len:.1f}"]
    
    file_exists = os.path.isfile(tsv_path)
    with open(tsv_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("\t".join(headers) + "\n")
        f.write("\t".join(map(str, row)) + "\n")
    print(f"Metrics appended to {tsv_path}")
