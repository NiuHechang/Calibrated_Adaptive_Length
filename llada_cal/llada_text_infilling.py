import os
import json
import torch
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from llada_cal import generate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np

"""
Unified evaluation script for LLaDA text infilling across multiple datasets 
(ROCStories, CS Abstracts, Yelp Reviews).
This script supports the CAL framework for adaptive length discovery.
See Section 5.1: Experimental Setup - Text Infilling in the paper.
"""

def calculate_metrics(reference_text, hypothesis_text, r_scorer):
    """Computes BLEU-2 and ROUGE-L metrics for text evaluation."""
    # BLEU-2
    ref_tokens = reference_text.split()
    hyp_tokens = hypothesis_text.split()
    smoothie = SmoothingFunction().method1
    bleu2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5), smoothing_function=smoothie)
    
    # ROUGE-L
    rouge_scores = r_scorer.score(reference_text, hypothesis_text)
    rougeL = rouge_scores['rougeL'].fmeasure

    return bleu2, rougeL

def run_on_gpu(rank, args, problems_list, output_dir):
    """
    Worker function for parallel inference on a single GPU.
    """
    gpu_id = args.gpu_ids[rank]
    world_size = len(args.gpu_ids)
    device = f'cuda:{gpu_id}'
    
    print(f"Process {rank} starting on GPU {gpu_id}...")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # Initialize ROUGE scorer
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Partition tasks among processes
    my_problems = [problems_list[i] for i in range(len(problems_list)) if i % world_size == rank]
    
    results = []
    pbar = tqdm(my_problems, desc=f"GPU {gpu_id}", disable=(rank != 0))
    for problem in pbar:
        # Flexible key mapping to support different processed datasets
        story_id = problem.get("id") or problem.get("storyid")
        prompt = problem["prompt"]
        suffix = problem["suffix"]
        answer = problem["answer"]

        # Tokenization
        encoded_prefix = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        encoded_suffix = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")
        encoded_answer = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
        
        prefix_ids = encoded_prefix['input_ids'].to(device)
        suffix_ids = encoded_suffix['input_ids'].to(device)
        answer_ids = encoded_answer['input_ids'].to(device)
        
        # Length configuration logic
        if args.strategy == "oracle":
            gen_length = answer_ids.shape[1]
            curr_dstep = -1 
        else:
            gen_length = args.init_len
            curr_dstep = args.dstep
            
        if gen_length <= 0: continue

        # Perform generation using CAL framework
        out, s_steps = generate(model, prefix=prefix_ids, suffix=suffix_ids, 
                     attention_mask=encoded_prefix['attention_mask'].to(device), 
                     suffix_attention_mask=encoded_suffix['attention_mask'].to(device), 
                     steps=gen_length, gen_length=gen_length, 
                     temperature=0., cfg_scale=0., remasking='low_confidence',
                     span=args.span, max_gen_length=args.max_gen_length, dstep=curr_dstep, use_bias=args.use_bias)

        # Extract the generated middle segment
        generated_ids = out[0, prefix_ids.shape[1] : out.shape[1] - suffix_ids.shape[1]]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute metrics
        bleu2, rougeL = calculate_metrics(answer, completion, r_scorer)

        results.append({
            "storyid": story_id,
            "bleu2": bleu2,
            "rougeL": rougeL,
            "actual_gen_len": len(generated_ids),
            "search_steps": s_steps,
            "prediction": completion.replace('\n', ' '),
            "ground_truth": answer.replace('\n', ' ')
        })

    # Save shard results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"gpu_{gpu_id}.jsonl"), 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')

if __name__ == "__main__":
    # --- Multi-processing configuration ---
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="LLaDA Unified Text Infilling Evaluation")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the processed JSONL file (e.g., benchmark/ROCstories/ROCStories_processed.jsonl)")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--strategy", type=str, choices=["oracle", "fixed"], default="fixed")
    parser.add_argument("--init_len", type=int, default=8, help="Initial generation length (Default 8 for Stories/Abstracts, 2 for Yelp)")
    parser.add_argument("--span", type=int, default=1, help="Search step size (Delta L)")
    parser.add_argument("--max_gen_length", type=int, default=16)
    parser.add_argument("--dstep", type=int, default=2, help="Tolerance (D)")
    parser.add_argument("--use_bias", action="store_true", default=False, help="Enable Length Bias calibration")
    args = parser.parse_args()
    

    # --- Setup Output Paths ---
    benchmark_name = os.path.basename(args.input_file).split('_')[0].lower()
    output_dir = f"method/llada/temp_{benchmark_name}_parts"
    summary_file = "method/llada/summary_llada_text.tsv"
    
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        exit(1)

    # --- Load Problems ---
    problems_list = []
    with open(args.input_file, 'r') as f:
        for line in f:
            problems_list.append(json.loads(line))
            
    timestamp = datetime.now().strftime("%m%d_%H%M")
    print(f"Total problems: {len(problems_list)}")
    print(f"Benchmark: {benchmark_name}, Strategy: {args.strategy}, Init Len: {args.init_len}, DStep: {args.dstep}")
    
    # --- Run Parallel Inference ---
    mp.spawn(
        run_on_gpu,
        args=(args, problems_list, output_dir),
        nprocs=len(args.gpu_ids),
        join=True
    )

    # --- Merge Shards ---
    print("Merging results...")
    all_results = []
    for gpu_id in args.gpu_ids:
        part_file = os.path.join(output_dir, f"gpu_{gpu_id}.jsonl")
        if os.path.exists(part_file):
            with open(part_file, 'r') as f:
                for line in f:
                    all_results.append(json.loads(line))
            os.remove(part_file)

    if not all_results:
        print("No results to summarize.")
        exit(0)

    # --- Evaluation Summary ---
    df = pd.DataFrame(all_results)
    avg_bleu2 = df['bleu2'].mean()
    avg_rougeL = df['rougeL'].mean()
    avg_len = df['actual_gen_len'].mean()
    avg_search_steps = df['search_steps'].mean()
    
    # Log summary metrics to TSV
    headers = ["Time", "Model", "Benchmark", "Init_Len", "Span", "DStep", "UseBias", "Avg_BLEU2", "Avg_RougeL", "Avg_Len", "Avg_SSteps"]

    row = {
        "Time": timestamp,
        "Model": args.model_name.split('/')[-1],
        "Benchmark": benchmark_name,
        "Init_Len": args.init_len,
        "Span": args.span,
        "DStep": args.dstep,
        "UseBias": args.use_bias,
        "Avg_BLEU2": f"{avg_bleu2:.4f}",
        "Avg_RougeL": f"{avg_rougeL:.4f}",
        "Avg_Len": f"{avg_len:.1f}",
        "Avg_SSteps": f"{avg_search_steps:.1f}"
    }
    
    summary_df = pd.DataFrame([row], columns=headers)
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    file_exists = os.path.isfile(summary_file)
    summary_df.to_csv(summary_file, mode='a', index=False, sep='\t', header=not file_exists)
    
    print(f"\n" + "="*30)
    print(f"Benchmark: {benchmark_name}")
    print(f"Average BLEU-2: {avg_bleu2:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")
    print(f"Average Actual Len: {avg_len:.1f}")
    print(f"Average Search Steps: {avg_search_steps:.1f}")
    print(f"Summary appended to: {summary_file}")
    print("="*30)

