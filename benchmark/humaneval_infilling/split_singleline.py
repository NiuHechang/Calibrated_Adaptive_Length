import gzip
import json
import random
import os

"""
Data split utility for HumanEval-SingleLineInfilling.
This script partitions the dataset into a 'Demo' set (used for fitting the 
Length Bias function B(L)) and a 'Rest' set for final evaluation.
"""

def split_dataset(input_path, output_100_path, output_rest_path, sample_size=100):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Reading from {input_path}...")
    data = []
    try:
        with gzip.open(input_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"Total records: {len(data)}")
    
    if len(data) < sample_size:
        print(f"Warning: Not enough data ({len(data)}) to sample {sample_size}. Taking all.")
        sample_size = len(data)

    # Use fixed seed for reproducibility across environments
    random.seed(42)
    random.shuffle(data)
    
    # Split: first N for Bias Fitting (Demo), the rest for Evaluation
    selected_100 = data[:sample_size]
    rest_data = data[sample_size:]
    
    print(f"Writing {len(selected_100)} records to {output_100_path} (Bias Fitting set)...")
    with gzip.open(output_100_path, 'wt', encoding='utf-8') as f:
        for entry in selected_100:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Writing {len(rest_data)} records to {output_rest_path} (Evaluation set)...")
    with gzip.open(output_rest_path, 'wt', encoding='utf-8') as f:
        for entry in rest_data:
            f.write(json.dumps(entry) + '\n')

    print("Success.")

if __name__ == "__main__":
    base_dir = "benchmark/humaneval/human-eval-infilling/data"
    input_file = os.path.join(base_dir, "HumanEval-SingleLineInfilling.jsonl.gz")
    output_100 = os.path.join(base_dir, "HumanEval-SingleLineInfilling-Demo.jsonl.gz")
    output_rest = os.path.join(base_dir, "HumanEval-SingleLineInfilling-Rest.jsonl.gz")
    
    split_dataset(input_file, output_100, output_rest)
