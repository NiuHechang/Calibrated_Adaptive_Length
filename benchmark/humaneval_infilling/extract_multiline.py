import gzip
import json
import os

"""
Utility for filtering Multi-line HumanEval tasks to prevent data leakage.
This script excludes Multi-line tasks whose corresponding Single-line tasks 
were used in the 'Demo' set for Length Bias fitting.
"""

def convert_single_to_multi_id(single_id):
    """
    Conversion rule for task IDs between Single-line and Multi-line benchmarks.
    Example: SingleLineInfilling/HumanEval/0/L0 -> MultiLineInfilling/HumanEval/0/L0_L0
    """
    parts = single_id.split('/')
    if parts[0] == "SingleLineInfilling":
        parts[0] = "MultiLineInfilling"
        last_part = parts[-1]
        parts[-1] = f"{last_part}_{last_part}"
    return "/".join(parts)

def filter_multiline_dataset(single_demo_path, multi_input_path, multi_output_path):
    # Load IDs used for bias fitting to ensure strict exclusion from evaluation
    print(f"Reading Single-line IDs from {single_demo_path}...")
    ids_to_exclude = set()
    if not os.path.exists(single_demo_path):
        print(f"Error: {single_demo_path} not found.")
        return

    with gzip.open(single_demo_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                original_id = data.get("task_id", "")
                converted_id = convert_single_to_multi_id(original_id)
                ids_to_exclude.add(converted_id)
    
    print(f"Total IDs to exclude: {len(ids_to_exclude)}")

    # Filter the Multi-line dataset
    print(f"Filtering {multi_input_path}...")
    filtered_data = []
    if not os.path.exists(multi_input_path):
        print(f"Error: {multi_input_path} not found.")
        return

    count_removed = 0
    with gzip.open(multi_input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                m_id = data.get("task_id", "")
                if m_id in ids_to_exclude:
                    count_removed += 1
                else:
                    filtered_data.append(data)
    
    print(f"Removed {count_removed} matching records.")
    print(f"Remaining records: {len(filtered_data)}")

    # Save the filtered evaluation set
    print(f"Writing filtered data to {multi_output_path}...")
    with gzip.open(multi_output_path, 'wt', encoding='utf-8') as f:
        for entry in filtered_data:
            f.write(json.dumps(entry) + '\n')

    print("Success.")

if __name__ == "__main__":
    base_dir = "benchmark/humaneval/human-eval-infilling/data"
    
    single_demo = os.path.join(base_dir, "HumanEval-SingleLineInfilling-Demo.jsonl.gz")
    multi_input = os.path.join(base_dir, "HumanEval-MultiLineInfilling.jsonl.gz")
    multi_output = os.path.join(base_dir, "HumanEval-MultiLineInfilling-Rest.jsonl.gz")
    
    filter_multiline_dataset(single_demo, multi_input, multi_output)
