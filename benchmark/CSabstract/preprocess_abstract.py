import gzip
import json
import random
import os

"""
Preprocessing script for the CS Abstracts dataset.
This script prepares the dataset for the text infilling task by applying 
random contiguous masks (2-8 tokens) to academic abstracts.
See Section 5.1: Experimental Setup - Text Infilling in the paper.
"""

def preprocess_cs_abstracts(input_file, output_file, min_words=30, sample_size=1000, seed=42):
    # Set seed for reproducibility
    random.seed(seed)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    all_candidates = []
    print(f"Processing file: {input_file} ...")
    
    # Support for both .gz and plain text files
    open_func = gzip.open if input_file.endswith('.gz') else open
    mode = 'rt' if input_file.endswith('.gz') else 'r'
    
    try:
        with open_func(input_file, mode, encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line.strip())
                # Data block is every 9 lines in this specific format
                if len(lines) == 9:
                    title = lines[4]
                    abstract = lines[6]
                    lines = [] 

                    # Filtering short abstracts
                    words = abstract.split()
                    if len(words) < min_words:
                        continue

                    # Random contiguous masking (Length: 2 to 8 tokens)
                    # This simulates the infilling task setup described in Section 2.
                    max_mask_len = min(8, len(words))
                    mask_len = random.randint(2, max_mask_len)
                    start_idx = random.randint(0, len(words) - mask_len)
                    
                    prompt = " ".join(words[:start_idx])
                    answer = " ".join(words[start_idx : start_idx + mask_len])
                    suffix = " ".join(words[start_idx + mask_len :])
                    
                    all_candidates.append({
                        "title": title,
                        "prompt": prompt,
                        "suffix": suffix,
                        "answer": answer
                    })
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Randomly sample N entries for the evaluation demo
    if sample_size > 0 and len(all_candidates) > sample_size:
        print(f"Sampling {sample_size} entries from {len(all_candidates)} candidates...")
        sampled_data = random.sample(all_candidates, sample_size)
    else:
        sampled_data = all_candidates
    
    # Add unique IDs
    final_data = []
    for i, entry in enumerate(sampled_data, 1):
        entry_with_id = {"id": i}
        entry_with_id.update(entry)
        final_data.append(entry_with_id)

    # Save as JSONL format
    print(f"Saving results to: {output_file} (Total: {len(final_data)} entries)...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in final_data:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print("Preprocessing complete.")

if __name__ == "__main__":
    # Configuration paths
    input_path = "benchmark/CSabstract/CSabstracts.txt" 
    output_path = "benchmark/CSabstract/CSabstracts_demo.jsonl"
    
    preprocess_cs_abstracts(input_path, output_path, sample_size=-1)
