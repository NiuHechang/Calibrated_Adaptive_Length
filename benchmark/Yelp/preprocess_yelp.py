import os
import json
import random
from datasets import load_dataset

"""
Preprocessing script for the Yelp Review dataset.
This script extracts reviews from the yelp_review_full dataset and applies 
random contiguous masks (2-8 tokens) to create infilling test cases.
See Section 5.1: Experimental Setup - Text Infilling in the paper.
"""

def preprocess_yelp(output_file, sample_size=1000, seed=42, min_words=30):
    # Set seed for reproducibility
    random.seed(seed)
    
    print("Loading Yelp dataset (test split) from Hugging Face...")
    try:
        # Load the test split of yelp_review_full
        dataset = load_dataset("yelp_review_full", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    all_candidates = []
    print(f"Dataset loaded with {len(dataset)} raw entries. Filtering and processing...")

    for i in range(len(dataset)):
        text = dataset[i]['text']
        
        # Filtering short reviews
        words = text.split()
        if len(words) < min_words:
            continue
            
        # Random contiguous masking (2-8 tokens)
        max_mask_len = min(8, len(words))
        mask_len = random.randint(2, max_mask_len)
        
        # Random starting position
        start_idx = random.randint(0, len(words) - mask_len)
        
        prompt = " ".join(words[:start_idx])
        answer = " ".join(words[start_idx : start_idx + mask_len])
        suffix = " ".join(words[start_idx + mask_len :])
        
        all_candidates.append({
            "prompt": prompt,
            "suffix": suffix,
            "answer": answer
        })
        
        if (i + 1) % 10000 == 0:
            print(f"Scanned {i + 1} raw entries...")

    # Randomly sample N entries for the evaluation set
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
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in final_data:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print("Preprocessing complete.")

if __name__ == "__main__":
    # Path configuration
    output_path = "benchmark/Yelp/Yelp_processed.jsonl"
    
    # Execute preprocessing for the evaluation demo
    preprocess_yelp(output_path, sample_size=-1)
