import csv
import json
import random
import os

"""
Preprocessing script for the ROCStories dataset.
This script formats five-sentence stories and applies random contiguous 
masks (2-8 tokens) to create infilling tasks.
See Section 5.1: Experimental Setup - Text Infilling in the paper.
"""

def preprocess_roc(input_file, output_file, sample_size=1000, seed=42):
    # Set seed for reproducibility
    random.seed(seed)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    stories = []
    print(f"Reading file: {input_file}...")
    with open(input_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stories.append(row)
            
    if sample_size > 0 and len(stories) >= sample_size:
        sampled_stories = random.sample(stories, sample_size)
    else:
        sampled_stories = stories
    
    processed_data = []
    print(f"Applying random masking for infilling tasks...")
    
    for story in sampled_stories:
        # Combine the 5 sentences into a full story
        sentences = [
            story['sentence1'],
            story['sentence2'],
            story['sentence3'],
            story['sentence4'],
            story['sentence5']
        ]
        full_text = " ".join(sentences)
        words = full_text.split()
        
        # Random mask length (2-8 tokens)
        max_possible_len = min(8, len(words))
        mask_len = random.randint(2, max_possible_len) if max_possible_len >= 2 else 1
        # mask_len = random.randint(7, 10)
        
        # Random starting index for the masked span
        start_idx = random.randint(0, len(words) - mask_len)
        
        # Construct prefix (prompt), ground-truth (answer), and suffix
        prompt = " ".join(words[:start_idx])
        answer = " ".join(words[start_idx : start_idx + mask_len])
        suffix = " ".join(words[start_idx + mask_len :])
        
        processed_data.append({
            "storyid": story['storyid'],
            "storytitle": story['storytitle'],
            "prompt": prompt,
            "suffix": suffix,
            "answer": answer
        })
        
    # Save as JSONL format
    print(f"Saving results to: {output_file}...")
    with open(output_file, mode='w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("Preprocessing complete.")

if __name__ == "__main__":
    # Path configuration
    base_path = "benchmark/ROCstories"
    input_csv = os.path.join(base_path, "ROCStories_winter2017_-_ROCStories_winter2017.csv")
    output_jsonl = os.path.join(base_path, "ROCStories_processed.jsonl")
    
    preprocess_roc(input_csv, output_jsonl)
