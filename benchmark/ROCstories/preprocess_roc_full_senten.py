import csv
import json
import random
import os

"""
Preprocessing script for the ROCStories dataset (Full Sentence Masking).
This script formats five-sentence stories and applies masking to a complete
random sentence (2nd, 3rd, or 4th) to create infilling tasks.
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
        # Use DictReader to automatically handle CSV headers
        reader = csv.DictReader(f)
        for row in reader:
            stories.append(row)
            
    # Check sample size availability
    if len(stories) < sample_size:
        print(f"Warning: Total data size ({len(stories)}) is less than requested {sample_size}. Processing all data.")
        sample_size = len(stories)
    
    # sampled_stories = random.sample(stories, sample_size)
    sampled_stories = stories
    
    processed_data = []
    total_mask_len = 0
    print(f"Processing data and applying random masking...")
    
    for story in sampled_stories:
        # Combine the 5 sentences into a list
        sentences = [
            story['sentence1'],
            story['sentence2'],
            story['sentence3'],
            story['sentence4'],
            story['sentence5']
        ]
        
        # Randomly select the 2nd, 3rd, or 4th sentence for masking (indices 1, 2, 3)
        mask_idx = random.randint(1, 3)
        
        # Construct prefix (prompt), answer, and suffix
        prompt = " ".join(sentences[:mask_idx])
        answer = sentences[mask_idx]
        suffix = " ".join(sentences[mask_idx + 1 :])
        
        # Count mask length (number of words)
        total_mask_len += len(answer.split())
        
        processed_data.append({
            "storyid": story['storyid'],
            "storytitle": story['storytitle'],
            "prompt": prompt,
            "suffix": suffix,
            "answer": answer
        })
        
    # Calculate average mask length
    avg_mask_len = total_mask_len / len(processed_data) if processed_data else 0
        
    # Save as JSONL format
    print(f"Saving results to: {output_file}...")
    with open(output_file, mode='w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Preprocessing complete. Average mask length: {avg_mask_len:.2f} words")

if __name__ == "__main__":
    # Path configuration
    base_path = "benchmark/ROCstories"
    input_csv = os.path.join(base_path, "ROCStories_winter2017_-_ROCStories_winter2017.csv")
    output_jsonl = os.path.join(base_path, "ROCStories_processed_full_senten.jsonl")
    
    preprocess_roc(input_csv, output_jsonl)
