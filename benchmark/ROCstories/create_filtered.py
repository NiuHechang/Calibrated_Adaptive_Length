"""
Filter ROCStories_processed.jsonl by removing the demo subset.

ROCStories_demo.jsonl (1,000 samples) is used for in-domain length-bias
fitting; we exclude these stories from the evaluation set to prevent
data leakage.

Output: ROCStories_processed_filtered.jsonl (used for evaluation).
"""
import json
import os

BASE = os.path.dirname(__file__)

demo_path      = os.path.join(BASE, "ROCStories_demo.jsonl")
processed_path = os.path.join(BASE, "ROCStories_processed.jsonl")
filtered_path  = os.path.join(BASE, "ROCStories_processed_filtered.jsonl")

# 1. Load demo storyid set.
demo_ids = set()
with open(demo_path) as f:
    for line in f:
        demo_ids.add(json.loads(line)["storyid"])
print(f"Demo stories to exclude: {len(demo_ids)}")

# 2. Filter.
kept = 0
with open(processed_path) as fin, open(filtered_path, "w") as fout:
    for line in fin:
        d = json.loads(line)
        if d["storyid"] not in demo_ids:
            fout.write(line)
            kept += 1

print(f"Original: {len(demo_ids) + kept}  ->  Filtered: {kept}  (excluded {len(demo_ids)})")
print(f"Saved to: {filtered_path}")
