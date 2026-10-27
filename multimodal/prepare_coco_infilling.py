"""Create the paired 50/500 COCO word-span protocol (seed 42)."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions-file", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--calibration-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(records, path):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def length_summary(records, field):
    lengths = [record[field] for record in records]
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": mean(lengths),
        "distribution": dict(sorted(Counter(lengths).items())),
    }


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

    with args.captions_file.open(encoding="utf-8") as file:
        coco = json.load(file)

    images = {image["id"]: image for image in coco["images"]}
    captions_by_image = {}
    for annotation in coco["annotations"]:
        captions_by_image.setdefault(annotation["image_id"], []).append(annotation)

    rng = random.Random(args.seed)
    eligible = []
    for image_id in sorted(images):
        annotation = min(captions_by_image[image_id], key=lambda item: item["id"])
        caption = annotation["caption"].strip()
        caption_words = caption.split()
        if len(caption_words) < 4:
            continue

        max_mask_word_length = min(8, len(caption_words) - 2)
        mask_word_length = rng.randint(2, max_mask_word_length)
        start = rng.randint(1, len(caption_words) - mask_word_length - 1)
        end = start + mask_word_length

        prefix_text = " ".join(caption_words[:start])
        target_text = " ".join(caption_words[start:end])
        suffix_text = " ".join(caption_words[end:])

        caption_ids = tokenizer.encode(caption, add_special_tokens=False)
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        reference_length = len(target_ids)

        image = images[image_id]

        eligible.append(
            {
                "example_id": f"coco_val2017_{image_id:012d}",
                "image_id": image_id,
                "caption_id": annotation["id"],
                "image_file": f"val2017/{image['file_name']}",
                "caption": caption,
                "caption_ids": caption_ids,
                "prefix_ids": prefix_ids,
                "target_ids": target_ids,
                "suffix_ids": suffix_ids,
                "prefix_text": prefix_text,
                "target_text": target_text,
                "suffix_text": suffix_text,
                "mask_word_start": start,
                "mask_word_end": end,
                "mask_word_length": mask_word_length,
                "reference_length": reference_length,
            }
        )

    rng.shuffle(eligible)
    required = args.calibration_size + args.test_size

    calibration = eligible[: args.calibration_size]
    test = eligible[args.calibration_size : required]
    for record in calibration:
        record["split"] = "calibration"
    for record in test:
        record["split"] = "test"

    calibration_path = args.output_dir / "coco_word_span_calibration.jsonl"
    test_path = args.output_dir / "coco_word_span_test.jsonl"
    stats_path = args.output_dir / "coco_word_span_stats.json"
    write_jsonl(calibration, calibration_path)
    write_jsonl(test, test_path)

    calibration_ids = {record["image_id"] for record in calibration}
    test_ids = {record["image_id"] for record in test}

    stats = {
        "source": "MS COCO 2017 validation captions",
        "tokenizer_checkpoint": str(args.checkpoint),
        "caption_policy": "lowest caption annotation ID per image",
        "span_policy": "uniform contiguous 2-8 whitespace-delimited words with at least one word in both prefix and suffix",
        "seed": args.seed,
        "total_coco_images": len(images),
        "eligible_examples": len(eligible),
        "calibration_examples": len(calibration),
        "test_examples": len(test),
        "calibration_test_image_overlap": len(calibration_ids & test_ids),
        "mask_word_length_range": [2, 8],
        "calibration_mask_word_lengths": length_summary(
            calibration,
            "mask_word_length",
        ),
        "test_mask_word_lengths": length_summary(test, "mask_word_length"),
        "calibration_reference_token_lengths": length_summary(
            calibration,
            "reference_length",
        ),
        "test_reference_token_lengths": length_summary(test, "reference_length"),
        "calibration_sha256": sha256(calibration_path),
        "test_sha256": sha256(test_path),
    }
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
