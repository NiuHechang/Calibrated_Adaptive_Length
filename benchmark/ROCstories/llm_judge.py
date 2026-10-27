"""
LLM-as-Judge for ROCStories Text Infilling (Appendix E.3 of the paper).
Three-way ranking: Fixed vs Code-fitted vs In-domain, judged by an external LLM
through an OpenAI-compatible API. This release does not show the reference
completion to the judge. Changing this prompt does not re-evaluate historical
judge results; new runs must be kept separate from them.

Before running, set the following environment variables:
    LLM_JUDGE_API_KEY   API key for the OpenAI-compatible judge endpoint.
    LLM_JUDGE_BASE_URL  Base URL of the judge endpoint (e.g. https://api.openai.com/v1).
    LLM_JUDGE_MODEL     (Optional) Override the judge model. Defaults to the
                        model specified in this file.

Usage:
    python llm_judge.py                       # Run evaluation (1,100 samples x 3 L_init)
    python llm_judge.py --num_samples 500     # Custom sample count
    python llm_judge.py --summarize           # Summarize existing results without calling the API
"""

import json
import os
import re
import random
import time
import argparse
from pathlib import Path
from collections import defaultdict

# ── Configuration ───────────────────────────────────────────────────────
API_KEY  = os.environ.get("LLM_JUDGE_API_KEY", "")
BASE_URL = os.environ.get("LLM_JUDGE_BASE_URL", "")
MODEL    = os.environ.get("LLM_JUDGE_MODEL", "gemini-3.1-pro-preview")

RESULT_DIR = Path(__file__).parent
# Each L_init has three result files (fixed / code-fitted / in-domain),
# produced by llada_text_infilling.py with different --bias_source settings.
# Replace the timestamps below with those produced by your local runs.
TEXT_RESULT_DIR = Path(__file__).resolve().parent.parent.parent / "llada_cal" / "text_result"

# (L_init) -> (fixed_timestamp, code_fitted_timestamp, in_domain_timestamp)
FILE_MAP = {
    2: ("FIXED_L2_TS", "CODE_L2_TS", "INDOMAIN_L2_TS"),
    4: ("FIXED_L4_TS", "CODE_L4_TS", "INDOMAIN_L4_TS"),
    8: ("FIXED_L8_TS", "CODE_L8_TS", "INDOMAIN_L8_TS"),
}

NUM_SAMPLES = 1100
SEED = 42

JUDGE_PROMPT = """\
You are evaluating text infilling quality. A story has a gap between a prefix and a suffix. \
Three different systems (A, B, C) each produced a completion to fill this gap.

**Prefix:** {prefix}
**[GAP]**
**Suffix:** {suffix}

**Completion A:** {comp_a}
**Completion B:** {comp_b}
**Completion C:** {comp_c}

Rank ALL THREE completions from best to worst based on fluency, coherence, and semantic appropriateness. You MUST include all three letters A, B, and C in your response. If some completions are equally good, you may use "=" to indicate a tie, but all three letters must still appear.

Reply with ONLY the ranking. Examples:
- A C B (A is best, C is middle, B is worst)
- A B=C (A is best, B and C are tied for second)
- A=B=C (all three are equally good)"""


def load_results_by_storyid(timestamp):
    """Load a results JSONL and index by storyid."""
    path = TEXT_RESULT_DIR / f"roc_results_{timestamp}.jsonl"
    data = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            data[r["storyid"]] = r
    return data


def select_samples(l_init, num_samples):
    """Select random storyids that exist in all three result files."""
    fixed_ts, code_ts, indom_ts = FILE_MAP[l_init]
    fixed = load_results_by_storyid(fixed_ts)
    code = load_results_by_storyid(code_ts)
    indom = load_results_by_storyid(indom_ts)

    common_ids = sorted(set(fixed) & set(code) & set(indom))
    rng = random.Random(SEED)
    selected = rng.sample(common_ids, min(num_samples, len(common_ids)))
    return selected, fixed, code, indom


def build_request(fixed_rec, code_rec, indom_rec):
    """Build a judge prompt with randomized presentation order."""
    methods = [
        ("fixed", fixed_rec["prediction"]),
        ("code_fitted", code_rec["prediction"]),
        ("in_domain", indom_rec["prediction"]),
    ]
    # Randomize order to eliminate position bias.
    random.shuffle(methods)
    label_map = {}  # A/B/C -> method_name
    completions = {}
    for i, (name, pred) in enumerate(methods):
        letter = chr(ord("A") + i)
        label_map[letter] = name
        completions[f"comp_{letter.lower()}"] = pred.strip() if pred.strip() else "(empty)"

    prompt = JUDGE_PROMPT.format(
        prefix=fixed_rec["prefix"],
        suffix=fixed_rec["suffix"],
        **completions,
    )
    return prompt, label_map


def parse_ranking(content):
    """Parse ranking from LLM response.

    Returns a list of (rank, method_letter) tuples, e.g.:
      "A C B"   -> [(1,"A"), (2,"C"), (3,"B")]
      "A B=C"   -> [(1,"A"), (2,"B"), (2,"C")]
      "A=B=C"   -> [(1,"A"), (1,"B"), (1,"C")]
      "A=B C"   -> [(1,"A"), (1,"B"), (3,"C")]
    Returns None if parsing fails.
    """
    text = content.upper().strip()
    groups = re.split(r'[\s,>]+', text)
    groups = [g.strip() for g in groups if g.strip()]

    result = []
    rank = 1
    for group in groups:
        letters = re.findall(r'[ABC]', group)
        if not letters:
            continue
        has_tie = '=' in group and len(letters) > 1
        if has_tie:
            for l in letters:
                result.append((rank, l))
        else:
            for l in letters:
                result.append((rank, l))
                rank += 1
        if has_tie:
            rank += len(letters)

    found = set(l for _, l in result)
    missing = {"A", "B", "C"} - found
    if len(result) == 3 and not missing:
        return result
    if result and len(missing) <= 2:
        worst_rank = max(r for r, _ in result) + 1
        for m in sorted(missing):
            result.append((worst_rank, m))
        found2 = set(l for _, l in result)
        if found2 == {"A", "B", "C"} and len(result) == 3:
            return result
    return None


def call_judge(prompt):
    """Call LLM judge via OpenAI-compatible API. Single attempt, no retry."""
    if not API_KEY or not BASE_URL:
        raise RuntimeError(
            "LLM_JUDGE_API_KEY and LLM_JUDGE_BASE_URL environment variables "
            "must be set before running the judge."
        )
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        content = response.choices[0].message.content.strip()
        ranking = parse_ranking(content)
        return content, ranking
    except Exception as e:
        return str(e), None


def run_evaluation(num_samples):
    """Run the full evaluation."""
    for l_init in [2, 4, 8]:
        print(f"\n{'='*60}")
        print(f"  Evaluating L_init = {l_init}  ({num_samples} samples)")
        print(f"{'='*60}")

        selected_ids, fixed, code, indom = select_samples(l_init, num_samples)
        output_path = RESULT_DIR / f"llm_judge_results_L{l_init}.jsonl"

        # Resume from existing progress.
        done_ids = set()
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    r = json.loads(line)
                    done_ids.add(r["storyid"])
            print(f"  Resuming: {len(done_ids)} already done, "
                  f"{len(selected_ids) - len(done_ids)} remaining")

        count = 0
        valid = 0
        errors = 0
        with open(output_path, "a") as fout:
            for sid in selected_ids:
                if sid in done_ids:
                    continue

                prompt, label_map = build_request(
                    fixed[sid], code[sid], indom[sid]
                )
                raw_response, parsed = call_judge(prompt)

                record = {
                    "storyid": sid,
                    "l_init": l_init,
                    "label_map": label_map,
                    "raw_response": raw_response,
                }

                if parsed:
                    record["ranks"] = {label_map[l]: r for r, l in parsed}
                    record["valid"] = True
                    valid += 1
                else:
                    record["ranks"] = {}
                    record["valid"] = False
                    errors += 1

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                count += 1

                if count % 50 == 0:
                    print(f"  Progress: {count}/{len(selected_ids) - len(done_ids)} "
                          f"(valid: {valid}, errors: {errors})")

        print(f"  Done: {count} evaluated, {valid} valid, {errors} errors")
        print(f"  Saved to: {output_path}")

    print("\n" + "="*60)
    print("  All evaluations complete. Run with --summarize to see results.")
    print("="*60)


def summarize():
    """Summarize results from existing JSONL files."""
    print("="*70)
    print("  LLM-as-Judge Results: Fixed vs Code-fitted vs In-domain")
    print("  Model: LLaDA-8B-Base  |  Benchmark: ROCStories")
    print(f"  Judge: {MODEL}")
    print("="*70)

    tsv_rows = []
    total_stats = defaultdict(lambda: {"rank_sum": 0, "first": 0, "n": 0})

    for l_init in [2, 4, 8]:
        path = RESULT_DIR / f"llm_judge_results_L{l_init}.jsonl"
        if not path.exists():
            print(f"\n  L_init={l_init}: no results file found")
            continue

        records = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("valid"):
                    records.append(r)

        n = len(records)
        if n == 0:
            continue

        stats = {
            "fixed":       {"rank_sum": 0, "wins": 0},
            "code_fitted": {"rank_sum": 0, "wins": 0},
            "in_domain":   {"rank_sum": 0, "wins": 0},
        }

        for rec in records:
            ranks = rec["ranks"]
            for method in stats:
                r = ranks.get(method, 2)  # default to middle if missing
                stats[method]["rank_sum"] += r
                if r == 1:
                    stats[method]["wins"] += 1

        print(f"\n  L_init = {l_init}  (N = {n} valid samples)")
        print(f"  {'Method':<14} | {'Avg Rank':>8} | {'Win Rate':>8}")
        print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}")

        for method, label in [("fixed", "Fixed"),
                              ("code_fitted", "Code-fitted"),
                              ("in_domain", "In-domain")]:
            s = stats[method]
            avg_rank = s["rank_sum"] / n
            win_rate = s["wins"] / n * 100
            print(f"  {label:<14} | {avg_rank:>8.2f} | {win_rate:>7.1f}%")

            total_stats[method]["rank_sum"] += s["rank_sum"]
            total_stats[method]["first"] += s["wins"]
            total_stats[method]["n"] += n

            tsv_rows.append({
                "L_init": l_init, "Method": label, "N": n,
                "Avg_Rank": f"{avg_rank:.2f}",
                "Win_Rate": f"{win_rate:.1f}%",
            })

    if total_stats:
        total_n = total_stats["fixed"]["n"]
        print(f"\n  {'Overall Average':<14} (N = {total_n})")
        print(f"  {'Method':<14} | {'Avg Rank':>8} | {'1st Rate':>8}")
        print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}")
        for method, label in [("fixed", "Fixed"),
                              ("code_fitted", "Code-fitted"),
                              ("in_domain", "In-domain")]:
            s = total_stats[method]
            avg_rank = s["rank_sum"] / s["n"]
            first_rate = s["first"] / s["n"] * 100
            print(f"  {label:<14} | {avg_rank:>8.2f} | {first_rate:>7.1f}%")

    if tsv_rows:
        tsv_path = RESULT_DIR / "llm_judge_summary.tsv"
        cols = ["L_init", "Method", "N", "Avg_Rank", "Win_Rate"]
        with open(tsv_path, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in tsv_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
        print(f"\n  TSV saved to: {tsv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--summarize", action="store_true",
                        help="Only summarize existing results")
    args = parser.parse_args()

    if args.summarize:
        summarize()
    else:
        random.seed(SEED)
        run_evaluation(args.num_samples)
        summarize()


if __name__ == "__main__":
    main()
