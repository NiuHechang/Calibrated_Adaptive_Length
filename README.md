# Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly

[![arXiv](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2602.00476)

This is the official implementation of the paper **"Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly"**. 

In this work, we propose **CAL** (Calibrated Adaptive Length), a training-free framework that enables Diffusion LMs to approximate optimal infilling lengths. If you have any questions about the paper, please feel free to open an issue for discussion. 

## Repository Structure

- `length_bias.py`: Script for fitting the **Length Bias** function $B(L)$ via double-exponential decay. (See Appendix A).
- `oracle_peak.py`: Script for visualizing the **Oracle Peak** phenomenon and evaluating calibrated confidence $\Phi_{\text{n}}(L)$. (See Section 3).
- `benchmark/`: Preprocessing scripts for all datasets (HumanEval, ROCStories, CSAbstracts, Yelp).
  - **HumanEval-Infilling** (Code)
  - **ROCStories** (Text)
  - **CSAbstracts** (Text)
  - **Yelp Reviews** (Text)
- `llada_cal/`, `diffucoder_cal/`, `dreamcoder_cal/`, `daedal_cal/`: Implementation of the CAL framework and evaluation scripts for respective models.


## Benchmark Setup

### 1. ROCStories
1. Apply for access at the [ROCStories website](https://cs.rochester.edu/nlp/rocstories/).
2. Upon approval, download the **"ROCStories (full five-sentence stories) winter 2017 set"**.
3. Place the CSV in `benchmark/ROCstories/` and run `preprocess_roc.py`.

### 2. CSAbstracts
1. Follow the instructions in the [ILM repository](https://github.com/chrisdonahue/ilm/blob/master/data/get_arxiv_cs_abstracts.sh) to download the dataset.
2. Place the raw text in `benchmark/CSabstract/` and run `preprocess_abstract.py`.

### 3. Yelp Reviews
The Yelp dataset is automatically downloaded via the Hugging Face `datasets` library. Simply run `benchmark/Yelp/preprocess_yelp.py`.

### 4. HumanEval-Infilling
1. Clone and install the official [HumanEval-Infilling](https://github.com/openai/human-eval-infilling) repository.
2. Use the scripts provided in our `benchmark/humaneval_infilling/` to process the official `.jsonl.gz` files.
3. This process splits the data into three parts: `Demo` (for bias fitting), `Rest (Single-Line)`, and `Rest (Multi-Line)` (for evaluation).
4. **Crucial Step:** To enable reading the split datasets, replace the `read_problems` function in `human_eval_infilling/data.py` with the following implementation:

```python
def read_problems(benchmark_name: str) -> Dict[str, Dict]:
    benchmark_file = {
        "demo": os.path.join(ROOT, "HumanEval-SingleLineInfilling-Demo.jsonl.gz"),
        "single-line": os.path.join(ROOT, "HumanEval-SingleLineInfilling-Rest.jsonl.gz"),
        "multi-line": os.path.join(ROOT, "HumanEval-MultiLineInfilling-Rest.jsonl.gz"),
        "random-span": os.path.join(ROOT, "HumanEval-RandomSpanInfilling.jsonl.gz"),
        "random-span-light": os.path.join(ROOT, "HumanEval-RandomSpanInfillingLight.jsonl.gz"),
        "test": os.path.join(ROOT, "example_problem.jsonl"),
    }[benchmark_name]
    return {task["task_id"]: task for task in stream_jsonl(benchmark_file)}
```

The random seeds used in our scripts are identical to those used for the results reported in the paper. Following the preprocessing steps above should yield datasets consistent with our experimental setup.


## Citation
```
@misc{liu2026diffusionlmsapproximateoptimal,
      title={Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly}, 
      author={Hengchang Liu and Zhao Yang and Bing Su},
      year={2026},
      eprint={2602.00476},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.00476}, 
}