# Calibrated Adaptive Length (CAL)

Code for **[Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly](https://arxiv.org/abs/2602.00476)**, NeurIPS 2026.

CAL uses calibrated first-step confidence to select an infilling length without retraining the model. This release includes the core method and additional experiments; model weights and datasets are not bundled.

## Setup

Run commands from the repository root. Install PyTorch for your CUDA environment, then:

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/human-eval-infilling.git
```

Models use `trust_remote_code=True`. HumanEval executes generated code: use an isolated environment and follow the upstream [instructions for enabling execution](https://github.com/openai/human-eval-infilling/blob/master/human_eval_infilling/execution.py). Use `--skip_evaluation` for generation only.

## Code infilling

Download the Single-Line and Multi-Line files from [HumanEval-Infilling](https://github.com/openai/human-eval-infilling) into a local `data/` directory. Prepare the calibration and test splits:

```bash
python benchmark/humaneval_infilling/split_singleline.py \
  --input data/HumanEval-SingleLineInfilling.jsonl.gz \
  --demo_output data/HumanEval-SingleLineInfilling-Demo.jsonl.gz \
  --test_output data/HumanEval-SingleLineInfilling-Rest.jsonl.gz
python benchmark/humaneval_infilling/extract_multiline.py \
  --single_demo data/HumanEval-SingleLineInfilling-Demo.jsonl.gz \
  --multi_input data/HumanEval-MultiLineInfilling.jsonl.gz \
  --output data/HumanEval-MultiLineInfilling-Rest.jsonl.gz
```

The split excludes matching infill tasks, not entire functions. Run CAL on LLaDA-Base:

```bash
python llada_cal/llada_code_infilling.py \
  --data_file data/HumanEval-SingleLineInfilling-Rest.jsonl.gz \
  --initial_gen_length 8 --span 1 --dstep 4 --max_gen_length 64 \
  --use_bias --gpu_ids 0
```

Use `--dstep -1` for Fixed or `--oracle` for reference-length decoding. Set `--model_name` to change checkpoints. Code decoding defaults to one step per selected token; search adds forwards. Results are saved under `llada_cal/code_results/`.

DiffuCoder, DreamCoder, and DAEDAL entry points are in their respective `*_cal/` folders and also accept `--data_file`. See each script's `--help`.

## Additional experiments

Use the LLaDA code runner above unless noted. Fixed disables search; CAL uses `--use_bias`.

| Experiment | Key settings |
|---|---|
| Search vs. more decoding (E1a) | Init32. Fixed-fast: `--steps 16 --dstep -1`; Fixed-full: `--steps 32 --dstep -1`; CAL: `--steps 16 --dstep 4`. Single-Line max64; Multi-Line max128. Approximately matched average total forwards, not per-example compute. |
| Longer fixed baselines (E1b) | Multi-Line, Base/Instruct, init48/64, max128; Fixed/CAL; omit `--steps`. |
| Calibration size (E2) | Single-Line, init8; fit n=3/5/10/25/50 with seeds40/41/42, plus n=100. |
| Confidence scores (E3) | Single-Line, init4/8/16/32; `--search_score margin` or `entropy`, paired with `motivation/bias_margin.json` or `bias_entropy.json` via `--bias_params_file`. |
| Long spans (E4) | Multi-Line, Base; `--min_ref_length 160 --max_ref_length 240 --initial_gen_length 200 --max_gen_length 256`; Fixed/CAL/reference length; omit `--steps`. |

Collect calibration confidence once, then fit on CPU:

```bash
python motivation/calibrate_llada.py collect \
  --data_file data/HumanEval-SingleLineInfilling-Demo.jsonl.gz \
  --output results/confidence.jsonl
python motivation/calibrate_llada.py fit \
  --cache results/confidence.jsonl --n 3 --seed 40 --score top1 \
  --output results/bias_n3_seed40.json
```

Pass the fitted file with `--bias_params_file`. E2's n=100 reference uses `motivation/bias_top1_fitted.json` (full precision); the main Top-1 baseline uses the rounded default curve.

## Text and multimodal infilling

Text preprocessing scripts are in `benchmark/`; records contain `prompt`, `suffix`, and `answer`. Obtain [ROCStories](https://cs.rochester.edu/nlp/rocstories/) and set the input path in its preprocessing script; use `sample_size=0` to process all stories.

```bash
python llada_cal/llada_text_infilling.py \
  --input_file benchmark/ROCstories/ROCStories_processed.jsonl \
  --strategy cal --init_len 4 --span 1 --dstep 2 --max_gen_length 16 \
  --use_bias --gpu_ids 0
```

Use `--strategy fixed` or `oracle` for controls. Text decoding defaults to the initial length, not the selected length. Main code-to-text experiments reuse the code-fitted bias.

For LaViDa and LLaDA-V, follow [multimodal/README.md](multimodal/README.md). Each model requires its own upstream environment and uses a separate COCO-fitted bias.

The optional `benchmark/ROCstories/llm_judge.py` uses environment-based API credentials and does not show reference answers. This released prompt does not establish how historical judge scores were obtained.

## Tests and license

```bash
python -m unittest discover -s tests -v
```

Tests cover CPU logic and interfaces, not GPU reproduction of paper results.

CAL-specific code is licensed under [Apache-2.0](LICENSE). Third-party notices and applicable licenses are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{liu2026diffusionlmsapproximateoptimal,
  title={Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly},
  author={Hengchang Liu and Zhao Yang and Bing Su},
  year={2026},
  eprint={2602.00476},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.00476},
}
```
