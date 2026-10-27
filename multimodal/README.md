# Image-conditioned caption infilling

This is the core implementation of the final LaViDa and LLaDA-V experiment.
Given an image and a caption's prefix/suffix, Fixed and CAL generate the missing
text span. Historical reference-peak diagnostics, model weights, raw COCO data,
and experiment logs are not bundled.

Run the commands below from the `Calibrated_Adaptive_Length` repository root.

## Model environments

Use **separate environments** for the two models: both upstream projects define
a package named `llava`. Follow each upstream project's installation instructions
and use the source revisions below. `--upstream-root` selects that checkout; it
does not install its dependencies.

| Model | Upstream source | Source commit |
|---|---|---|
| LaViDa | [jacklishufan/LaViDa](https://github.com/jacklishufan/LaViDa) | `24220c0077f403d24ca52968b6a7c55b687d5c2f` |
| LLaDA-V | [ML-GSAI/LLaDA-V](https://github.com/ML-GSAI/LLaDA-V) | `f8b02ce04b09f4f271fe55a3652059a73bbc7a32` |

Use the Python/PyTorch/transformers dependencies specified by each pinned
upstream checkout. Add the evaluation dependencies in each environment:

```bash
python -m pip install nltk==3.10.1 rouge-score==0.1.2
```

Download the following checkpoint snapshots to local directories and pass those
directories to `--checkpoint` / `--vision-checkpoint`:

| Model / encoder | Hugging Face repository | Revision |
|---|---|---|
| LaViDa-LLaDA | `jacklishufan/lavida-llada-v1.0-instruct` | `814b2e364e82390f03df451bdf4e81e8ba8eab37` |
| LaViDa vision encoder | `google/siglip-so400m-patch14-384` | `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` |
| LLaDA-V | `GSAI-ML/LLaDA-V` | `a10ba1790083e10e07cfcab6e7be93e1823f5792` |
| LLaDA-V vision encoder | `google/siglip2-so400m-patch14-384` | `18e4e5df0cd4868902ea7a4c09e84d8017a512a2` |

Use one visible GPU per process (`CUDA_VISIBLE_DEVICES=...`, `--device cuda:0`).
The measured peak allocated memory was about 17 GiB for LaViDa and 31 GiB for
LLaDA-V; allow extra memory for the runtime. The upstream LLaDA-V loader places
its vision encoder on the first visible GPU.

## Prepare the paired caption split

Obtain the MS COCO 2017 validation images and caption annotations. The image
root passed below must contain `val2017/`; the caption file is
`annotations/captions_val2017.json`.

```bash
python multimodal/prepare_coco_infilling.py \
  --captions-file /path/to/coco/annotations/captions_val2017.json \
  --checkpoint /path/to/lavida-checkpoint \
  --output-dir multimodal/data \
  --calibration-size 50 --test-size 500 --seed 42
```

The script selects the lowest annotation-ID caption per image. It samples a
span length from 2--8 whitespace words (subject to caption length), then a valid
position, keeping at least one word on each side. Prefix, target and suffix are
tokenized separately. After a seeded shuffle, the first 50 images are used for
calibration and the next 500 for evaluation. Reuse these same files for both
models; the tested tokenizers produce identical span token IDs on this split.

## Run Fixed and CAL

The model-specific bias parameters used in the paper are provided in
`configs/lavida_bias.json` and `configs/llada_v_bias.json`. The runner selects
the corresponding file by default. They are frozen COCO fits, **not** the
original code-domain bias.

LaViDa (inside its own environment):

```bash
CUDA_VISIBLE_DEVICES=0 python multimodal/run_infilling_quality.py \
  --model lavida --upstream-root /path/to/LaViDa \
  --checkpoint /path/to/lavida-checkpoint \
  --vision-checkpoint /path/to/siglip-checkpoint \
  --data-file multimodal/data/coco_word_span_test.jsonl \
  --image-root /path/to/coco \
  --num-examples 500 --init-length 2 --span 1 --dstep 2 --max-length 16 \
  --output-dir outputs/lavida_init2
```

LLaDA-V (inside its separate environment; `--upstream-root` is the repository
root, **not** its `train/` directory):

```bash
CUDA_VISIBLE_DEVICES=0 python multimodal/run_infilling_quality.py \
  --model llada_v --upstream-root /path/to/LLaDA-V \
  --checkpoint /path/to/llada-v-checkpoint \
  --vision-checkpoint /path/to/siglip2-checkpoint \
  --data-file multimodal/data/coco_word_span_test.jsonl \
  --image-root /path/to/coco \
  --num-examples 500 --init-length 2 --span 1 --dstep 2 --max-length 16 \
  --output-dir outputs/llada_v_init2
```

Repeat each command with `--init-length 4` and `8`, changing `--output-dir`
accordingly. For a short smoke test, use `--num-examples 2`. Each run writes
`records.jsonl`, `summary.tsv`, and `metadata.json`. Summary metric values are
on a 0--1 scale; multiply by 100 for the paper's tables.

Fixed and CAL use the same **configured** decoding steps, equal to the initial
length. The official decoders can use fewer effective denoising forwards when
the selected canvas is shorter. CAL additionally performs length-search
forwards, so this is not a compute-matched comparison. Scores average raw
first-step top-1 probabilities over masked positions. LLaDA-V retains its
official special-token filtering for decoding, but not for CAL's raw score.
The instruction is `Provide a one-sentence caption for the provided image.`

Metrics are missing-span sentence BLEU-2 (whitespace tokens, NLTK method1
smoothing) and ROUGE-L F1 (`rouge_score`, stemming enabled), averaged over
examples. They are not corpus BLEU or caption-length matching metrics.

## Optional: refit a model-specific bias

The calibration uses lengths 1--16, excluding lengths within one token of the
reference, followed by the original count-weighted double-exponential fit.
For LaViDa:

```bash
CUDA_VISIBLE_DEVICES=0 python multimodal/fit_bias.py \
  --model lavida --upstream-root /path/to/LaViDa \
  --checkpoint /path/to/lavida-checkpoint \
  --vision-checkpoint /path/to/siglip-checkpoint \
  --data-file multimodal/data/coco_word_span_calibration.jsonl \
  --image-root /path/to/coco \
  --num-examples 50 --min-length 1 --max-length 16 --exclusion-radius 1 \
  --output outputs/lavida_bias.json
```

For LLaDA-V, substitute `--model llada_v` and its source/checkpoint/encoder
paths. Use the fitted file with `run_infilling_quality.py --bias-file ...`.
No test captions are used in fitting. The provided frozen files assume the
listed checkpoints, prompt and preprocessing; local storage paths do not need
to match those of the original experiment.

## Reported result

These are the paper's averages across initial lengths 2, 4 and 8 (scores ×100),
not results from a new run of this release:

| Model | Method | BLEU-2 | ROUGE-L |
|---|---|---:|---:|
| LaViDa | Fixed / CAL | 17.3 / 20.4 | 37.9 / 41.5 |
| LLaDA-V | Fixed / CAL | 18.9 / 22.0 | 39.9 / 43.2 |

The result supports average quality improvements in this image-conditioned
text-infilling setting, not a unique optimal reference length or uniform gains
at every initialization.
