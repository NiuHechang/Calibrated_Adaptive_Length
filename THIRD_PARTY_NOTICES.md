# Third-party code

CAL-specific code is licensed under Apache-2.0; see [LICENSE](LICENSE).
Existing third-party portions retain their original notices and license terms.

| Files / components | Source | License |
|---|---|---|
| `diffucoder_cal/generation_utils.py`, `dreamcoder_cal/generation_utils.py` | Dream generation utilities distributed with the DiffuCoder / DreamCoder checkpoints | Apache-2.0 |
| LLaDA sampling routines in `llada_cal/llada_cal.py` and shared baseline sampling utilities | [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) | MIT; full text in [LICENSES/LLaDA-MIT.txt](LICENSES/LLaDA-MIT.txt) |
| `daedal_cal/daedal_cal.py` | [Li-Jinsong/DAEDAL](https://github.com/Li-Jinsong/DAEDAL), adapted to infilling | Apache-2.0, with the LLaDA-derived portions retaining their MIT notice |

The two `generation_utils.py` files retain the original notice:

> Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.

CAL adaptations include prefix/suffix-conditioned generation and length search.
Changed files identify the adaptations in their headers. The Apache-2.0 text
is included in the root LICENSE file.

LaViDa and LLaDA-V implementations are loaded from separately installed upstream
projects, as documented in [multimodal/README.md](multimodal/README.md).
HumanEval-Infilling is also an external dependency. Model weights, upstream
repositories, and datasets are not bundled; their licenses remain separate.
