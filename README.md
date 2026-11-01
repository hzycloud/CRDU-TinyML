# CRDU TinyML

Code and compact result records for **A Controlled Diagnostic Study of
Calibration Ranking and Decision Utility in Mixed Precision TinyML
Quantization**.

The study concerns compact depthwise-convolutional models for TinyML and edge
AI: CIFAR-10 and source-disjoint Speech Commands, five training seeds per task,
six separate data roles, and exhaustive spaces of 64 and 128 W4/W8 assignments.
The main experiments use fixed-range fake quantization. Host INT8 and STM32
execution are separate engineering checks.

## Contents

| Path | Contents |
| --- | --- |
| `cig_mpq/` | Models, metrics, quantization, calibration, selection and data protocols |
| `scripts/` | Training, exhaustive evaluation, analysis, plotting and MCU utilities |
| `tests/` | 86 unit tests for numerical and protocol behavior |
| `results/` | Compact CSV/JSON analysis records and training histories |
| `figures/` | Seven color manuscript figures |
| `firmware/stm32f103/` | Standalone scalar integer runtime and board sources |
| `docs/REPRODUCIBILITY.md` | Environment, data layout and reproduction commands |
| `SHA256_manifest.json` | File sizes and SHA-256 hashes for this snapshot |

## Quick Checks

Use the archived numerical environment described in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). From the repository root:

```powershell
python -m unittest discover -s tests
python scripts/plot_third_revision.py
python scripts/plot_training_feasibility.py
```

The second command regenerates the seven base plots from compact records. The
third applies the print-legible color layout for Figures 2 and 3. The archived
`figures/` assets retain the manuscript layout, including the conceptual
protocol illustration. Figure 3 remains in color; markers and line styles also
permit comparison without relying on color alone.

## Evidence and Data

This repository does not redistribute raw benchmark images or audio, candidate
logits, feature caches or trained checkpoints. Full prediction-level reanalysis
requires the accompanying manuscript evidence archive. The compact records
permit inspection of all candidate metrics, decisions, paired summaries,
stability diagnostics and plotted training histories without retraining.

Obtain raw data from the original providers:
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and
[Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands).
Their respective terms continue to apply.

## Interpretation

Ranking differences are not themselves evidence of improved held-out utility.
The exploratory stability screen accepted no switch on transfer to speech.
The measured STM32 result is a single-board, eight-unique-example-per-mask
engineering check, not a full test-set or energy evaluation. Candidate-layer
W4 saved 37.2% of the complete firmware image but increased mean network
latency by 4.0% on the scalar Cortex-M3 backend.

## Licensing

No project-source reuse license has been selected for this snapshot. Public
access is not a grant of an open-source license. See [`NOTICE.md`](NOTICE.md).
