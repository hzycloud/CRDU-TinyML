# Reproducibility

## Environment

The archived training environment used Windows, Python 3.13.15,
PyTorch 2.6.0+cu124, Torchvision 0.21.0+cu124 and an NVIDIA RTX 4070 SUPER.
`requirements-round2.txt` records numerical dependencies; CUDA wheels use the
official PyTorch CUDA 12.4 index. Requirements are an execution record, not a
claim that all package combinations have been rebuilt on a clean machine.
`requirements-mcu.txt` lists additional MCU capture and inspection packages.

## Compact Records

The repository contains:

- `results/round2/analysis_cifar/` and `analysis_speech/`: candidate metrics,
  feasibility counts, selections, paired summaries and stability diagnostics.
- `results/review_revision/cifar_static/seed*/`: CIFAR training histories and
  summaries.
- `results/round2/speech/seed*/`: speech training histories and summaries.
- `results/round2/baselines/allocation_scores.json` and
  `speech_baseline/allocation_scores.json`: archived CLADO-objective adaptation
  scores, not a reproduction of the original paper's benchmarks.
- `results/round2/integer/report.json`: auxiliary host INT8 results.
- `results/stm32f103/hardware_summary.json`: measured MCU resource and timing
  summaries.

These records are copied without numerical changes. Their relative paths retain
the original experimental layout. Historical pilot scripts are included where
needed by imports; their results are not substituted for the main experiments.

## Plotting and Unit Tests

```powershell
python -m unittest discover -s tests
python scripts/plot_third_revision.py
python scripts/plot_training_feasibility.py
```

Plots are written under `results/round2/figures_v3/`. The final two plots use
archived histories and coverage CSVs, with explicit seed/budget markers.

## Prediction-Level Reanalysis

Obtain the manuscript evidence archive and extract it into the repository root,
preserving the `results/` tree. Work in a separate copy to preserve the supplied
snapshot. Candidate logits, checkpoints, role manifests and feature caches are
not bundled with this compact repository.

```powershell
python scripts/analyze_round2.py --task cifar
python scripts/analyze_round2.py --task speech
```

The analysis reads frozen candidate outputs and archived scores. It does not
tune allocations using test labels. Later exploratory comparisons and the
partial prior test exposure of CIFAR seeds 42 and 43 are disclosed in the paper.

## Retraining

Download the original datasets separately. The expected layout is:

```text
datasets/cifar10/extracted/cifar10/train/<class>/*.png
datasets/cifar10/extracted/cifar10/test/<class>/*.png
datasets/speech_commands/extracted/<word>/*.wav
datasets/speech_commands/extracted/validation_list.txt
datasets/speech_commands/extracted/testing_list.txt
datasets/speech_commands/extracted/_background_noise_/*.wav
```

Exact reproduction requires the original image conversion, filenames, class
order and input hashes in the evidence archive. The code expects the extracted
image-folder representation; it does not convert the CIFAR provider's Python
batches automatically.

```powershell
python scripts/run_review_cifar.py --seeds 42 43 44 45 46 --epochs 100 --min-epochs 50 --patience 12 --max-configs 64
python scripts/evaluate_review_uniform_w4.py
python scripts/run_round2_speech.py
python scripts/run_round2_baseline.py
python scripts/run_round2_speech_baseline.py
python scripts/run_round2_integer.py
```

Training can overwrite outputs. Use a separate working copy, inspect `--help`,
and preserve the archived snapshot. The five speech seeds share one source
partition; they are not five independent source splits. Each speech data role
has one background recording, so silence crops are not independent sources.

## STM32

See [`STM32.md`](STM32.md). Compiler, probe tools, original board Flash backups
and unrelated firmware are not included in the public repository.
