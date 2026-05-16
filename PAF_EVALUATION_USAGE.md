# PAF Evaluation Suite - Usage Guide

Comprehensive command-line interface for running Probabilistic Activation Flow (PAF) attribution evaluation tests.

## Overview

The PAF Evaluation Suite provides a unified entry point for conducting four types of attribution evaluation experiments:

1. **Perturbation Tests** - Deletion/Insertion curves (AUDC/AUIC metrics)
2. **Randomization Tests** - Weight randomization sanity checks (Adebayo et al.)
3. **Qualitative Tests** - Visualization and layer-wise analysis
4. **Pointing Game** - Bounding box localization evaluation

## Installation

Ensure PAF is installed with all dependencies:

```bash
pip install -r requirements.txt
```

## Running PAF Evaluation

### Recommended: Use Entry Point Scripts

These scripts automatically handle Python path setup:

**Option 1: Python Entry Point (Cross-platform - Windows, Mac, Linux)**

```bash
# From project root
python paf_eval.py --test-type perturbation --steps 50
```

**Option 2: Bash Script (Mac/Linux)**

```bash
# From project root
chmod +x paf_eval.sh
./paf_eval.sh --test-type perturbation --steps 50
```

### Direct Execution (Requires Correct Directory)

If running directly, you must run from the `src/` directory:

```bash
# From src/ directory
cd src/
python Evaluation/paf_evaluation.py --test-type perturbation --steps 50
```

**Or** set PYTHONPATH:

```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/Evaluation/paf_evaluation.py --test-type perturbation --steps 50
```

## Test Types

### 1. Perturbation Test (Deletion/Insertion)

Measures how much the model's prediction changes when image regions are deleted or inserted.

**What it does:**
- Progressively deletes important patches → measures confidence drop (Deletion curve)
- Progressively inserts important patches → measures confidence increase (Insertion curve)
- Computes AUDC (Area Under Deletion Curve) and AUIC (Area Under Insertion Curve)
- Lower AUDC and higher AUIC indicate better localization

**Command:**
```bash
python src/Evaluation/paf_evaluation.py --test-type perturbation \
  --steps 50 \
  --patch-size 3 \
  --num-samples 100 \
  --paf-modes ABS:tau=1.0 POWER:tau=2.0 NORM:tau=1.0
```

**Parameters:**
- `--steps` (int): Number of deletion/insertion steps (default: 10)
- `--patch-size` (int): Size of patches to delete/insert (default: 3)
- `--num-samples` (int): Number of test images (default: 50)

**Output:**
- AUDC/AUIC curves for each PAF mode
- Comparison plots saved to `--output-dir`
- CSV with quantitative metrics

---

### 2. Randomization Test (Sanity Check)

Validates that the attribution method is actually learning, not just reproducing input gradients.

**What it does:**
- Trains model normally → get attribution heatmap
- Randomizes last conv layer → get attribution heatmap
- Randomizes all weights → get attribution heatmap
- Measures correlation between heatmaps at each randomization stage
- If attribution method is valid, correlation should drop significantly

**Command:**
```bash
python src/Evaluation/paf_evaluation.py --test-type randomization \
  --num-samples 100 \
  --paf-modes ABS:tau=1.0 POWER:tau=2.0
```

**Parameters:**
- `--num-samples` (int): Number of randomization iterations (default: 50)

**Output:**
- Correlation matrices for each randomization stage
- Heatmaps showing attribution consistency
- Pass/Fail verdict for sanity check

---

### 3. Qualitative Test (Visualization)

Generate high-quality visualizations for qualitative analysis.

**What it does:**
- Creates layer-wise heatmap visualizations
- Generates Canny edge alignment maps
- Produces main comparison figure
- Supports multiple PAF modes side-by-side

**Command:**
```bash
python src/Evaluation/paf_evaluation.py --test-type qualitative \
  --visualize-layers \
  --visualize-canny \
  --sample-idx 42 \
  --analyze-misclassification
```

**Parameters:**
- `--visualize-layers`: Generate layer-wise heatmaps
- `--visualize-canny`: Generate Canny edge visualizations
- `--sample-idx` (int): Specific sample to analyze (default: random)
- `--analyze-misclassification`: Analyze wrong predictions instead of correct ones
- `--contrastive`: Use contrastive interpretation

**Output:**
- PNG files: `qualitative_sample_*.png`
- PDF files: `qualitative_main_*.pdf`
- Layer visualizations for detailed inspection

---

### 4. Pointing Game (Bounding Box Localization)

Evaluates how well attribution heatmaps localize objects within bounding boxes.

**What it does:**
- For each image with bounding box annotation:
  - Extract heatmap from attribution method
  - Check if max activation falls inside bounding box
  - Compute hit rate (percentage of correct localizations)
- Compares different PAF modes and baseline methods

**Command:**
```bash
python src/Evaluation/paf_evaluation.py --test-type pointing_game \
  --dataset voc \
  --max-samples 500 \
  --use-baselines \
  --data-path ./data/voc
```

**Parameters:**
- `--dataset-type` (str): Dataset with bounding boxes: `voc`, `imagenet`, `imagenette` (default: `imagenet`)
- `--max-samples` (int): Number of samples to evaluate (default: 300)
- `--use-baselines`: Include Captum and GradCAM baselines for comparison
- `--data-path` (str): Path to dataset root

**Output:**
- Hit rate percentages for each method
- Comparison plots
- Per-class performance breakdown

---

## PAF Modes

Control how PAF distributes attribution through the network using different scoring modes.

**Available modes:**

```
ABS              → |activation × weight|
POWER:tau=X      → |activation × weight|^X (default X=1.0)
NORM:tau=1.0     → Normalized: |â × ŵ| where â=a/max(a)
NORM_POWER:tau=X → |â × ŵ|^X
SIGNED_SPLIT     → LRP-style: separate positive/negative
SIGNED_FULL      → Full signed attribution: activation × weight
```

**Examples:**

```bash
# Test single mode
--paf-modes ABS:tau=1.0

# Test multiple modes
--paf-modes ABS:tau=1.0 POWER:tau=2.0 NORM:tau=1.0 NORM_POWER:tau=2.0

# With custom parameters
--paf-modes POWER:tau=0.5 POWER:tau=1.0 POWER:tau=2.0
```

---

## Common Workflows

### Workflow 1: Full Evaluation Suite

Run all tests on a model:

```bash
# Test 1: Qualitative - see what PAF produces
python src/Evaluation/paf_evaluation.py --test-type qualitative \
  --visualize-layers --visualize-canny --seed 42

# Test 2: Randomization - verify it's not cheating
python src/Evaluation/paf_evaluation.py --test-type randomization \
  --num-samples 50 --seed 42

# Test 3: Perturbation - quantitative evaluation
python src/Evaluation/paf_evaluation.py --test-type perturbation \
  --steps 50 --patch-size 3 --num-samples 100 --seed 42

# Test 4: Pointing game - localization ability
python src/Evaluation/paf_evaluation.py --test-type pointing_game \
  --dataset-type voc --max-samples 500 --use-baselines --seed 42
```

### Workflow 2: Compare PAF Modes

Test if different PAF modes produce different results:

```bash
python src/Evaluation/paf_evaluation.py --test-type perturbation \
  --steps 50 \
  --paf-modes ABS:tau=1.0 POWER:tau=0.5 POWER:tau=1.0 POWER:tau=2.0 NORM:tau=1.0 \
  --num-samples 50 \
  --output-dir ./results/mode_comparison
```

### Workflow 3: Misclassification Analysis

Analyze why the model makes mistakes:

```bash
python src/Evaluation/paf_evaluation.py --test-type qualitative \
  --visualize-layers \
  --analyze-misclassification \
  --sample-idx 100 \
  --output-dir ./results/misclassification
```

### Workflow 4: Reproducible Experiments

With fixed random seed and output directory:

```bash
python src/Evaluation/paf_evaluation.py --test-type perturbation \
  --steps 50 \
  --num-samples 200 \
  --seed 42 \
  --output-dir ./results/exp_v1 \
  --device cuda
```

---

## Global Parameters

Available for all test types:

### Model & Data
```bash
--model          Model name (default: resnet18)
--dataset        Dataset name (default: imagenet)
--data-path      Path to dataset root (default: ./data/imagenet-1k)
```

### System
```bash
--device         Device: auto, cuda, cpu, mps (default: auto)
--seed           Random seed for reproducibility (default: None)
--output-dir     Output directory (default: ./PAF-output)
--debug-level    Verbosity: 0=silent, 1=info, 2=verbose (default: 0)
```

### Visualization
```bash
--visualize-heatmap    Show heatmaps during test
--contrastive          Use contrastive interpretation
```

---

## Output Files

### Perturbation Test
```
PAF-output/
├── deletion_curve.png           # Deletion curves for all modes
├── insertion_curve.png          # Insertion curves for all modes
├── metrics_comparison.csv       # AUDC/AUIC scores
└── heatmaps/
    └── sample_*.png            # Attribution heatmaps
```

### Randomization Test
```
PAF-output/
├── correlation_matrix.png       # Correlation heatmaps
├── randomization_stages.png     # Stage-by-stage comparison
└── sanity_check_report.txt      # Pass/Fail verdict
```

### Qualitative Test
```
PAF-output/
├── qualitative_sample_*.png     # Layer heatmaps
├── qualitative_canny_*.png      # Edge alignment
└── qualitative_main_*.pdf       # Main figure
```

### Pointing Game
```
PAF-output/
├── hit_rates.csv                # Hit rates by method
├── hit_rates_per_class.csv      # Per-class breakdown
└── localization_comparison.png  # Comparison plot
```

---

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'core.paf'

**Problem:** Getting `ModuleNotFoundError: No module named 'core.paf'` when running paf_evaluation.py

**Solution:** Use the provided entry point scripts instead of running directly:

```bash
# ✓ Correct - Use entry point (handles paths automatically)
python paf_eval.py --test-type perturbation

# ✗ Wrong - Running directly causes path issues
python src/Evaluation/paf_evaluation.py --test-type perturbation
```

**If you must run directly:**

```bash
# Option A: Run from src/ directory
cd src/
python Evaluation/paf_evaluation.py --test-type perturbation

# Option B: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/Evaluation/paf_evaluation.py --test-type perturbation
```

**Why this happens:** Python needs to know where the `src/` directory is to find the `core` module. The entry point scripts (`paf_eval.py` and `paf_eval.sh`) automatically add `src/` to Python's search path.

**Problem:** CUDA out of memory error

**Solutions:**
```bash
# Reduce batch size
--num-samples 10

# Use CPU instead
--device cpu

# Enable gradient checkpointing (if supported)
export PAFPY_GRADIENT_CHECKPOINTING=1
```

### Issue: Pointing Game Fails

**Problem:** "Pointing game evaluation failed: module 'Evaluation.core' has no attribute..."

**Solution:** Pointing game requires VOC dataset with bounding boxes
```bash
python src/Evaluation/paf_evaluation.py --test-type pointing_game \
  --dataset-type voc \
  --data-path ./data/voc  # Ensure this path exists
```

### Issue: All PAF Modes Give Same Results

**Problem:** Different modes should produce different attributions, but all look identical

**Debugging:**
```bash
# Check if modes are being parsed correctly
--debug-level 2 --paf-modes POWER:tau=0.5 POWER:tau=2.0

# Run qualitative test to visualize differences
python src/Evaluation/paf_evaluation.py --test-type qualitative \
  --paf-modes POWER:tau=0.5 POWER:tau=2.0 \
  --visualize-layers \
  --debug-level 2
```

### Issue: Different Results on Different Runs

**Solution:** Use `--seed` for reproducibility
```bash
--seed 42
```

---

## Advanced Usage

### Custom Configuration File

Create `paf_config.yaml`:

```yaml
model:
  name: resnet18
  pretrained: true

dataset:
  name: imagenet
  path: ./data/imagenet-1k

paf_modes:
  - mode: ABS
    params: {tau: 1.0}
  - mode: POWER
    params: {tau: 2.0}
  - mode: NORM
    params: {tau: 1.0}

experiment:
  debug_level: 1
  device: cuda
  seed: 42
```

Then run:
```bash
# Custom config support (implement in code if needed)
python src/Evaluation/paf_evaluation.py --test-type perturbation \
  --steps 50 --config paf_config.yaml
```

### Batch Processing Multiple Models

```bash
for model in resnet18 vgg16 densenet121; do
  for steps in 25 50 100; do
    python src/Evaluation/paf_evaluation.py \
      --test-type perturbation \
      --model $model \
      --steps $steps \
      --output-dir ./results/${model}_steps${steps} \
      --seed 42
  done
done
```

---

## Citation & References

If you use PAF Evaluation Suite, please cite:

```bibtex
@article{paf2024,
  title={Probabilistic Activation Flow: A New Method for Attribution-based Explainability},
  author={...},
  journal={...},
  year={2024}
}
```

---

## Support

For issues, questions, or feature requests:
1. Check the Troubleshooting section above
2. Review output logs with `--debug-level 2`
3. Open an issue on GitHub with:
   - Command that failed
   - Error message
   - System info (device, PyTorch version)
   - Sample config/data (if possible)

---

## License

See LICENSE file in repository root.
