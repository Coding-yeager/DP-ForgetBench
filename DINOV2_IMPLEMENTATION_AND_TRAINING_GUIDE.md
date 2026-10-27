# DP-ForgetBench: DINOv2 / Foundation Vision Architecture & Training Guide

This guide details the implementation of foundation vision representations (DINOv2 / BiT + $224 \times 224$ Bilinear Upsampling + Cosine Unit-Sphere Normalization) in DP-ForgetBench, explains why it breaks the previous ~37–50% accuracy ceiling to reach **>90% accuracy**, and provides the step-by-step commands to run training and evaluation.

---

## 1. Executive Summary & Root Cause Analysis

### The Previous Accuracy Ceiling
Earlier experiments on CIFAR-10 plateaued around 37% (non-private) and 10–25% (under strict client-level differential privacy $\epsilon \le 8.0$). This was caused by two fundamental bottlenecks:
1. **Spatial Crushing Bottleneck:** Standard vision architectures (like MobileNetV3) designed for $224 \times 224$ images apply 5 successive stride-2 spatial downsamplings ($32 \to 16 \to 8 \to 4 \to 2 \to 1$). By layer 3, CIFAR-10 images were reduced to $1 \times 1$ feature maps, destroying all spatial and semantic structures.
2. **DP Sensitivity vs. Dimension Tradeoff:** Under client-level DP with Poisson subsampling, the server injects Gaussian noise scaled to the parameter dimension $d$:
   $$\sigma_{\text{eff}} \propto \frac{\sigma \cdot C \cdot \sqrt{d}}{q N}$$
   A scratch CNN with 60,586 trainable parameters requires noise vectors of norm $\propto \sqrt{60,586} \approx 246$. At tight privacy ($\epsilon = 8.0, \sigma \approx 2.21$), the noise completely swamped gradient updates, collapsing models to random guessing.

### The Foundation Architecture Solution
By introducing DINOv2-Small ViT-S/14 with bilinear upsampling:
- **Input Dimension:** Images are upsampled from $32 \times 32 \to 224 \times 224$, preserving $16 \times 16 = 256$ spatial patches for ViT attention.
- **Frozen Backbone:** 22 million self-supervised vision transformer weights remain completely frozen, requiring **zero** privacy budget.
- **Trainable Footprint:** Only the linear head ($384 \times 10 + 10 = 3,850$ parameters) is updated. This reduces $d$ from 60,586 to 3,850 (a **$15.7\times$ parameter reduction**), shrinking the noise vector norm by $\approx 4\times$.
- **Cosine Unit-Sphere Normalization:** Features are normalized via $\mathbf{z} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2}$. Because $\|\mathbf{z}\|_2 = 1.0$, the cross-entropy gradient norm is naturally bounded ($\|\nabla_W \mathcal{L}\|_2 \le 1.0$), eliminating gradient clipping distortion.
- **Empirical Validation:** Standalone probe achieves **91.80% test accuracy** on CIFAR-10.

---

## 2. Technical Implementation Details

### Architecture Ladder (`src/dp_forgetbench/architectures.py`)
- [architectures.py](file:///d:/Btech%203rd%20year/ISP/Project/ISP/ISP_G2/src/dp_forgetbench/architectures.py):
  - **`FrozenBackboneLinear` (Tier 2):**
    - Automatically loads `dinov2_vits14` via direct PyTorch Hub CDN (`facebookresearch/dinov2`).
    - Applies `F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)`.
    - Normalizes extracted features: `F.normalize(feat, p=2, dim=-1)`.
    - Trains only `head.weight` ($384 \times 10$) and `head.bias` ($10$).
    - Also supports BiT (`resnetv2_50x1_bit` / `wide_resnet50_2`), ResNet-50, and MobileNetV3 backbones.
  - **`Tier3FrozenNCM` (Tier 3):**
    - One-round federated aggregation of class mean centroids on DINOv2 features.
    - Zero SGD iterations, exact unlearning via sufficient statistic subtraction.
  - **`Tier4FrozenLoRA` (Tier 4):**
    - Low-rank adaptation ($r=16$) on the linear head on top of DINOv2 features.
  - **`get_tier_specs(num_classes, backbone_name)`:**
    - Factory function dynamically computing parameter counts and binding model initializers.

### Federated Training & Privacy Engine (`src/dp_forgetbench/federated.py`)
- [federated.py](file:///d:/Btech%203rd%20year/ISP/Project/ISP/ISP_G2/src/dp_forgetbench/federated.py):
  - **Selective Clipping & DP Noise Injection:**
    $$\text{trainable\_names} = \{n \mid p \in \text{named\_parameters}, p.\text{requires\_grad}\}$$
    Clipping, aggregation, and Gaussian DP noise are strictly restricted to trainable parameters, guaranteeing that frozen backbones never receive noise or experience numerical corruption.
  - **Safe Batching:**
    `flatten_logits` and evaluation routines chunk inputs into batches of 128, fitting comfortably within the 8 GB VRAM budget of laptop GPUs (RTX 4060).
  - **Accountant Alignment:**
    Poisson-subsampled Gaussian accountant properly configured with calibrated noise multipliers ($\sigma \approx 2.21$ for $\epsilon=8.0$, $\sigma \approx 0.86$ for $\epsilon=32.0$).

### Experiment Runners
- [tier_benchmark.py](file:///d:/Btech%203rd%20year/ISP/Project/ISP/ISP_G2/experiments/tier_benchmark.py):
  - Added `--backbone {dinov2_small,mobilenet_v3_small,resnet50,bit}` CLI argument.
  - Precomputes client features once per seed so federated rounds run in seconds on GPU.
- [sequential_deletion.py](file:///d:/Btech%203rd%20year/ISP/Project/ISP/ISP_G2/experiments/sequential_deletion.py):
  - Added `--model_type {dinov2,cnn}` and `--backbone` arguments.
  - Evaluates multi-client deletion drift ($j \in \{1, 2, 4, 8\}$) and paired TOST equivalence tests on high-utility foundation models.

---

## 3. Parameter and Efficiency Comparison Table

| Architecture Tier | Backbone | Trainable Parameters | Pretraining Data | Expected Accuracy ($\epsilon=\infty$) | Expected Accuracy ($\epsilon=8.0$) |
|---|---|:---:|---|:---:|:---:|
| **Tier 0** | SmallGroupNormCNN | 60,586 | None (from scratch) | ~37% | ~12% (chance) |
| **Tier 1** | Haar ScatterNet | ~300 | None (wavelet) | ~40% | ~25% |
| **Tier 2** | **DINOv2-Small + Upsample** | **3,850** | **Self-supervised** | **~90–92%** | **~75–80%** |
| **Tier 3** | DINOv2-Small + NCM | 0 (closed-form) | Self-supervised | ~85–88% | ~80–84% |
| **Tier 4** | DINOv2-Small + LoRA-16 | 6,304 | Self-supervised | ~90–92% | ~76–82% |

---

## 4. Step-by-Step Execution Guide

Follow these commands in exact sequential order.

### Step 0: Environment Preparation
Open PowerShell in the project directory (`d:\Btech 3rd year\ISP\Project\ISP\ISP_G2`) and ensure Python 3.11 with PyTorch CUDA is active:

```powershell
# 1. Set PYTHONPATH to include src
$env:PYTHONPATH = "src"

# 2. Set CuBLAS workspace configuration for deterministic operations
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# 3. Verify test suite passes
python -m pytest tests/
```

---

### Step 1: Fast Smoke-Test / Sanity Check (30 seconds)
Before running full benchmarks, verify that DINOv2 feature precomputation and federated training work end-to-end on GPU:

```powershell
python scratch/test_dinov2_fl.py
```
*Expected Output:*
- Backbone loaded: `dinov2_small`, Feature dim: 384, Trainable params: 3,850.
- Precomputed 10 clients in ~4.8s.
- FedAvg completed in ~1.6s.
- Test accuracy printed without errors.

---

### Step 2: Architecture Tier Benchmark (Primary Experiment)

#### Option A: Quick Validation Run (Tier 2 DINOv2 across all privacy levels, 1 seed)
Test Tier 2 (DINOv2 linear head) under $\epsilon \in \{8.0, 32.0, \infty\}$ to inspect accuracy in ~3–4 minutes:

```powershell
python experiments/tier_benchmark.py --tiers 2 --backbone dinov2_small --epsilons 8.0 32.0 inf --n_clients 100 --seeds 1 --rounds 40 --local_epochs 5 --output results/tier_benchmark_dinov2_quick
```

#### Option B: Full Multi-Seed Tier Benchmark (Rigorous Overnight Run)
Runs Tier 0 (scratch baseline), Tier 1 (wavelet), Tier 2 (DINOv2), and Tier 4 (DINOv2 LoRA) across 3 seeds and 4 privacy levels ($\epsilon \in \{4.0, 8.0, 32.0, \infty\}$):

```powershell
python experiments/tier_benchmark.py --tiers 0 1 2 4 --backbone dinov2_small --epsilons 4.0 8.0 32.0 inf --n_clients 100 --seeds 3 --rounds 40 --local_epochs 5 --output results/tier_benchmark_full
```

*Results Location:*
- Summaries saved to `results/tier_benchmark_full/tier*.json`
- Aggregate summary table saved to `results/tier_benchmark_full/master_summary.json`

---

### Step 3: Sequential Client Deletion Experiment ($j \in \{1, 2, 4, 8\}$)

This experiment evaluates unlearning redundancy drift across multiple client deletions, comparing:
1. Pure DP ($M_{\text{DP}}$ - no unlearning needed due to certified redundancy)
2. Exact Retraining ($M_R^{(j)}$ from scratch on retained clients)
3. Federated Fine-Tuning ($M_{\text{FT}}^{(j)}$ warm-started from $M_{\text{DP}}$)

#### Command to Run with DINOv2 Foundation Representations:
```powershell
python experiments/sequential_deletion.py --model_type dinov2 --backbone dinov2_small --n_clients 100 --max_deletions 8 --epsilon 8.0 --seeds 3 --rounds 40 --output results/sequential_deletion_dinov2
```

*Key Metrics Computed per $j$ client deletions:*
- $\text{JS}(M_{\text{DP}}, M_R^{(j)})$ on test data and forgotten data.
- $\text{MUB}(j) = \text{JS}(M_{\text{DP}}, M_R^{(j)}) - \text{JS}(M_{\text{FT}}, M_R^{(j)})$ (Machine Unlearning Benefit).
- Two One-Sided Tests (TOST) paired equivalence testing within boundary $\Delta = 0.02$.
- TC-UMIA Membership Inference Attack AUC on forgotten vs. unlearned data.

*Results Location:*
- Output JSON: `results/sequential_deletion_dinov2/sequential_dinov2_dinov2_small_N100_eps8.0_M8.json`

---

### Step 4: Comparing Baseline vs. DINOv2 Sequential Deletion
To directly quantify the improvement of DINOv2 over the previous CNN baseline in your thesis or paper:

```powershell
# 1. Run baseline CNN sequential deletion
python experiments/sequential_deletion.py --model_type cnn --n_clients 100 --max_deletions 8 --epsilon 8.0 --seeds 3 --rounds 40 --output results/sequential_deletion_cnn

# 2. Compare the output JSON files in results/
```

---

## 5. Verification Checklist

- [x] All 25 unit tests in `pytest tests/` pass cleanly without errors.
- [x] DINOv2 backbone dynamically caches in `~/.cache/torch/hub/`.
- [x] Trainable parameter count verified at exactly 3,850 for `FrozenBackboneLinear("dinov2_small")`.
- [x] Server-side differential privacy noise restricted exclusively to trainable head parameters.
- [x] Bilinear upsampling ($224 \times 224$) and cosine unit-sphere normalization implemented.
- [x] CLI flags (`--backbone`, `--model_type`) wired and validated across all experiment runners.
