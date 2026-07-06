# ASSTF — Architecture & Technical Explanation

This document explains how ASSTF works under the hood. It maps the white-paper notation to the reference PyTorch implementation and describes the design rationale for each component.

---

## 1. The ASSTF Layer

ASSTF redefines a neural network layer as the composition of two transfer functions:

- **Γ (Gamma)** — the *core* transfer function, analogous to a standard linear or convolutional layer.
- **Ψ (Psi)** — the *structural* transfer function, a low-rank, adaptive modulator.

The layer output is:

```
out = Γ(x) ⊛ Ψ(x)
```

where `⊛` is an element-wise additive composition.

### 1.1 Linear Formulation

For a dense input `x ∈ ℝ^B×N`:

```
base = x · W_c^T + b_c                     (Γ, core transform)
M    = σ(gate) · (U · V)                   (Ψ, structural modulator)
out  = base + x · M^T                      (Γ ⊛ Ψ)
```

Where:

- `W_c ∈ ℝ^O×N`, `b_c ∈ ℝ^O` — core parameters **θc**.
- `U ∈ ℝ^O×R`, `V ∈ ℝ^R×N` — low-rank structural factors **θs**.
- `gate ∈ ℝ` — learnable scalar that gates the structural branch.
- `R ≪ min(N, O)` — structural rank; controls the extra capacity added to the layer.
- `σ(·)` — sigmoid function.

Because `M` is low-rank, the structural branch adds only `R·(N + O)` parameters instead of `N·O`.

### 1.2 Optional Neuron Gate

ASSTF also supports a per-output activity gate `g_i(t)`:

```
g_i(t) = σ(gate_scale_i · mean_d |x_d| + gate_bias_i)
out_i  = out_i · g_i(t)
```

This lets individual neurons become input-dependent, similar to gating mechanisms in recurrent networks.

### 1.3 Convolutional Formulation

`ASSTFConv1d` applies the same principle to 1-D convolutions:

```
base = conv1d(x, W_c, b_c)
K    = σ(gate) · reshape(U · V → [out, in, kernel_size])
out  = base + conv1d(x, K, bias=None)
```

The core convolution remains full-rank; the structural kernel is a low-rank residual.

---

## 2. Parameter Split

| Group | Symbol | Parameters | Role |
|-------|--------|------------|------|
| Core | θc | `weight`, `bias` | Base knowledge |
| Structural | θs | `structural_u`, `structural_v`, `structural_gate`, `alpha`, `zeta`, `beta` | Adaptive topology |
| Gating (optional) | θg | `gate_scale`, `gate_bias` | Input-dependent neuron activation |

The `BilevelTrainer` automatically detects θs by parameter name. In production use, we recommend passing an explicit parameter filter to avoid misclassification.

---

## 3. Training: Bilevel Optimization

ASSTF supports two training modes:

### 3.1 Alternating (Bilevel) Updates

```
for each batch:
    # Phase 1: update core parameters θc
    freeze θs
    forward/backward on θc
    step optimizer_c

    # Phase 2: update structural parameters θs
    freeze θc
    forward/backward on θs
    step optimizer_s
```

This mimics a bilevel program where θc learns the task and θs learns how to adapt the layer topology.

### 3.2 Joint Updates

```
for each batch:
    forward/backward on both θc and θs
    step both optimizers
```

Joint updates are simpler and often work well in practice. The reference implementation supports both.

---

## 4. Soft-Rank Projection

The structural branch can be further regularized by projecting `M` onto a lower-rank manifold. Four backends are provided:

| Method | Description | Cost |
|--------|-------------|------|
| `elementwise` | Soft-threshold individual values; no SVD | Low |
| `full_svd` | Exact SVD + singular-value thresholding | High |
| `randomized_svd` | Fast approximate SVD | Medium |
| `power_iter` | Power iteration for top singular values | Medium |

In the current implementation, projection is disabled by default because exact SVD every forward pass is expensive. The elementwise method is recommended for production edge use.

---

## 5. Inference-Time Adaptation

The most distinctive feature of ASSTF is online adaptation through the `SurpriseMinimizer`.

### 5.1 Objective

At inference time, θc is frozen and θs is updated for a small number of steps to minimize:

```
L = reconstruction_loss(model(x), target) + β · KL_prior(θs)
```

The KL prior is a zero-mean Gaussian regularizer that prevents θs from diverging far from the trained prior.

### 5.2 Supervised vs. Self-Supervised Targets

- **Supervised**: target is a known label or reference output.
- **Self-supervised**: target can be a reconstruction target (autoencoder) or a consistency objective.

For streaming data without labels, the self-supervised path must be carefully designed to provide a non-zero gradient.

### 5.3 Use Cases

- **User personalization**: adapt to a new user's gesture style.
- **Domain shift**: adapt a wake-word model to a noisy car environment.
- **Concept drift**: update an anomaly detector as machine vibration patterns change.
- **Few-shot transfer**: adapt structural parameters to a new class from a few examples.

---

## 6. Design Rationale

### Why a low-rank residual?

Low-rank matrices are parameter-efficient and have well-understood spectral properties. They let ASSTF add adaptive capacity without exploding parameter counts.

### Why gate the structural branch?

The sigmoid gate allows the layer to **selectively disable** the structural branch when it is not needed, preventing over-parameterization on simple examples.

### Why separate core and structural parameters?

Separating θc and θs makes it safe to update θs at inference time without forgetting the base task knowledge stored in θc.

### Why element-wise composition (`⊛ = +`)?

Additive residuals are easy to implement, preserve gradients, and make ASSTF a drop-in replacement for existing layers.

---

## 7. Comparison with Related Work

| Method | Mechanism | Trainable Params | Inference Adaptation |
|--------|-----------|------------------|---------------------|
| Standard fine-tuning | Update all weights | High | No |
| LoRA | Low-rank adapter on frozen weights | Low | No |
| Adapters (Houlsby) | Bottleneck MLP inserts | Low | No |
| BitFit | Bias terms only | Very low | No |
| Prompt tuning | Learn soft prompts | Low | No |
| Test-Time Training (TTT) | Update all weights at test time | High | Yes, expensive |
| TTA (BN stats) | Update normalization stats | Very low | Yes, limited |
| **ASSTF** | Low-rank structural adapter | Low | Yes, focused on θs |

ASSTF sits at the intersection of **parameter-efficient fine-tuning** and **test-time adaptation**.

---

## 8. Implementation Notes

- **Initialization**: The structural branch should start near zero so the layer initially behaves like a standard layer. In the current code, `structural_gate` is initialized to `0`, which yields `σ(0) = 0.5`; this should be changed to a negative value for ideal warm-start behavior.
- **Device support**: All operations are standard PyTorch; ASSTF runs on CPU, CUDA, and MPS.
- **Quantization**: The core branch can be quantized independently; structural parameters are small enough to remain in full precision for adaptation.

---

## 9. Further Reading

- [`docs/INTRODUCTION_EN.md`](INTRODUCTION_EN.md) — high-level overview.
- [`docs/USAGE_EN.md`](USAGE_EN.md) — code examples.
- [`docs/API_EN.md`](API_EN.md) — complete API reference.
- White-paper: `Document/FSDM_ASSTF_V1.0[86455230].pdf`
