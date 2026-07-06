# ASSTF — API Reference

This document provides the complete public API of the `asstf` package.

---

## Package Imports

```python
from asstf import (
    ASSTFLinear,
    ASSTFBlock,
    ASSTFConv1d,
    ASSTFConvBlock,
    BilevelTrainer,
    SurpriseMinimizer,
    count_parameters,
)
```

---

## `ASSTFLinear`

Drop-in replacement for `torch.nn.Linear` with an adaptive low-rank structural branch.

### Signature

```python
ASSTFLinear(
    in_features: int,
    out_features: int,
    structural_rank: int = 4,
    use_neuron_gate: bool = False,
    use_svd_projection: bool = False,
    svd_method: str = "elementwise",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | — | Size of each input sample |
| `out_features` | `int` | — | Size of each output sample |
| `structural_rank` | `int` | `4` | Rank of the low-rank structural modulator |
| `use_neuron_gate` | `bool` | `False` | Enable input-dependent per-output gate |
| `use_svd_projection` | `bool` | `False` | Enable soft-rank projection of structural matrix |
| `svd_method` | `str` | `"elementwise"` | Projection backend: `"elementwise"`, `"full_svd"`, `"randomized_svd"`, `"power_iter"` |

### Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `weight` | `(out_features, in_features)` | Core weight **θc** |
| `bias` | `(out_features,)` | Core bias **θc** |
| `structural_u` | `(out_features, structural_rank)` | Left structural factor **θs** |
| `structural_v` | `(structural_rank, in_features)` | Right structural factor **θs** |
| `structural_gate` | `(1,)` | Gating scalar **θs** |
| `alpha` | `(1,)` | Soft-thresholding scale **θs** |
| `zeta` | `(1,)` | Gradient-damping scale **θs** |
| `beta` | `(1,)` | Sharpness hyperparameter **θs** |
| `gate_scale` | `(out_features,)` | Neuron gate scale (if enabled) |
| `gate_bias` | `(out_features,)` | Neuron gate bias (if enabled) |

### Example

```python
layer = ASSTFLinear(128, 64, structural_rank=4)
x = torch.randn(16, 128)
out = layer(x)  # shape: (16, 64)
```

---

## `ASSTFBlock`

Convenience wrapper: `ASSTFLinear` → activation → dropout.

### Signature

```python
ASSTFBlock(
    in_features: int,
    out_features: int,
    activation: Callable = F.relu,
    dropout: float = 0.0,
    structural_rank: int = 4,
    use_neuron_gate: bool = False,
)
```

### Example

```python
block = ASSTFBlock(128, 64, activation=F.gelu, dropout=0.1)
out = block(x)
```

---

## `ASSTFConv1d`

Drop-in replacement for `torch.nn.Conv1d` with a low-rank structural convolution kernel.

### Signature

```python
ASSTFConv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    structural_rank: int = 4,
    use_neuron_gate: bool = False,
)
```

### Notes

- `groups > 1` is currently not supported.
- The `svd_method` constructor argument is accepted but ignored; projection defaults to elementwise.

### Example

```python
conv = ASSTFConv1d(1, 16, kernel_size=3, structural_rank=2)
x = torch.randn(8, 1, 128)
out = conv(x)  # shape: (8, 16, 126)
```

---

## `ASSTFConvBlock`

Convenience wrapper: `ASSTFConv1d` → BatchNorm → activation → dropout.

### Signature

```python
ASSTFConvBlock(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    activation: Callable = F.relu,
    batch_norm: bool = True,
    dropout: float = 0.0,
    structural_rank: int = 4,
)
```

---

## `BilevelTrainer`

Trainer that updates core (θc) and structural (θs) parameters separately.

### Signature

```python
BilevelTrainer(
    model: nn.Module,
    lr_core: float = 1e-3,
    lr_struct: float = 1e-4,
    weight_decay_core: float = 0.0,
    weight_decay_struct: float = 0.0,
    alternate: bool = True,
    device: str = "cpu",
)
```

### Methods

#### `train_epoch(model, dataloader, loss_fn)`

Run one training epoch.

```python
train_loss = trainer.train_epoch(model, train_loader, F.cross_entropy)
```

#### `evaluate(model, dataloader, loss_fn, metric_fn=None)`

Run evaluation.

```python
val_loss, val_acc = trainer.evaluate(model, val_loader, F.cross_entropy, accuracy)
```

#### `save_checkpoint(path)` / `load_checkpoint(path)`

Save and resume training state.

```python
trainer.save_checkpoint("checkpoints/ckpt.pt")
trainer.load_checkpoint("checkpoints/ckpt.pt")
```

### Parameter Splitting

By default, parameters are classified as structural if their name contains:

- `structural_`
- `alpha`, `zeta`, `beta`
- `gate_scale`, `gate_bias`

For complex models, pass an explicit filter in a future release.

---

## `SurpriseMinimizer`

Online inference-time adapter for structural parameters.

### Signature

```python
SurpriseMinimizer(
    model: nn.Module,
    lr: float = 1e-4,
    beta: float = 0.01,
    max_steps: int = 5,
    prior_std: float = 1.0,
    device: str = "cpu",
)
```

### Methods

#### `adapt(model, x, target=None)`

Perform a few adaptation steps on θs while keeping θc frozen.

```python
adapter = SurpriseMinimizer(model, lr=1e-4, max_steps=5)
loss = adapter.adapt(model, x, target=y_true)
```

If `target` is `None`, the model output is used as its own target (self-supervised mode); this only works if the model architecture provides a meaningful reconstruction or consistency signal.

### Important Notes

- The adapter currently forces `model.train()` during adaptation. For BatchNorm/dropout-sensitive models, call `model.eval()` manually after adaptation or modify the source.
- After adaptation, `SurpriseMinimizer` currently re-enables gradients on all parameters. If you rely on custom frozen parameters, re-apply your freezing logic.

---

## `count_parameters`

Count total and trainable parameters in a model.

### Signature

```python
count_parameters(model: nn.Module) -> Tuple[int, int]
```

### Example

```python
from asstf import count_parameters

total, trainable = count_parameters(model)
print(f"Total: {total:,}, Trainable: {trainable:,}")
```

---

## Shared Utilities

### `shared.baselines`

Reference static baselines for comparison.

```python
from shared.baselines import StaticMLP, StaticCNN1D, StaticAutoencoder
```

### `shared.metrics`

Evaluation metrics.

```python
from shared.metrics import accuracy, f1_score, mse, reconstruction_error, auc_roc
```

### `shared.early_stopping`

Generic early stopping helper.

```python
from shared.early_stopping import EarlyStopping

es = EarlyStopping(patience=10, mode="min")
if es(val_loss):
    break
```

---

## Type Hints & Device Handling

All `asstf` layers are standard `nn.Module` subclasses. They support:

- CPU, CUDA, and MPS devices
- `torch.compile` (PyTorch 2.0+)
- Standard PyTorch data parallel and mixed-precision training

When using `SurpriseMinimizer`, ensure the model and input tensors are on the same device.
