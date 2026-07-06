# ASSTF — Usage Guide

This guide shows how to install ASSTF, replace standard PyTorch layers, train models, and adapt them at inference time.

---

## 1. Installation

### From source

```bash
git clone https://github.com/YOUR_ORG/Project_LeanAI.git
cd Project_LeanAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### From PyPI (when published)

```bash
pip install leanai-asstf
```

### Optional NLP dependencies

```bash
pip install -r requirements-optional.txt
```

---

## 2. Replace `nn.Linear` with `ASSTFLinear`

```python
import torch
import torch.nn as nn
from asstf import ASSTFLinear

class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = ASSTFLinear(in_dim, hidden_dim, structural_rank=4)
        self.fc2 = ASSTFLinear(hidden_dim, num_classes, structural_rank=2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = TinyMLP(128, 64, 10)
x = torch.randn(8, 128)
logits = model(x)
```

`ASSTFLinear` is a drop-in replacement: it accepts the same arguments as `nn.Linear` plus `structural_rank`.

---

## 3. Replace `nn.Conv1d` with `ASSTFConv1d`

```python
from asstf import ASSTFConv1d

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ASSTFConv1d(1, 16, kernel_size=3, structural_rank=2)
        self.conv2 = ASSTFConv1d(16, 32, kernel_size=3, structural_rank=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        return torch.relu(self.conv2(x))
```

---

## 4. Train with Bilevel Optimization

```python
from asstf import BilevelTrainer
import torch.nn.functional as F

trainer = BilevelTrainer(
    model,
    lr_core=1e-3,
    lr_struct=5e-4,
    alternate=True,      # alternating θc / θs updates
)

for epoch in range(100):
    train_loss = trainer.train_epoch(model, train_loader, F.cross_entropy)
    val_loss, val_acc = trainer.evaluate(model, val_loader, metric_fn=accuracy)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
```

Use `alternate=False` for simpler joint updates.

---

## 5. Adapt at Inference Time

```python
from asstf import SurpriseMinimizer

adapter = SurpriseMinimizer(
    model,
    lr=1e-4,
    beta=0.01,
    max_steps=5,
)

# Example: adapt to a new user's gestures
for batch in new_user_data:
    # target can be a pseudo-label, reconstruction target, or reference output
    loss = adapter.adapt(model, batch, target=batch)

# After adaptation, run inference as usual
model.eval()
with torch.no_grad():
    predictions = model(new_user_data)
```

### Self-supervised adaptation (autoencoder)

```python
loss = adapter.adapt(model, x, target=x)  # minimize reconstruction error
```

### Supervised adaptation

```python
loss = adapter.adapt(model, x, target=y_true)
```

---

## 6. Run the Built-In Demos

### All demos

```bash
python run_all.py
```

### Quick smoke test

```bash
python run_all.py --quick
```

### Single application

```bash
python app_01_gesture/train.py
python app_01_gesture/evaluate.py
```

---

## 7. Hyperparameter Tuning

| Hyperparameter | Typical Range | Effect |
|----------------|---------------|--------|
| `structural_rank` | 2–16 | Higher rank → more capacity, more params |
| `lr_core` | 1e-4 – 1e-3 | Learning rate for base knowledge |
| `lr_struct` | 1e-5 – 5e-4 | Learning rate for structural adapter; usually smaller than `lr_core` |
| `alternate` | True / False | Alternating updates can improve stability; joint updates are faster |
| `use_neuron_gate` | True / False | Adds input-dependent gating; useful for sparse activity |
| `use_svd_projection` | True / False | Enable soft-rank projection (expensive; use elementwise) |

### Rule of thumb

Start with `structural_rank=4`, `lr_core=1e-3`, `lr_struct=5e-4`, `alternate=True`. Increase `structural_rank` if validation accuracy plateaus early.

---

## 8. Counting Parameters

```python
from asstf import count_parameters

total, trainable = count_parameters(model)
print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
```

---

## 9. Checkpointing

```python
# Save
torch.save(model.state_dict(), "checkpoints/my_model.pt")

# Load
model.load_state_dict(torch.load("checkpoints/my_model.pt"))
```

`BilevelTrainer` also provides `save_checkpoint()` and `load_checkpoint()` methods.

---

## 10. Tips for Production

1. **Start with the core branch only** by initializing `structural_gate` to a negative value so the model behaves like a standard network at the beginning.
2. **Use real validation data** to pick `structural_rank`; too high rank removes the efficiency benefit.
3. **Limit adaptation steps** on-device to bound latency and power consumption.
4. **Quantize the core branch** for edge deployment while keeping structural parameters in float32 for adaptation.
5. **Profile on target hardware** early; ASSTF's theoretical parameter savings must translate to real latency/memory savings.

---

## 11. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| ASSTF performs worse than static baseline | `structural_rank` too high or `lr_struct` too large | Reduce rank and structural LR |
| Adaptation does not improve accuracy | Target signal is too weak or `max_steps` too low | Use supervised target or increase steps |
| Training is slow | SVD projection enabled | Disable `use_svd_projection` or use `elementwise` |
| NaN loss | `lr_struct` too high or structural gate diverging | Clip gradients, reduce LR |
| Memory error on edge device | Core branch too large | Reduce base layer dimensions, keep structural rank small |

---

**Next:** See [`docs/API_EN.md`](API_EN.md) for the complete class and function reference.
