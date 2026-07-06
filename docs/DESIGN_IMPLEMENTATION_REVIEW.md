# ASSTF Design & Implementation Review

**Project:** LeanAI / Project_LeanAI  
**Review date:** 2026-07-05  
**Scope:** Core algorithm (`asstf/`), shared utilities (`shared/`), six application demos (`app_01_*` … `app_06_*`), tests, documentation, licensing, and GitHub launch readiness.

---

## 1. Executive Summary

ASSTF (Adaptive State-Space Transfer Function) is implemented as a **modular, drop-in low-rank residual adapter framework** for PyTorch. Its most distinctive feature is the ability to update a small set of structural parameters at inference time via `SurpriseMinimizer`. The code is clean, well-docstringed, and easy to read.

However, **the repository is currently a research/demo reference, not yet a production-ready or scientifically bulletproof open-source release.** Several demos contain correctness or reproducibility gaps, the claimed "bilevel" optimization is not consistently applied, and the benchmark documentation reports numbers that are not reproduced by the default `run_all.py` path.

**Verdict for conditional open source:** Publishable, but only after fixing the critical issues listed in Section 4 and aligning the README/benchmarks with the actual code. Until then, the project risks credibility damage from conflicting claims and broken adaptation paths.

---

## 2. Architecture Review

### 2.1 Core layer design

`ASSTFLinear` (and the analogous `ASSTFConv1d`) is a drop-in replacement for `nn.Linear` (`nn.Conv1d`). The forward pass is:

```
base = x @ W_c^T + b_c                         (Γ, core transfer)
M    = sigmoid(gate) * (U @ V)                 (Ψ, low-rank structural modulator)
out  = base + x @ M^T                          (Γ ⊛ Ψ, element-wise composition)
```

- **Core parameters θc**: `weight`, `bias`.
- **Structural parameters θs**: `structural_u`, `structural_v`, `structural_gate`, `alpha`, `zeta`, `beta`.
- **Neuron gate (optional)**: per-output sigmoid gate driven by mean input activity.

This is mathematically sound as a **parameter-efficient residual adapter**. It is *not*, however, a fundamentally new state-space or spatio-temporal architecture; it is closest in spirit to a jointly trained, inference-adaptable LoRA-style layer.

### 2.2 Training design

`BilevelTrainer` splits parameters by **name heuristics** (`structural_*`, `alpha`, `zeta`, `beta`, `gate_*`) and can update θc and θs either jointly or alternately. The alternating mode is the algorithmic headline, but **only apps 01 and 02 actually call it**; apps 03–06 train θc and θs jointly or with a single optimizer.

### 2.3 Inference adaptation design

`SurpriseMinimizer` freezes θc and runs a few SGD steps on θs to minimize `reconstruction_loss(out, target) + β * KL_prior`. This is the most novel component and a strong differentiation vs. standard LoRA/adapter approaches. Its practical value depends heavily on having a meaningful target signal at inference time.

### 2.4 Projection backends

Four soft-rank projection methods are implemented (elementwise, full SVD, randomized SVD, power iteration). They are disabled by default because full SVD per forward pass is expensive. `ASSTFConv1d` accepts an `svd_method` argument but ignores it.

---

## 3. Strengths

1. **Drop-in API** — `ASSTFLinear`/`ASSTFConv1d` are true `nn.Linear`/`nn.Conv1d` replacements.
2. **Parameter efficiency on wide layers** — strong savings when `structural_rank ≪ min(in, out)` (app 06 FFN replacement is the best example).
3. **Online adaptation concept** — `SurpriseMinimizer` is a defensible, distinctive feature for personalization, drift, and domain shift.
4. **Broad demo coverage** — MLPs, CNNs, LSTMs, transformers, RL, meta-learning.
5. **Readable code** — extensive docstrings tie implementation to white-paper notation.
6. **Dual-licensing intent** — clearly signals commercial monetization path.

---

## 4. Critical Issues Requiring Fixes Before Public Launch

| # | Issue | Location | Severity | Recommended Fix |
|---|-------|----------|----------|-----------------|
| 1 | **Structural gate initializes to 0.5 effect, not near-zero** | `asstf/core.py:83`, `reset_parameters()` | High | Initialize `structural_gate` to a strongly negative value (e.g., `-4.0`) so `sigmoid(gate) ≈ 0` at start, making the layer behave like a plain Linear/Conv. |
| 2 | **App 05 RL adaptation is a no-op** | `app_05_rl/evaluate.py` adapts toward `asstf_policy(s_t).detach()` | **Critical** | Provide a meaningful target (e.g., action from a reference policy, human demonstration, or model predictive controller). |
| 3 | **SurpriseMinimizer forces `model.train()` at inference** | `asstf/adaptation.py:136` | High | Use `model.eval()` during adaptation or let the caller decide; document BatchNorm/dropout handling. |
| 4 | **Self-supervised adaptation loss is broken** | `adaptation.py` target can equal `out.detach()` | High | Remove or fix the self-supervised path; ensure loss has non-zero gradient. |
| 5 | **Bilevel optimization not consistently used** | Apps 03–06 bypass `BilevelTrainer.train_epoch(alternate=True)` | High | Either (a) make all demos use alternating training, or (b) reframe ASSTF as supporting both joint and alternating updates. |
| 6 | **`ASSTFConv1d` ignores `svd_method`** | `asstf/core.py:377`, `core.py:440-445` | Medium | Implement the switch or remove the constructor argument. |
| 7 | **`BilevelTrainer` structural split is name-based and brittle** | `asstf/trainer.py:76-83` | Medium | Expose an explicit `structural_params` filter callback or registry. |
| 8 | **`BilevelTrainer.evaluate` metric branch is dead code** | `asstf/trainer.py:196-204` | Medium | Fix the `metric_fn is None` / `is not None` logic so metrics are returned. |
| 9 | **Accuracy metric broadcasting edge case** | `shared/metrics.py:11-17` | Medium | Assert/reshape labels to 1D before comparison. |
| 10 | **Benchmarks/README report numbers not reproduced by `run_all.py`** | `README.md`, `docs/BENCHMARKS.md`, `results/` | High | Align default scripts with the reported real-data results; add confidence intervals and multiple seeds. |
| 11 | **Most demos use synthetic data only** | Apps 01–05 | High for credibility | Add real public datasets (UCI HAR, Speech Commands, SWaT, Omniglot/miniImageNet, OpenAI Gym). |
| 12 | **PyTorch random seed not fixed** | All `train.py` files | Medium | Set `torch.manual_seed` and `torch.cuda.manual_seed_all` for reproducibility. |

### 4.1 Additional correctness concerns

- **App 01**: pre-adaptation accuracy is already 100%, so the demo cannot show adaptation improvement.
- **App 02**: evaluation adaptation uses ground-truth labels, which is supervised adaptation, not the promised online/unsupervised personalization.
- **App 03**: global F1 is the only meaningful metric; per-window F1 is misleading. Mean reconstruction error is higher for ASSTF than static, which should be explained.
- **App 04**: both ASSTF and static are near chance (~21% on 5-way); the result does not support the few-shot claim.
- **App 06**: default `run_all.py` path falls back to synthetic data and produces worse ASSTF accuracy than static. The best real-data result is from a script not called by `run_all.py`.

---

## 5. Scientific & Academic Rigor

To be "credible and withstand scrutiny," the project needs:

1. **Reproducible benchmarks**
   - Fix random seeds across Python, NumPy, and PyTorch.
   - Report mean ± std over ≥5 seeds.
   - Pin dependency versions in `requirements.lock`.

2. **Proper baselines**
   - App 01: compare with standard CNN/LSTM on real IMU data.
   - App 02: compare with standard audio CNN on Speech Commands or Google Speech Commands.
   - App 03: compare with LSTM-VAE, Deep SVDD, Isolation Forest.
   - App 04: compare with MAML, Prototypical Networks, Matching Networks.
   - App 05: compare with PPO/SAC on standard Gym/MuJoCo tasks.
   - App 06: compare with a real pre-trained BERT-Tiny, not a from-scratch tiny tokenizer.

3. **Ablation studies**
   - Effect of `structural_rank`.
   - Effect of `lr_struct` / `lr_core`.
   - Alternating vs. joint training.
   - With/without `SurpriseMinimizer`.
   - With/without neuron gate.

4. **Diagnostics & visualizations**
   - Effective rank evolution over training.
   - Gate value histograms.
   - Structural gradient norms.
   - Latent space visualizations for anomaly and few-shot tasks.

5. **Real-world datasets**
   - Replace synthetic demos with at least one real dataset per app, or clearly label synthetic results as "toy validation only."

---

## 6. Licensing & Open Source Readiness

The current dual license (Community for non-commercial, Commercial for paid use) is **legally valid but is not "open source" by the OSI definition**. The Community License discriminates against commercial use, which violates Open Source Definition criterion 6.

**Implications:**
- GitHub's license detector will not recognize it.
- Many enterprises, Linux distros, and package indexes will treat it as proprietary source-available software.
- The README repeatedly says "open source," which can be legally misleading.

**Recommendations:**
1. Replace "Open Source License" terminology with **"Community License"** or **"Non-Commercial Source-Available License."**
2. Put the full Community License text in the top-level `LICENSE` file so GitHub can display it.
3. Add SPDX headers to source files.
4. Consider adopting an established source-available license such as **PolyForm Noncommercial 1.0.0** if broad adoption is the goal, or an **AGPL/commercial dual license** if copyleft is acceptable.
5. Add `CONTRIBUTING.md` clearly stating that contributors grant the licensor rights to relicense their contributions for commercial offerings.

---

## 7. Code Quality & Maintainability

| Area | Status | Notes |
|------|--------|-------|
| Type hints | Partial | Core layers use hints; apps are inconsistent. |
| Tests | Weak | 8 unit tests + 6 smoke tests for ~4,700 LOC. No SVD/gate/adaptation tests. |
| Linting | Missing | No `ruff`, `black`, or `mypy` configuration. |
| CI/CD | Missing | No GitHub Actions. |
| Packaging | Missing | No `pyproject.toml` or `setup.py`. |
| Logging | Weak | Apps use `print()`. |
| Documentation | Mixed | Good docstrings; inconsistent language (Chinese in `BENCHMARKS.md`). |

---

## 8. Priority Action Plan

### Must-fix before GitHub launch (P0)
1. Fix structural gate initialization.
2. Fix or remove App 05 RL adaptation no-op.
3. Fix `SurpriseMinimizer` inference `train()` mode and self-supervised loss.
4. Make all demos consistently use bilevel optimization OR reframe the claim.
5. Align `run_all.py` and `BENCHMARKS.md` with actual reproducible results.
6. Add real datasets or clearly scope demos as synthetic-only.
7. Rename license terminology and add full `LICENSE` text.

### High priority after launch (P1)
8. Add `pyproject.toml` and split requirements into core/dev/optional.
9. Add GitHub Actions CI for pytest + lint.
10. Expand tests to cover SVD backends, gates, checkpoint round-trip, metrics.
11. Translate `BENCHMARKS.md` and app READMEs to English.
12. Remove absolute local paths from documentation.

### Medium priority (P2)
13. Add architecture diagram and result visualizations.
14. Add ablation study scripts.
15. Add multi-seed benchmark runner.
16. Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`.

---

## 9. Conclusion

ASSTF has a clear, defensible value proposition as a **parameter-efficient, inference-adaptable residual layer**. The code is clean enough to open source, but the **demos and benchmarks are not yet credible enough to support strong scientific or commercial claims**. Fix the critical correctness issues, replace synthetic-only claims with real data, and tighten the license language before a public launch. Doing so will make the repository far more defensible to investors, reviewers, and potential commercial licensees.
