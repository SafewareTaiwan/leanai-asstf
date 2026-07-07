# Changelog

All notable changes to the ASSTF project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-07-05

### Changed

- Expanded PyPI keywords with core and extended SEO terms for better discoverability.

## [1.0.0] - 2026-07-05

### Added

- Initial public release of the ASSTF reference implementation.
- `ASSTFLinear` and `ASSTFConv1d` drop-in PyTorch layers.
- `BilevelTrainer` for alternating core/structural parameter updates.
- `SurpriseMinimizer` for inference-time structural adaptation.
- Six reproducible application demos (gesture, wake-word, anomaly, few-shot, RL, edge NLP).
- Open data-generation scripts for all synthetic demos.
- Real SST-2 validation for the edge NLP demo.
- English-first documentation: Introduction, Architecture, Usage, API, and Open-Source Stages.
- Dual licensing: Community License for non-commercial use, Commercial License for commercial use.
- `pyproject.toml` and PyPI package `leanai-asstf`.
- GitHub Actions CI workflow.
- Community files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`.

### Fixed

- `structural_gate` initialized to a negative value for near-zero structural contribution at startup.
- `SurpriseMinimizer` no longer forces `model.train()` during adaptation.
- Self-supervised adaptation loss now produces meaningful gradients.
- App 05 RL adaptation target changed from self-distillation to policy-gradient objective.
- `BilevelTrainer.evaluate` now correctly returns custom metrics.
- `ASSTFConv1d` now honors the selected `svd_method`.
- `shared.metrics.accuracy` broadcasting edge case resolved.

---

## Unreleased

### Planned

- Real-world benchmarks on public datasets for Apps 01–05.
- Hugging Face `peft` integration.
- Edge deployment examples (Raspberry Pi, Arduino, ONNX).
- arXiv technical report.
- Community benchmark leaderboard.
