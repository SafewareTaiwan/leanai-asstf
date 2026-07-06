# GitHub Content Placement Guide

This guide explains where each document should live on GitHub to maximize clarity, discoverability, and star conversion.

---

## Document Map

| Document | Suggested GitHub Location | Purpose |
|----------|---------------------------|---------|
| `README_EN.md` | **Root `README.md`** | Main landing page; the single most important file for stars and conversions. |
| `docs/INTRODUCTION_EN.md` | `docs/INTRODUCTION_EN.md` + link from README "Introduction" section | High-level overview for newcomers and SEO. |
| `docs/ARCHITECTURE_EN.md` | `docs/ARCHITECTURE_EN.md` + link from README "How It Works" | Technical deep-dive for researchers and contributors. |
| `docs/USAGE_EN.md` | `docs/USAGE_EN.md` + link from README "Quick Start" | Practical coding guide for users. |
| `docs/API_EN.md` | `docs/API_EN.md` + link from README "API Reference" | Complete class/function reference. |
| `docs/BENCHMARKS.md` | `docs/BENCHMARKS.md` (translate to English) + link from README "Benchmarks" | Reproducible results and comparisons. |
| `docs/DESIGN_IMPLEMENTATION_REVIEW.md` | `docs/DESIGN_IMPLEMENTATION_REVIEW.md` or internal wiki | Internal review; optionally public for transparency. |
| `docs/PROMOTION_STRATEGY.md` | Internal wiki or `docs/PROMOTION_STRATEGY.md` (optional public) | Marketing and community growth plan. |
| `docs/BUSINESS_ROADMAP.md` | `docs/BUSINESS_ROADMAP.md` (optional public) | Investor and partner-facing commercial strategy. |
| `docs/GITHUB_PLACEMENT.md` | `docs/GITHUB_PLACEMENT.md` | This guide. |

---

## Root-Level Files to Create

| File | Why It Matters on GitHub |
|------|--------------------------|
| `README.md` | Primary conversion page; controls first impression, SEO, and star rate. |
| `LICENSE` | Full Community License text so GitHub's license detector can display it. |
| `LICENSE-COMMERCIAL.md` | Commercial terms; linked from README and `LICENSE`. |
| `CONTRIBUTING.md` | Required for external contributors; sets expectations and CLA. |
| `CODE_OF_CONDUCT.md` | Expected by GitHub and enterprise users. |
| `SECURITY.md` | How to report vulnerabilities privately. |
| `CHANGELOG.md` | Version history and release notes. |
| `ROADMAP.md` | Public product/research roadmap; drives engagement. |
| `pyproject.toml` | Packaging metadata; enables PyPI and `pip install`. |
| `requirements.txt` | Core runtime dependencies. |
| `requirements-dev.txt` | Developer/test dependencies. |
| `requirements-optional.txt` | Optional NLP/audio dependencies. |

---

## `.github/` Directory to Create

| Path | Purpose |
|------|---------|
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Structured bug reports. |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | Structured feature requests. |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR checklist (tests, docs, license). |
| `.github/workflows/ci.yml` | Run pytest, lint, and type checks on every PR. |
| `.github/workflows/benchmark.yml` | Run full benchmarks on schedule or manual dispatch. |
| `.github/CODEOWNERS` | Auto-assign reviewers. |
| `.github/FUNDING.yml` | Sponsor links and commercial licensing contact. |

---

## README.md Section-by-Section Placement

| README Section | Source Content | Notes |
|----------------|----------------|-------|
| Hero banner + badges | `README_EN.md` lines 1–8 | Put at the very top; include license badge. |
| One-liner value prop | `README_EN.md` "Tiny, self-adapting neural layers..." | Keep under 140 characters. |
| Why ASSTF? | `INTRODUCTION_EN.md` Section "Why ASSTF Matters" | Use emojis and bullets for scannability. |
| 30-second demo | `USAGE_EN.md` Section 2 | Make it copy-paste runnable. |
| Installation | `USAGE_EN.md` Section 1 | One-liner PyPI install + source fallback. |
| How It Works | `ARCHITECTURE_EN.md` Section 1 | Include the math snippet and a diagram. |
| Applications grid | `README_EN.md` "The 5+1 Demos" | Link each app folder; add screenshots/GIFs later. |
| Benchmark highlights | `BENCHMARKS.md` summary table | Must be reproducible via `run_all.py`. |
| Quick start (single demo) | `USAGE_EN.md` Section 6 | Lower the barrier to first success. |
| Project structure | `README_EN.md` tree | Helps users navigate. |
| License | `LICENSE` summary | Clear community/commercial split. |
| Citation | `README_EN.md` BibTeX | Critical for academic adoption. |
| Roadmap | `BUSINESS_ROADMAP.md` / `ROADMAP.md` | Shows momentum and future value. |
| Contributing | `CONTRIBUTING.md` summary | Encourage contributions. |
| Keywords | `README_EN.md` bottom | Improves GitHub search indexing. |

---

## Repository Settings Checklist

- [ ] **Repository name:** Consider renaming to `leanai-asstf` or `asstf` for discoverability.
- [ ] **Description:** Use the one-liner from `README_EN.md`.
- [ ] **Topics:** Add all 20 topics listed in `PROMOTION_STRATEGY.md` Section 3.
- [ ] **Social preview:** Upload 1280×640 banner image.
- [ ] **Website URL:** Link to company landing page or docs site.
- [ ] **Discussions:** Enable GitHub Discussions for Q&A.
- [ ] **Releases:** Publish `v1.0.0` with release notes.
- [ ] **Packages:** Publish to PyPI and link in repository sidebar.

---

## Suggested Workflow

1. Replace root `README.md` with `README_EN.md` content.
2. Keep `README.md` in English as the primary README; add `README_ZH.md` only if you want a Chinese mirror.
3. Translate `docs/BENCHMARKS.md` to English and align numbers with `run_all.py`.
4. Remove absolute local paths from all documentation.
5. Create the `.github/` directory and CI workflow.
6. Add root-level community files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`).
7. Upload a social preview image.
8. Publish GitHub Release `v1.0.0` and PyPI package.

---

## Optional: GitHub Wiki vs. `docs/`

For a research project, keeping documentation in the repo under `docs/` is preferable because:

- It is version-controlled alongside code.
- It is searchable from the repository.
- It renders natively on GitHub.

Use GitHub Wiki only for community-curated content (FAQs, tutorials from users).

---

## Summary

Your GitHub repository is both a **product page** and a **documentation hub**. Every document should have a clear home, and the README should act as the central index. Invest in the first 30 seconds of a visitor's experience — that is what converts visitors into stars, contributors, and customers.
