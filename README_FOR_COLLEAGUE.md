# Upload Instructions for Colleague

This folder contains the cleaned ASSTF open-source repository ready for upload to GitHub.

## Target Repository

Upload all contents of this folder to:
https://github.com/SafewareTaiwan/leanai-asstf

(Or whatever repository name was created.)

## How to Upload

### Option 1: Git Command Line (Recommended)

```bash
cd /Volumes/LaCie/Temp/Project_LeanAI
git init
git add .
git commit -m "feat: ASSTF v1.0.0 Stage 1 open-source release"
git remote add origin https://github.com/SafewareTaiwan/leanai-asstf.git
git branch -M main
git push -u origin main
```

### Option 2: GitHub Desktop

1. Open GitHub Desktop
2. File → Add Local Repository
3. Select `/Volumes/LaCie/Temp/Project_LeanAI`
4. Publish repository to `SafewareTaiwan/leanai-asstf`

## What is Included

- Core ASSTF framework (`asstf/`)
- Shared utilities (`shared/`)
- 6 application demos (`app_01_gesture/` to `app_06_edge_nlp/`)
- Tests (`tests/`)
- Documentation (`docs/`)
- Visual assets (`assets/`)
- Colab notebook (`notebooks/`)
- Launch copy (`marketing/launch/`)
- Whitepaper PDF (`Document/FSDM_ASSTF_V1.0[86455230].pdf`)
- GitHub CI workflows (`.github/workflows/`)
- pyproject.toml, README.md, LICENSE, etc.

## What is NOT Included (Intentionally Excluded)

- Python virtual environment (`.venv/`)
- Python cache (`__pycache__/`)
- PyTorch checkpoint files (`.pt`, `.pth`)
- Large regenerated data files (`sst2_sentencepiece.npz`, `sst2_spm.model`, etc.)
- Internal strategy documents (`.docx`)
- Internal KOL outreach and analytics files

## After Upload

1. Go to repository Settings on GitHub
2. Upload `assets/social_preview.png` as the Social Preview image
3. Add topics: `pytorch`, `deep-learning`, `edge-ai`, `tinyml`, `parameter-efficient-finetuning`, `continual-learning`, `test-time-adaptation`
4. Enable Discussions: Settings → General → Discussions
5. Publish Release: Create `v1.0.0` release

## Questions?

Contact Bentley.
