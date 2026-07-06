# Contributing to ASSTF

Thank you for your interest in making ASSTF better! We welcome contributions from researchers, developers, and the broader ML community.

This project is dual-licensed under the [LeanAI ASSTF Community License](LICENSE) (free for non-commercial use) and a separate [Commercial License](docs/COMMERCIAL_LICENSE.md). By contributing, you agree that your contributions may be used under both licenses and that LeanAI may relicense your contributions for commercial offerings.

---

## How to Contribute

### Reporting Bugs

1. Check if the issue already exists in [GitHub Issues](https://github.com/safeware/Project_LeanAI/issues).
2. If not, open a new issue using the **Bug Report** template.
3. Include:
   - A clear description of the bug
   - Steps to reproduce
   - Expected vs. actual behavior
   - Your environment (OS, Python version, PyTorch version)
   - A minimal code snippet if possible

### Suggesting Features

1. Open a new issue using the **Feature Request** template.
2. Explain the use case and why the feature would benefit ASSTF.
3. If applicable, propose an API design.

### Contributing Code

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Make your changes.
4. Add or update tests as needed.
5. Run the test suite locally:
   ```bash
   pytest tests/ -v
   ```
6. Run linting and formatting:
   ```bash
   ruff check asstf shared tests
   black asstf shared tests
   mypy asstf shared
   ```
7. Commit with clear messages.
8. Open a Pull Request and fill out the PR template.

### Contributing Real-World Benchmarks

We especially welcome benchmarks on public datasets! To contribute:

1. Add your benchmark script under `benchmarks/<dataset_name>/`.
2. Include a README explaining the dataset, setup, and results.
3. Report mean ± std over at least 3 random seeds.
4. Open a PR and we will add it to the community leaderboard.

---

## Development Setup

```bash
git clone https://github.com/safeware/Project_LeanAI.git
cd Project_LeanAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Code Style

- Python 3.9+
- Format code with `black`
- Lint with `ruff`
- Type hints encouraged for new code
- Docstrings should follow the existing style

---

## Communication

- GitHub Issues: bug reports and feature requests
- GitHub Discussions: questions, ideas, and show-and-tell
- For commercial inquiries: **Bentley@safeware.com.tw**

---

## Attribution

Contributors will be acknowledged in release notes and, with permission, in the project README.
