# Reddit r/MachineLearning Post

**Title:** [P] ASSTF: Adaptive State-Space Transfer Function – drop-in PyTorch layers with test-time adaptation

**Body:**

We just open-sourced ASSTF, a new parameter-efficient adaptive layer for PyTorch.

TL;DR: ASSTF replaces nn.Linear / nn.Conv1d with a core transfer function + low-rank structural modulator. The structural parameters can be updated both during training and at inference time via a `SurpriseMinimizer`, giving parameter efficiency + online adaptation in one layer.

**Key differences vs. LoRA / adapters:**
- LoRA is parameter-efficient but static after training.
- Adapters insert extra modules; ASSTF replaces the layer itself.
- ASSTF updates structural params at test time for personalization, drift, or domain shift.

**What’s in the repo:**
- Reference PyTorch implementation
- 6 reproducible demos (gesture, wake-word, anomaly, few-shot, RL, edge NLP)
- Synthetic demos with open generation scripts + real SST-2 validation
- Colab notebook and docs

**License:** free for non-commercial research, paid commercial license.

We’re treating this as Stage 1: reproducible reference + proof-of-concepts. Stage 2 will add public benchmarks, Hugging Face integration, and a paper.

Happy to answer questions and take suggestions on which real datasets or baselines would be most convincing.

GitHub: https://github.com/SafewareTaiwan/leanai-asstf
Colab: https://colab.research.google.com/github/SafewareTaiwan/leanai-asstf/blob/main/notebooks/ASSTF_Quickstart.ipynb
