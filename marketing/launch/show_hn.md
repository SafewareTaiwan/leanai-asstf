# Show HN: ASSTF – Parameter-efficient, inference-adaptive PyTorch layers

**Title:** Show HN: ASSTF – Parameter-efficient, inference-adaptive PyTorch layers for edge AI

**Link:** https://github.com/SafewareTaiwan/leanai-asstf

**Body:**

Hi HN,

We’re open-sourcing ASSTF (Adaptive State-Space Transfer Function), a drop-in replacement for nn.Linear / nn.Conv1d in PyTorch.

The core idea: instead of freezing a model after training, let each layer keep a small set of structural parameters that can be updated at inference time. This gives you parameter efficiency (like LoRA) plus online personalization / domain adaptation (like test-time training), but in a single layer.

What’s included:
- Reference implementation in PyTorch
- ASSTFLinear, ASSTFConv1d, BilevelTrainer, SurpriseMinimizer
- 6 reproducible demos (gesture, wake-word, anomaly, few-shot, RL, edge NLP)
- Open data-generation scripts + real SST-2 validation
- Colab notebook and full docs

We’re calling this Stage 1: a clean, reproducible foundation. Stage 2 will add real-world public benchmarks and Hugging Face integration.

License: free for research/personal, paid for commercial use.

Would love feedback from the HN crowd on the API, the math, or where you’d most want to see real benchmarks.

GitHub: https://github.com/SafewareTaiwan/leanai-asstf
