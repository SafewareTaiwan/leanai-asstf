# Introducing ASSTF: Adaptive State-Space Transfer Function

**We are open-sourcing ASSTF, a new drop-in adaptive layer for PyTorch that makes neural networks smaller, faster to personalize, and ready for the edge.**

The AI industry is racing toward bigger models, more GPUs, and larger data centers. We believe there is another path: **adaptability beats scale**. A small model that can reconfigure itself at inference time can often outperform a giant static model on real-world, personalized, and resource-constrained tasks.

## What is ASSTF?

ASSTF (Adaptive State-Space Transfer Function) is a neuron-level adaptive framework. Each ASSTF layer is composed of:

- A **core transfer function** Γ that stores base task knowledge.
- A **structural modulator** Ψ that continuously reconfigures the layer's effective state space.
- An **inference-time adaptation** mechanism called `SurpriseMinimizer` that updates only the structural parameters when the model sees new data.

You can replace `nn.Linear` with `ASSTFLinear` in minutes:

```python
from asstf import ASSTFLinear
layer = ASSTFLinear(128, 64, structural_rank=4)
```

## What makes it different?

Unlike LoRA or adapters, ASSTF is:

- **Trainable and adaptable** — structural parameters are updated during training *and* at inference time.
- **Privacy-first** — personalization happens on-device; user data never leaves the device.
- **Edge-ready** — designed for CPUs, microcontrollers, and mobile hardware.

## Stage 1 Release

Today's release is **Stage 1: Grounding Credit**. We are publishing:

- A clean reference implementation of ASSTF in PyTorch.
- Six reproducible demos covering gesture recognition, wake-word detection, anomaly detection, few-shot learning, reinforcement learning, and edge NLP.
- Open data-generation scripts so every synthetic demo can be reproduced exactly.
- Real-world validation on the SST-2 sentiment dataset.
- Comprehensive English documentation and a Colab quickstart notebook.

We clearly label App 01–05 as reproducible proof-of-concepts and App 06 as real-world validation. We welcome community contributions of additional real-dataset benchmarks.

## Licensing

ASSTF is dual-licensed:

- **Community License** — free for personal learning, academic research, and non-commercial use.
- **Commercial License** — required for any commercial product, service, SaaS, or hardware integration.

This model keeps the core open and accessible while funding long-term development through commercial partnerships.

## Try it now

```bash
pip install leanai-asstf
```

Or run it in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safeware/Project_LeanAI/blob/main/notebooks/ASSTF_Quickstart.ipynb)

## What's next?

- Stage 2 will add real-world benchmarks on public datasets, Hugging Face integration, and an academic paper.
- Stage 3 will bring enterprise tools, an edge deployment SDK, and managed inference services.

We invite researchers, developers, and builders to try ASSTF, report issues, and help us prove that the future of AI is not just bigger — it's adaptive.

**GitHub:** https://github.com/safeware/Project_LeanAI  
**Documentation:** https://safeware.github.io/Project_LeanAI/  
**Commercial inquiries:** Bentley@safeware.com.tw
