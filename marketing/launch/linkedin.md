# LinkedIn Launch Post

Today, Safeware Technologies is open-sourcing ASSTF: the Adaptive State-Space Transfer Function.

ASSTF is a new kind of neural network layer for PyTorch. Unlike conventional layers that freeze after training, ASSTF layers keep a small set of structural parameters that can be updated at inference time — enabling on-device personalization, domain adaptation, and concept-drift recovery without sending user data to the cloud.

We believe the future of AI is not just bigger models. It is smaller, adaptive models that run efficiently on edge devices while respecting privacy.

This Stage 1 release includes:
- A clean reference implementation in PyTorch
- Drop-in replacements for nn.Linear and nn.Conv1d
- Six reproducible demos across gesture, audio, anomaly detection, few-shot learning, RL, and NLP
- Real-world validation on the SST-2 dataset
- Full documentation and a Colab quickstart

We are releasing under a dual-license model: free for research and personal use, with commercial licensing available for enterprises.

This is the beginning of a longer journey. Stage 2 will bring public benchmarks and Hugging Face integration; Stage 3 will bring enterprise tools and an edge deployment SDK.

If you are building AI for edge devices, personalized experiences, or sustainable computing, we invite you to try ASSTF and share your feedback.

🔗 GitHub: https://github.com/SafewareTaiwan/leanai-asstf
📓 Colab: https://colab.research.google.com/github/SafewareTaiwan/leanai-asstf/blob/main/notebooks/ASSTF_Quickstart.ipynb
📧 Commercial inquiries: Bentley@safeware.com.tw

#EdgeAI #TinyML #EfficientAI #PyTorch #OpenSource #MachineLearning #ASSTF #LeanAI
