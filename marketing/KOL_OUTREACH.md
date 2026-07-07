# KOL Outreach Plan for ASSTF Launch

## Goal

Identify and engage 10–20 relevant AI researchers, practitioners, and content creators before launch so they can provide feedback and potentially amplify the release.

## Outreach Principles

1. **Ask for feedback, not promotion.** People are more likely to respond to a genuine technical question than a promotional request.
2. **Personalize every message.** Reference their specific work.
3. **Give early access.** Offer a private preview link 3–7 days before public launch.
4. **Make it easy.** Include a one-sentence summary, a relevant link, and a clear ask.
5. **No spam.** If they don’t respond, do not follow up more than once.

## Target Categories

### Tier 1 — Efficiency / PEFT Researchers
People who work on LoRA, adapters, quantization, or efficient fine-tuning.

| Name | Platform | Why Relevant | Angle |
|------|----------|--------------|-------|
| Example: Edward J. Hu | X / GitHub | LoRA co-author | "ASSTF adds inference-time adaptation to the low-rank adapter idea" |
| [Add names] | | | |

### Tier 2 — Edge AI / TinyML Practitioners
People building or writing about on-device ML.

| Name | Platform | Why Relevant | Angle |
|------|----------|--------------|-------|
| [TinyML Summit speakers] | X / LinkedIn | Edge deployment focus | "ASSTF reduces model size and adapts on-device" |
| [Add names] | | | |

### Tier 3 — AI Engineering Influencers
People with large followings who discuss practical ML tools.

| Name | Platform | Why Relevant | Angle |
|------|----------|--------------|-------|
| @_akhaliq | X | Retweets AI papers and code | Tag in launch post with concise summary |
| [Add names] | | | |

### Tier 4 — Chinese AI Community
知乎、机器之心、掘金上的作者与 KOL。

| Name | Platform | Why Relevant | Angle |
|------|----------|--------------|-------|
| [Add names] | | | |

## Outreach Template

### Cold DM / Email

```
Subject: Quick technical feedback on ASSTF (parameter-efficient + inference-adaptive layers)

Hi [Name],

I’ve been following your work on [specific paper/project]. We’re building ASSTF, a PyTorch layer that combines low-rank parameter efficiency with inference-time adaptation.

The core idea is similar to LoRA, but structural parameters keep updating after deployment, so the model can personalize to users or adapt to drift without retraining.

We’re launching the open-source reference implementation next week. I’d love to get your quick technical feedback if you have 10 minutes to skim the repo:

https://github.com/SafewareTaiwan/leanai-asstf

No pressure to share — just genuinely curious if the abstraction resonates with problems you’ve seen.

Best,
Bentley
```

### X Reply / Mention

```
@[name] We built ASSTF (Adaptive State-Space Transfer Function) — a drop-in PyTorch layer that combines parameter efficiency with inference-time adaptation. Reminded me of your work on [topic]. Would love your take: https://github.com/SafewareTaiwan/leanai-asstf
```

## Tracking

Use a simple spreadsheet with columns:

| Name | Platform | Status | Sent Date | Response | Action |
|------|----------|--------|-----------|----------|--------|
| | | Not contacted / Contacted / Replied / Shared | | | |

## Notes

- Do not offer payment for shares. This can backfire and look inauthentic.
- If someone shares, thank them publicly and privately.
- Keep a list of people who gave feedback for future updates and acknowledgments.
