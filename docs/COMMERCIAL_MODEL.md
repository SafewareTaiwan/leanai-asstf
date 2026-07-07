# ASSTF Commercial Licensing Model Recommendation

**Prepared for:** Safeware Technologies Inc., Ltd.  
**Date:** 2026-07-06  
**Objective:** Recommend a commercially reasonable and industry-aligned business model for ASSTF when enterprise customers begin inbound inquiries.

---

## Executive Summary

ASSTF's commercial model should follow the proven **open-core + dual-license** playbook used by MongoDB, Redis, Elastic, Qt, and MySQL. The core principle:

> **The algorithm and reference implementation are open for non-commercial use to drive adoption and credibility; commercial production use requires a paid license that unlocks legal safety, support, and proprietary enterprise tools.**

Recommended revenue mix:

| Revenue Stream | Share | Description |
|----------------|-------|-------------|
| **Commercial Licenses** | 60–70% | Tiered annual licenses based on company size and deployment model |
| **Enterprise Support** | 15–20% | SLA, onboarding, debugging, architecture review |
| **Edge SDK / Professional Services** | 10–15% | Quantization, deployment, customization |
| **Managed Platform** (future) | 0–10% | Cloud API for adaptive inference |

---

## 1. Core Licensing Philosophy

### What Companies Are Paying For

When an enterprise pays for ASSTF, they are not just paying for code. They are paying for:

1. **Legal clearance** to use ASSTF in commercial products without violating the Community License.
2. **IP protection** via patent and copyright license grants.
3. **Risk reduction** through indemnification and warranty clauses.
4. **Support and maintenance** for production deployments.
5. **Competitive advantage** from a unique parameter-efficient, inference-adaptive technology.

### Pricing Anchor: Value-Based, Not Cost-Based

Do not price based on your development cost. Price based on **customer value**:

- **Cost saved**: smaller models → less memory, lower compute, cheaper edge chips.
- **Performance gained**: on-device personalization → better user experience, higher retention.
- **Time saved**: drop-in layer → faster time-to-market vs. building adaptation in-house.
- **Differentiation**: unique capability competitors do not have.

A reasonable starting point is **1–5% of the value the customer captures**, or a flat fee that is small relative to their product's BOM / revenue.

---

## 2. Recommended License Tiers

### Tier 1: Startup License

| Attribute | Detail |
|-----------|--------|
| **Target** | Companies with <$1M ARR or pre-revenue startups |
| **Price** | $2,000–$5,000 per year flat fee |
| **Deployment** | Up to 10,000 devices or 1 production SKU |
| **Includes** | Commercial use license, email support, bug fixes |
| **Excludes** | OEM redistribution, white-label, sublicensing |

**Why this tier:** Build goodwill with early adopters. Many will grow into larger deals.

### Tier 2: Growth License

| Attribute | Detail |
|-----------|--------|
| **Target** | Companies with $1M–$10M ARR |
| **Price** | $15,000–$50,000 per year |
| **Deployment** | Up to 100,000 devices or unlimited SKUs |
| **Pricing basis** | Annual revenue band + deployment model |
| **Includes** | Commercial license, priority support, quarterly roadmap review |
| **Options** | Per-device add-on beyond 100K units |

### Tier 3: Enterprise License

| Attribute | Detail |
|-----------|--------|
| **Target** | Companies with >$10M ARR, Fortune 500, global deployments |
| **Price** | $100,000+ per year, negotiated |
| **Deployment** | Unlimited devices, global deployment |
| **Includes** | Full IP license, indemnification, SLA, dedicated support channel, custom feature development |
| **Terms** | Multi-year contract with annual renewal |

### Tier 4: OEM / Chip Vendor / Platform License

| Attribute | Detail |
|-----------|--------|
| **Target** | Hardware vendors, AI platforms, cloud providers |
| **Price** | Revenue-share (1–5%) or $250,000+ annual license |
| **Use case** | Bundling ASSTF into SDKs, chip toolchains, cloud services |
| **Includes** | Sublicensing rights, co-marketing, roadmap influence |

**Examples:** Edge Impulse partnership, ARM SDK integration, NVIDIA TAO-style bundle.

---

## 3. Deployment-Model Pricing

Within each tier, the unit economics differ by deployment model:

| Deployment Model | Pricing Unit | Typical Range |
|------------------|--------------|---------------|
| **Edge device / IoT** | Per device shipped | $0.05–$0.50 per unit |
| **Mobile app** | Monthly active user (MAU) | $0.001–$0.01 per MAU/month |
| **SaaS / Cloud API** | Per API call or per active user | $0.001–$0.005 per inference call |
| **On-premise enterprise** | Annual license per server/instance | $10,000–$100,000/year |
| **Development / evaluation** | Free trial (30–90 days) | $0 |

**Recommendation:** Offer **annual minimum commitment** rather than pure usage billing. Enterprises prefer predictable costs.

Example:

> "Edge device license: $30,000/year minimum, includes up to 500,000 devices. Additional devices at $0.06/unit."

---

## 4. What Each License Includes

### Commercial License Grant

- Non-exclusive, non-transferable license to use ASSTF in commercial products and services.
- Right to modify and create derivative works for internal use.
- Right to distribute ASSTF as embedded binary within the licensee's product.
- Patent license for ASSTF-related patents held by Safeware.

### Support and Maintenance

| Level | Response Time | Channels | Included in Tier |
|-------|---------------|----------|------------------|
| Community | Best effort | GitHub Issues | Free |
| Standard | 2 business days | Email | Startup/Growth |
| Priority | 24 hours | Email + Slack/Teams | Growth/Enterprise |
| Dedicated | 4 hours | Dedicated Slack + phone | Enterprise |

### Updates

- Access to bug-fix releases.
- Access to minor version updates.
- Major version upgrades may require additional fee or renewal.

### Indemnification

- Enterprise and OEM licenses should include limited indemnification against IP infringement claims.
- Startup/Growth licenses exclude indemnification or include minimal coverage.

---

## 5. Professional Services and Add-Ons

### Edge Deployment SDK

A proprietary toolkit sold separately:

- ONNX conversion optimized for ASSTF layers.
- Quantization-aware adaptation.
- Target-specific kernels (ARM CMSIS-NN, NVIDIA TensorRT, Apple Neural Engine).
- Memory and latency profiling.

**Pricing:** $25,000–$100,000 per platform port, plus annual maintenance.

### Custom Engineering

- Architecture review: $5,000–$15,000
- Custom adaptation strategy design: $10,000–$30,000
- On-site/on-demand training: $3,000–$5,000 per day

### Certification Program

Train and certify engineers on ASSTF deployment:

- Individual certification: $500–$1,000
- Corporate training: $10,000–$25,000

---

## 6. Negotiation Framework

When a company contacts you, ask these questions to determine the right tier:

1. **What is your annual revenue?** → Determines tier.
2. **What is your deployment model?** → Edge/SaaS/on-prem determines pricing unit.
3. **How many devices/users/API calls per month?** → Volume discount.
4. **Do you need to redistribute ASSTF as part of an SDK or platform?** → OEM tier.
5. **What is your timeline to production?** → Urgency affects pricing.
6. **Do you need custom support or indemnification?** → Enterprise tier.

### Standard Discounts

| Situation | Discount |
|-----------|----------|
| Multi-year contract (2–3 years) | 10–15% |
| Prepayment annually | 5–10% |
| Strategic reference customer | 20–30% (first year only) |
| Non-profit / academic spin-off | Case-by-case |

---

## 7. Competitive Benchmarks

| Company | Model | Relevant to ASSTF? |
|---------|-------|-------------------|
| **MongoDB** | SSPL + Commercial dual license | Highly relevant |
| **Redis** | RSAL + Commercial dual license | Highly relevant |
| **Elastic** | ELv2 + Commercial / Cloud | Relevant for SaaS path |
| **Qt** | LGPL + Commercial dual license | Relevant for device/embedded |
| **MySQL** | GPL + Commercial dual license | Classic reference |
| **Hugging Face** | Open core + Enterprise Hub | Relevant for platform path |
| **Edge Impulse** | Free tier + Enterprise | Relevant for edge device path |

ASSTF sits between **Qt** (embedded/device) and **MongoDB/Redis** (dual-license open core). The closest comparable is a parameter-efficient ML layer like LoRA/PEFT, but those are mostly open-source without strong commercial enforcement.

---

## 8. Recommended First Deal Structure

For the first 1–3 commercial customers, use a simple structure to close quickly:

```
Annual License Fee: $30,000
Includes:
- Commercial use of ASSTF in one product line
- Up to 500,000 edge devices or 10,000 MAU
- Standard email support (2 business days)
- Access to all updates for 12 months

Add-ons:
- Indemnification: +$10,000/year
- Priority support (24h): +$10,000/year
- Edge SDK for one platform: +$25,000 one-time
```

This is low enough to reduce friction but high enough to establish value.

---

## 9. Red Lines

Do **not** agree to:

- Exclusive license (unless for very high strategic value).
- Transfer of IP ownership.
- Unlimited sublicensing in Growth/Startup tiers.
- Perpetual license without annual maintenance.
- Royalty-free OEM without minimum commitment.

---

## 10. Practical Next Steps

1. **Create a one-page commercial license summary** for sales conversations.
2. **Set up a CRM** to track inbound inquiries and deal stages.
3. **Prepare a standard Order Form template** with the tiers above.
4. **Train the founding team** on the negotiation framework.
5. **Engage a lawyer** to review the Commercial License before signing the first deal.

---

**Bottom line:** Start with a **tiered annual commercial license** based on company size and deployment model, add **support and edge SDK** as upsells, and reserve **OEM/platform deals** for strategic partnerships with revenue-share or large upfront fees.
