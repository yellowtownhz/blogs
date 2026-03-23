---
title: "Video-KTR: Token Attribution-Driven Video Reasoning Enhancement"
date: 2026-03-23T00:00:00+08:00
tags: ["Reinforcement Learning", "Video Understanding", "Token Selection", "Multimodal LLM", "GRPO"]
categories: ["Reinforcement Learning"]
draft: false
summary: "Video-KTR proposes a modality-aware token-level reinforcement learning framework that identifies three types of key tokens—visual, temporal, and entropy—through counterfactual perturbation, performing policy updates only on key tokens. Surpassing GPT-4o with 42.7% on Video-Holmes benchmark, it provides new insights for fine-grained RL in video reasoning."
author: "Huang Zhen"
lang: "en"
type: "posts"
math: true
---

## Introduction: The Granularity Dilemma in Video Reasoning

Video understanding tasks pose unique challenges to multimodal large language models (MLLMs): they must not only "comprehend" visual content but also understand temporal evolution of events, and finally perform complex reasoning. Traditional reinforcement learning methods (such as GRPO) rely on sequence-level reward signals. While they can improve overall performance, they struggle to finely shape the model's reasoning capabilities.

The core problem lies in: **not all tokens are equally important**. When the model answers "What did the person do after entering the room?", words like "person", "after", "entering" directly relate to visual perception and temporal reasoning, while other words are less critical. If we can identify these key tokens and perform reinforcement learning updates only on them, we can achieve more precise policy shaping.

This is the core idea of Video-KTR (Key Token Attribution for Video Reasoning). This ICLR 2026 work is the first to introduce **modality-aware attribution** into video reasoning RL, implementing token-level selective updates through three complementary attribution signals.

## Core Method: Design of Three Attribution Signals

The key innovation of Video-KTR lies in designing three complementary token attribution signals, capturing visual dependency, temporal dependency, and reasoning uncertainty respectively.

### 1. Visual-Aware Tokens: Visual Perception Attribution

**Objective**: Identify which tokens' generation depends on visual input.

**Method**: Counterfactual perturbation—measure logits changes after masking video input.

$$
\Delta_{\text{vis}}^i = \left| \log \text{softmax}(z_i^{\text{full}})_{y_i} - \log \text{softmax}(z_i^{\text{masked}})_{y_i} \right|
$$

Where $z_i^{\text{full}}$ is the logits under normal video input, and $z_i^{\text{masked}}$ is the logits after masking the video. The larger the change, the more sensitive the token is to visual input.

**Intuitive Understanding**: When video is masked, probabilities of words describing visual content like "person", "door", "blue" will significantly decrease—these are Visual-Aware Tokens.

### 2. Temporal-Aware Tokens: Temporal Perception Attribution

**Objective**: Identify which tokens depend on the temporal order structure of the video.

**Method**: Frame Shuffling—measure logits changes after randomly shuffling frame order.

$$
\Delta_{\text{temp}}^i = \left| \log \text{softmax}(z_i^{\text{ordered}})_{y_i} - \log \text{softmax}(z_i^{\text{shuffled}})_{y_i} \right|
$$

**Intuitive Understanding**: When frame order is shuffled, probabilities of words describing event sequence like "first", "then", "appear" will decrease—these are Temporal-Aware Tokens.

### 3. Entropy-Aware Tokens: Entropy Perception Attribution

**Objective**: Capture the model's prediction uncertainty, identifying reasoning keypoints.

**Method**: Calculate prediction entropy for each token.

$$
\mathcal{H}(i) = -\sum_w p(z_i = w) \log p(z_i = w)
$$

**Intuitive Understanding**: High-entropy tokens like "however", "wait" mark discourse transitions or reasoning keypoints, where the model has higher uncertainty.

### Complementarity of Three Signals

Ablation experiments reveal the complementary relationship among three attribution signals:

| Attribution Signal | Primary POS | Typical Words | Captured Dimension |
|--------------------|-------------|---------------|---------------------|
| Visual-Aware | NOUN (24.8%) | person, door, blue | Perception dependency |
| Temporal-Aware | VERB (21.2%), PRON (11.0%) | first, then, appear | Temporal dependency |
| Entropy-Aware | ADV (8.8%) | however, wait | Reasoning uncertainty |

Key finding: **Using Temporal-Aware alone yields the largest gain on Video-Holmes (+3.3), but combining V+E+T is most stable across all benchmarks**. This indicates that three signals capture different dimensions of reasoning requirements, none dispensable.

## Token Selection and Policy Update

### Key Token Merging

Token sets identified by three attribution signals are merged through union:

$$
S = S_{\text{vis}} \cup S_{\text{temp}} \cup S_{\text{ent}}
$$

Experiments show that selecting Top 20% tokens is the optimal strategy; too high introduces noise.

### Policy Update

Using binary mask mechanism, gradient updates are performed only on key tokens:

$$
J_{\text{Video-KTR}}(\theta) = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} m_{i,t} \cdot \min\left( r_{i,t} \cdot \hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \cdot \hat{A}_{i,t} \right) \right]
$$

Where $m_{i,t} \in \{0, 1\}$ is the binary mask, being 1 only at key tokens.

### Computational Overhead

Each response requires 2 additional forward passes:
- 1 for visual masking
- 1 for frame shuffling

Overall computational overhead is about 3x of Vanilla GRPO, but rollout count remains unchanged (G=8).

## Experimental Results

### Main Experiment: Surpassing GPT-4o

Video-KTR comprehensively outperforms baseline methods on 5 benchmarks:

| Method | Video-Holmes | VideoMMMU | MMVU | TempCompass | VideoMME |
|--------|--------------|-----------|------|-------------|----------|
| Base Model | 36.5% | 41.2% | 38.9% | 62.1% | 58.3% |
| GRPO | 39.8% | 44.5% | 41.2% | 65.8% | 60.1% |
| **Video-KTR** | **42.7%** | **46.8%** | **43.1%** | **68.4%** | **62.5%** |
| GPT-4o | 42.0% | - | - | - | - |

**Core Breakthrough**: Video-KTR surpasses GPT-4o (42.0%) with 42.7% on Video-Holmes.

### Key Findings from Ablation Experiments

1. **Hard selection outperforms soft weighting**: Binary Top-20% strategy is optimal
2. **20% update ratio is best**: Too high introduces noise
3. **Perturbation strength is robust**: Specific strength of frame shuffling/visual masking has limited impact on results

## Case Study: Key Token Identification in Video QA

Take the question "What did the person do after entering the room?" as example:

```
1. Visual-Aware Token: "person"
   → Probability drops 0.35 after masking video
   
2. Temporal-Aware Tokens: "after", "entering"
   → Probability drops 0.28 after shuffling frames
   
3. Entropy-Aware Token: "did"
   → High entropy 2.7

Merged key token set: {person, after, entering, did}
Perform reinforcement learning updates only on these tokens
```

This example clearly demonstrates how three attribution signals work together: Visual-Aware locks onto visual entities, Temporal-Aware captures temporal relations, Entropy-Aware marks reasoning keypoints.

## Comparison with Related Work

### Relationship with Entropy Selection Methods

Wang et al. (2025) found that high-entropy tokens ("forking tokens") are key to LLM reasoning. Video-KTR builds on this by introducing modality awareness:

| Dimension | Wang et al. 2025 | Video-KTR |
|-----------|------------------|-----------|
| Attribution Signal | Entropy only | Entropy + Visual + Temporal |
| Applicable Scenario | Text-only LLM | Video MLLM |
| Modality Awareness | ❌ None | ✅ Yes |

Ablation experiments show that in video reasoning, **using entropy selection alone misses important temporal dependency tokens** (Temporal-only scores 2.6 points higher than Entropy-only on Video-Holmes).

### Comparison with SDFT

SDFT (Self-Distillation from Foundation Model) leverages environmental feedback through self-distillation, while Video-KTR utilizes modality attribution signals. Both are different paths for RL signal enhancement and can be complementary.

## Methodological Insights

### 1. Effectiveness of Counterfactual Analysis

Revealing causal dependencies through "perturbation → measure change" is a concise and effective analysis method:
- Visual masking → Visual dependency
- Frame shuffling → Temporal dependency
- Probability distribution → Uncertainty

### 2. Universal Value of Selective Optimization

"Selectivity" in reinforcement learning is key to performance improvement—not all tokens/samples/gradients are equally important:

| Work | Selection Dimension | Implementation |
|------|---------------------|----------------|
| Video-KTR | Token selection | Modality attribution signals |
| RAL | Attention selection | Attention distribution |
| VC-STaR | Sample selection | Visual contrast filtering |
| SDPO | Gradient selection | Self-distillation |

### 3. Value of Interpretability

Token-level attribution reveals "what the model focuses on", providing handles for debugging. High Visual-Aware tokens concentrate on nouns, high Temporal-Aware tokens concentrate on verbs and pronouns—this linguistically meaningful correspondence enhances the method's credibility.

## Limitations and Future Directions

### Limitations

1. **Computational overhead**: Requires additional forward passes to compute attribution scores
2. **Hyperparameter sensitivity**: Top-K ratio needs tuning
3. **Task-specific**: Designed for video reasoning, adaptation needed for other modalities

### Future Directions

1. **Adaptive K value**: Dynamically adjust selection ratio based on sample difficulty
2. **Cross-modal attribution**: Extend to image-text and other scenarios
3. **Online attribution**: Real-time computation during inference to reduce training overhead
4. **Combination with other RL algorithms**: Such as combining with RAL's attention optimization

## Conclusion

The core contribution of Video-KTR lies in proposing the first **modality-aware token-level reinforcement learning** framework, achieving fine-grained policy shaping through three complementary attribution signals (visual, temporal, entropy). Surpassing GPT-4o on Video-Holmes demonstrates the effectiveness of token-level selective updates.

This work opens new directions for reinforcement learning in video reasoning: from coarse-grained sequence-level rewards to fine-grained token-level attribution, from single signals to multi-modal awareness. The core insight—**identify and focus on key tokens**—has universal methodological significance and is expected to inspire selective optimization strategies in more domains.

---

**References**:

- Wang, Z., et al. (2026). Video-KTR: Reinforcing Video Reasoning via Key Token Attribution. ICLR 2026.
- Wang, et al. (2025). Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning.