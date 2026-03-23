---
title: "SDFT: Self-Distillation Enables Continual Learning"
date: 2026-03-23T00:00:00+08:00
tags: ["Self-Distillation", "Continual Learning", "Trust-Region"]
categories: ["Reinforcement Learning"]
draft: false
summary: "How can large models continually learn new knowledge without forgetting old knowledge? The SDFT method proposed by MIT and ETH teams enables on-policy continual learning without reward signals through self-distillation."
author: "黄镇"
lang: "en"
type: "posts"
math: true
---

## 1. The Problem: The "Forgetting Dilemma" of Large Models

Large models are static after deployment. GPT-4 doesn't know about yesterday's news, and LLaMA isn't aware of last month's API releases. If fine-tuned on new knowledge, models experience "catastrophic forgetting"—learning the new while forgetting the old.

Traditional solutions each have their flaws:

| Method | Problem |
|--------|---------|
| **On-policy RL** | Requires reward signals, but demonstration data has no rewards |
| **SFT Supervised Fine-Tuning** | Off-policy, drastically changes model distribution → forgetting |
| **Replay Buffer** | Requires storing old data, privacy/storage issues |
| **Parameter Isolation** | Model size grows with number of tasks |

**Core Question**: When only demonstration data is available, how can we obtain the benefits of on-policy learning to avoid catastrophic forgetting?

MIT and ETH Zurich teams proposed the **SDFT** method in the paper "Self-Distillation Enables Continual Learning," providing an elegant solution.

---

## 2. Core Idea: Implementing Self-Distillation via ICL

### 2.1 Same Model, Two Roles

The core insight of SDFT is: **the same model, through different conditioning approaches, can simultaneously play both teacher and student roles**.

```
Student model: π_θ(y | x)           # Only sees the question
Teacher model: π(y | x, c)           # Sees question + demonstration
```

Where `c` is the demonstration, for example:
- Question: "What disaster happened in 2025 in Myanmar?"
- Demonstration: "In 2025, a major earthquake struck Myanmar..."

The teacher, through **ICL (In-Context Learning)** capability, "understands" the demonstration content and generates a more accurate distribution. The student learns this distribution, thereby acquiring new knowledge.

### 2.2 Why Does It Work? ICL Is Not SFT

The key difference lies in: **ICL generates a "soft" probability distribution, not hard labels**.

**SFT's Problem**:
```
Training data: (question, correct_answer)
Goal: Make model precisely copy every token of the correct answer
Result: Model distribution is drastically changed → forgetting
```

**ICL's Advantage**:
```
Teacher output: Has probability on various semantically equivalent expressions
Student goal: Make its distribution **close to** the teacher's distribution
Result: Maintains original distribution structure, just "shifts toward the correct direction"
```

ICL acts as an **implicit reward function**: telling the student "go this direction," rather than "must precisely copy this answer."

---

## 3. Mathematical Principles: Optimal Policy Under Trust Region Constraints

### 3.1 Trust Region Review

The standard Trust Region Policy Optimization (TRPO) formula:

$$\pi_{k+1} = \arg\max_\pi \mathbb{E}_{y \sim \pi}[r(y, x)] - \beta D_{KL}(\pi(\cdot|x) \| \pi_k(\cdot|x))$$

Two objectives:
1. **Maximize reward** — Make the model better
2. **Stay close** — Don't deviate too far from current policy

This formula naturally balances "learning new" and "not forgetting old."

### 3.2 SDFT's Key Assumption

The paper proves an important conclusion: **The demonstration-conditioned policy π(y|x,c) can approximate the optimal policy**.

The paper experimentally verifies two key conditions:

| Condition | Verification Result |
|-----------|---------------------|
| **Optimality** | After seeing the demonstration, teacher achieves 100% accuracy |
| **Minimal Deviation** | KL divergence between teacher and base model is only 0.68 nats (SFT is 1.26 nats) |

This means: **The teacher exactly satisfies Trust Region's two requirements**—both optimal and not deviating too far.

### 3.3 Objective Function Derivation

From Trust Region, we can derive the **implicit reward function**:

$$r(y, x, c) = \log \pi(y|x,c) - \log \pi_k(y|x)$$

SDFT's objective function (reverse KL divergence):

$$\mathcal{L}(\theta) = D_{KL}(\pi_\theta(\cdot|x) \| \pi(\cdot|x,c)) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi(y|x,c)}\right]$$

**Why Reverse KL?**

| KL Direction | Characteristic | Applicable Scenario |
|--------------|---------------|---------------------|
| Forward KL D(p_T \|\| p_S) | Teacher covers student | Inference optimization, pursuing certainty |
| **Reverse KL D(p_S \|\| p_T)** | Student covers teacher | **Continual learning, maintaining diversity** |

Reverse KL encourages the student to explore multiple modes of the teacher, which is crucial for continual learning—we cannot let the model "collapse" into a single response style.

---

## 4. Key Technique: EMA Teacher Mechanism

### 4.1 Why EMA?

If directly using the student as teacher (self-teacher), training will collapse: student and teacher change simultaneously, with no stable "anchor point."

The paper's ablation study verifies this:

| Teacher Type | Effect |
|--------------|--------|
| **EMA Teacher** | ✅ Best, training stable |
| Frozen base model | ❌ Poor performance |
| Student = Teacher | ❌ Training diverges |

### 4.2 EMA Update Formula

$$\theta_{\text{teacher}} = \alpha \cdot \theta_{\text{teacher}} + (1-\alpha) \cdot \theta_{\text{student}}$$

Where α ∈ {0.01, 0.02, 0.05} (typical values).

**Key point: Teacher doesn't compute gradients!** The teacher is only used to compute log probability, not involved in backpropagation.

```python
# Gradients only for student
∇_θ L = E_{y~π_θ}[log(π_θ(y_t) / π(y_t; teacher)) × ∇_θ log π_θ(y_t)]
                                          ↑
                                      stop gradient
```

### 4.3 EMA's Role: Soft Anchor Point

The EMA teacher acts as a **slowly moving anchor point**:
- Updates slowly enough to provide stable reference
- Updates timely enough to reflect student progress
- No need to store old models, memory-friendly

---

## 5. Complete Process: A Knowledge Acquisition Example

### Scenario: Learning About the 2025 Myanmar Earthquake

**Input**:
- Question x: "What disaster happened in 2025 in Myanmar?"
- Demonstration c: "In 2025, a major earthquake struck Myanmar..."

### Step 1: Teacher Model Processing

The teacher sees the complete prompt:

```
<Question>What disaster happened in 2025 in Myanmar?
This is an example for a response to the question:
<Demonstration>In 2025, a major earthquake struck Myanmar, causing widespread damage...
Now answer with a response of your own, including the thinking process:
```

The teacher generates distribution π(y|x,c) through ICL, "knowing" about the 2025 Myanmar earthquake.

### Step 2: Student Model On-Policy Sampling

The student only sees question x (no demonstration), samples from distribution:

```
y ~ π_θ(·|x)
```

Generates response: "In 2025, there was a significant earthquake event in Myanmar..."

### Step 3: Token-Level Gradient Computation

For each token t in generated sequence y:

$$\nabla_\theta L = \sum_t \sum_{y_t \in \mathcal{V}} \log\frac{\pi_\theta(y_t|y_{<t}, x)}{\pi(y_t|y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(y_t|y_{<t}, x)$$

### Step 4: Parameter Update

- Update student parameters θ to make its distribution close to teacher distribution
- Use EMA to update teacher parameters

### Final Effect

| Metric | SFT | SDFT |
|--------|-----|------|
| New knowledge accuracy (exact match) | 80% | **89%** |
| OOD generalization | 80% | **98%** |
| Prior capability retention | 53.4-60.2 | **64.5-65.4** |

---

## 6. Experimental Results: SDFT Comprehensively Outperforms SFT

### 6.1 Knowledge Acquisition Capability

The paper tested SDFT on multiple tasks:

| Task | SFT | SDFT | Improvement |
|------|-----|------|-------------|
| Wikipedia knowledge (exact match) | 80% | **89%** | +9% |
| OOD generalization questions | 80% | **98%** | +18% |
| Science Q&A | 58% | **71%** | +13% |
| Tool Use | 62% | **75%** | +13% |

### 6.2 Prior Capability Retention

This is the most critical metric for continual learning:

| Benchmark | Base | SFT | SDFT |
|-----------|------|-----|------|
| HellaSwag | 65.5 | 53.4 | **64.5** |
| TruthfulQA | 65.5 | 56.0 | **65.4** |
| MMLU | 65.5 | 60.2 | **65.4** |

**Conclusion**: SDFT almost completely preserves prior capabilities while learning new knowledge. SFT causes significant forgetting.

### 6.3 Sequential Task Learning

The paper also tested performance after learning on 3 sequential tasks:

- **SDFT**: All tasks maintain ~100% normalized performance
- **SFT**: Prior task performance drops to near 0

### 6.4 Reasoning Capability Ablation

Testing reasoning capability using Olmo-3-7B-Think model:

| Model | Accuracy | Tokens |
|-------|----------|--------|
| Base | 31.2% | 4612 |
| SFT | 23.5% | 3273 |
| **SDFT** | **43.7%** | 4180 |

**Surprising finding**: SFT is even worse than base model! SDFT not only recovers but significantly surpasses base performance (+40%).

---

## 7. Ablation Studies: Key Design Decisions

### 7.1 KL Estimator Comparison

| Estimator | Effect |
|-----------|--------|
| **Analytical token-level estimator** | ✅ Best performance and stability |
| Token-level estimator | Biased but usable |
| Rao-Blackwellized estimator | Cost not worth the benefit |

### 7.2 Teacher Context Requirements

| Context | Accuracy |
|---------|----------|
| **Complete text + answer** | **89%** |
| Text-only (question only) | 75% |
| Answer-only | Moderate |

**Key finding**: The teacher must see both question and answer to effectively guide the student.

### 7.3 Prompt Design Considerations

The paper emphasizes that the prompt must:
1. Include complete demonstration (question + answer)
2. Clearly instruct "respond with your own answer" to trigger ICL capability
3. Require including thinking process to promote reasoning ability

---

## 8. Comparison with Other Methods

### 8.1 SDFT vs SFT

| Dimension | SFT | SDFT |
|-----------|-----|------|
| Learning method | Off-policy | **On-policy** |
| Objective | Copy hard labels | **Approximate soft distribution** |
| Forgetting degree | Severe | **Mild** |
| New knowledge learning | Fast but unstable | **Fast and stable** |

### 8.2 SDFT vs Traditional Knowledge Distillation

| Dimension | Traditional Distillation | SDFT |
|-----------|-------------------------|------|
| Teacher source | External stronger model | **Self (different conditioning)** |
| Need pretrained teacher | Yes | **No** |
| Teacher update | None | **EMA** |

### 8.3 SDFT vs Video-KTR

Video-KTR is another work on continual learning, adopting a **Token-level selective update** strategy:
- SDFT: Implements global on-policy learning through self-distillation
- Video-KTR: Identifies key tokens, only updates parameters important for current task

The two methods can be complementary: SDFT addresses "how to learn," Video-KTR addresses "which parameters to learn." (For detailed introduction of Video-KTR, please look forward to another blog post.)

---

## 9. Limitations and Future Directions

### 9.1 Current Limitations

1. **ICL Dependency**: Method effectiveness depends on model's ICL capability, may be limited for small models
2. **Hyperparameter Tuning**: EMA coefficient α needs tuning for different tasks
3. **Computational Cost**: Requires two forward passes (student sampling + teacher scoring)
4. **Demonstration Quality Dependency**: Effectiveness depends on demonstration data quality

### 9.2 Interesting Open Questions

1. **Adaptive EMA**: Can we design strategies to dynamically adjust α based on training stage?
2. **Demonstration Selection**: How to select demonstrations most valuable for continual learning? Can active learning be introduced?
3. **Multimodal Extension**: Can SDFT be applied to vision-language model continual learning?
4. **Theoretical Analysis**: What are SDFT's convergence and generalization bounds?

---

## 10. Summary

SDFT proposes an elegant solution answering a core question:

> **When only demonstration data is available, how can we obtain the benefits of on-policy learning?**

The answer is: **Leverage ICL capability to let the model teach itself**.

Core contributions:
1. **Proof**: Demonstration-conditioned policy can approximate optimal policy under Trust Region constraints
2. **Method**: EMA teacher + reverse KL + on-policy sampling
3. **Effect**: New knowledge learning close to Oracle RAG, prior capability retention close to base model

This work represents a trend in LLM training paradigm: from "relying on external teachers" to "self-enhancement." When models are sufficiently powerful, the teacher that knows them best might just be themselves.

---

## References

**[1]** Shenfeld, I., Damani, M., Hübotter, J., & Agrawal, P. (2026). *Self-Distillation Enables Continual Learning*. arXiv:2601.19897.

**[2]** Zhao, S., et al. (2026). *Self-Distilled Reasoner: On-Policy Self-Distillation for Mathematical Reasoning*. arXiv:2601.18734.

**[3]** Hübotter, J., et al. (2026). *Reinforcement Learning via Self-Distillation*. arXiv:2601.20802.