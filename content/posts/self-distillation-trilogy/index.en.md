---
title: "Self-Distillation Trilogy: From Continual Learning to Inference Optimization"
date: 2026-02-10T12:00:00+08:00
tags: ["Self-Distillation", "Continual Learning", "Reinforcement Learning", "Knowledge Distillation"]
categories: ["RL"]
draft: false
summary: "Three papers on Self-Distillation explore the core idea of letting models teach themselves from perspectives of continual learning, mathematical reasoning, and code generation"
author: "yellowtown"
lang: "en"
type: "posts"
---

## 1. Background

Recently, three papers on Self-Distillation were released almost simultaneously, exploring the same core idea—**letting models teach themselves**—from three different angles: **continual learning**, **mathematical reasoning**, and **code generation**. These three papers are:

| Paper | Team | Core Scenario | arXiv |
|-------|------|---------------|-------|
| Self-Distillation Enables Continual Learning (SDFT) | MIT + ETH | Continual learning, knowledge acquisition | 2601.19897 |
| Self-Distilled Reasoner (OPSD) | UCLA | Mathematical reasoning | 2601.18734 |
| Reinforcement Learning via Self-Distillation (SDPO) | ETH + MIT + Stanford | Code generation, RL | 2601.20802 |

Interestingly, SDFT and SDPO share common authors (Jonas Hübotter, Idan Shenfeld) and both come from the academic circles of ETH Zurich and MIT, indicating that this is an emerging research hotspot.

---

## 2. Core Objective: Breaking the "External Teacher Required" Assumption

The core assumption of traditional distillation methods is: **there must be a stronger teacher model to guide the student**. This assumption brings several problems:

1. **Teacher unavailable**: In online learning scenarios, powerful external teachers often do not exist
2. **Distribution mismatch**: Offline distillation (SFT) uses fixed teacher-generated data, causing cumulative errors when students reason on their own distribution
3. **Sparse rewards**: RL methods (like GRPO), although on-policy, only have sequence-level scalar rewards with severe information bottlenecks

**The core insight of Self-Distillation**: The same sufficiently strong LLM can simultaneously play the roles of teacher and student by **conditioning on different information**—the teacher sees "privileged information" (answers, environmental feedback, demonstrations), while the student only sees the problem.

---

## 3. Comparison of the Three Methods

### 3.1 SDFT [1]: Continual Learning Scenario

**Problem setting**: Models need to continuously learn new knowledge (e.g., events in 2025) without forgetting old knowledge.

**Self-Teacher design**:
- **Student**: Only sees problem x
- **Teacher**: Sees problem x + demonstration
- **Distillation objective**: Reverse KL divergence D(p_S || p_T)

**Key techniques**:
- EMA (Exponential Moving Average) teacher: prevents student collapse
- Reverse KL: prevents mode collapse, maintains output diversity
- On-policy sampling: trains on the student's own distribution

### 3.2 OPSD [2]: Mathematical Reasoning Scenario

**Problem setting**: Improve LLM's mathematical reasoning ability while maintaining token efficiency.

**Self-Teacher design**:
- **Student**: Rollout generates reasoning paths (cannot see the answer)
- **Teacher**: Sees ground-truth answer y*, generates "ideal" distribution
- **Distillation objective**: Forward KL divergence D(p_T || p_S)

**Key techniques**:
- Answer as privileged information: teacher knows the final answer, generates "correct" reasoning steps
- Token-level supervision: every position has gradient signal (vs GRPO's sparse rewards)
- 8× token efficiency: saves 8× computation compared to GRPO

### 3.3 SDPO [3]: Code Generation Scenario

**Problem setting**: In code generation tasks, utilize environmental feedback (compilation errors, test failures) for reinforcement learning.

**Self-Teacher design**:
- **Student**: Generates code, receives environmental feedback after execution
- **Teacher**: Sees environmental feedback (error messages, stack traces, failed test cases), generates improved code
- **Distillation objective**: Forward KL divergence D(p_T || p_S) + RL reward

**Key techniques**:
- Environmental feedback as privileged information: error messages, stack traces, failed inputs
- Hybrid objective: combines distillation + RL reward
- Breaking information bottleneck: from scalar rewards to token-level supervision

---

## 4. Key Insights: Commonalities and Differences

### 4.1 Common Framework: Conditional Information Gap

The three papers share the same core idea:

```
Student distribution: p_S(y | x)           # Only sees the problem
Teacher distribution: p_T(y | x, z)         # Sees problem + privileged information z

Where z can be:
- SDFT: demonstrations/knowledge snippets
- OPSD: ground-truth answers
- SDPO: environmental feedback/error information
```

### 4.2 Differences in Training Objectives

| Method | KL Direction | Purpose | Applicable Scenario |
|--------|--------------|---------|---------------------|
| SDFT | Reverse KL D(p_S \|\| p_T) | Prevent mode collapse | Continual learning |
| OPSD | Forward KL D(p_T \|\| p_S) | Strengthen high-probability paths | Inference optimization |
| SDPO | Forward KL + RL | Combine feedback and exploration | Code generation |

### 4.3 Role of EMA Teacher [1]

SDFT uses EMA teacher, while OPSD/SDPO use online teachers (i.e., the student itself). This reflects the essential differences between the two scenarios:

- **Continual learning**: Needs to preserve old knowledge, EMA teacher serves as an "anchor"
- **Inference optimization**: Needs to quickly adapt to new tasks, online teacher is more flexible

### 4.4 Logic Behind KL Direction Choice [1][2][3]

- **Reverse KL (SDFT)**: D(p_S || p_T) → student covers teacher, prevents mode collapse (maintains diversity)
- **Forward KL (OPSD/SDPO)**: D(p_T || p_S) → teacher covers student, strengthens high-probability paths (improves certainty)

This reflects the essential differences between the two scenarios: continual learning needs to be "conservative", while inference optimization needs to be "aggressive".

---

## 5. Golden Case: Comparison of Three Mechanisms

### Scenario 1: Learning 2025 New Knowledge (SDFT)

{{< admonition example "Example" >}}
**Problem**: "What disaster happened in Myanmar in 2025?"  
**Demonstration**: "In 2025, a major earthquake occurred in Myanmar, causing widespread destruction..."

**Student** (only sees problem):  
"Myanmar in 2025... let me think... there was an earthquake?"

**Teacher** (sees problem + demonstration):  
Generates distribution based on demonstration, knows key information like "major earthquake", "widespread destruction"

**Training objective**: Let student distribution approach teacher distribution
{{< /admonition >}}

### Scenario 2: Mathematical Problem Solving (OPSD)

{{< admonition example "Example" >}}
**Problem**: "Solve 2x + 5 = 13"  
**Answer**: "2x = 8, x = 4"

**Student rollout**:  
"2x + 5 = 13... subtract 5... 2x = 8... divide by 2..."

**Teacher distribution** (sees answer):  
Knows what the correct answer should be at each token position

**Training objective**: Let Student distribution approach Teacher distribution
{{< /admonition >}}

### Scenario 3: Code Debugging (SDPO)

{{< admonition example "Example" >}}
**Problem**: "Implement separateSquares function"

**Student code**:
```python
def separateSquares(...):
    result = x / denominator  # Line 73
```

**Environmental feedback**:
```
RuntimeError: ZeroDivisionError at Line 73
Last Input: [[26,30,2],[11,23,1]]
```

**Self-teacher** (sees feedback):  
Knows there's a division-by-zero error at line 73, knows specific triggering input

**Training objective**: Let student learn to avoid similar errors
{{< /admonition >}}

---

## 6. Insights: Academic Consensus and Paradigm Shift

### 6.1 Consensus 1: ICL Capability is Prerequisite for Self-Distillation [1][2][3]

All three papers implicitly rely on the same prerequisite: **the model must have sufficient In-Context Learning capability** to extract effective teaching signals from privileged information.

- SDFT [1]: Teacher "understands" knowledge from demonstrations through ICL
- OPSD [2]: Teacher utilizes ground-truth to guide generation through ICL
- SDPO [3]: Teacher diagnoses errors from environmental feedback through ICL

This means self-distillation may **not be suitable for small models**—this creates tension with the recent trend of "small models + large amounts of data".

### 6.2 Consensus 2: On-Policy is Key to Reducing Distribution Mismatch [1][2][3]

All three papers adopt on-policy sampling (training on the student's own rollouts), contrasting with the off-policy approach of traditional SFT.

**Academic consensus is forming**: In the LLM post-training stage, on-policy methods have systematic advantages over off-policy methods, especially in scenarios requiring stable model behavior.

### 6.3 Consensus 3: Dense Signals Outperform Sparse Signals [2][3]

- GRPO's sparse rewards: only sequence-level binary reward
- Self-Distillation's dense signals: every token has gradient signal

OPSD [2] and SDPO [3] jointly point to a conclusion: **when dense supervision signals are available, they should be prioritized**. This is an important correction to the traditional RL paradigm.

### 6.4 Hidden Paradigm Shift: From "External Teacher" to "Self-Enhancement"

| Traditional Paradigm | Self-Distillation Paradigm |
|----------------------|---------------------------|
| Requires stronger external teacher | Model teaches itself |
| Relies on manually labeled data | Utilizes structure of environment/problem itself |
| Training and inference are separate | Training and inference are unified (on-policy) |
| Single-objective optimization | Multi-perspective learning (student/teacher) |

This marks that LLM training is shifting from **"big data-driven"** to **"self-enhancement-driven"**.

### 6.5 Limitations and Open Questions [1][2][3]

1. **Ground-Truth dependency [2]**: OPSD only applies to closed problems with standard answers
2. **Environmental feedback dependency [3]**: SDPO requires environments to provide rich textual feedback
3. **ICL capability threshold [1][2][3]**: Models need to be strong enough to effectively utilize privileged information
4. **Not truly game-theoretic**: Current Teacher-Student structure is "collusion" rather than game, lacking true strategic interaction

### 6.6 Directions Worth Exploring

1. **Introduce adversarial mechanisms**: Let Teacher and Student have different objectives, forming true Nash equilibrium
2. **Replace Ground-Truth [2]**: Use Judge models or self-consistency to estimate answer quality
3. **Multi-solution space exploration**: Allow multiple "correct" trajectories, not just single y*
4. **Extend to open problems**: Explore self-distillation methods that don't require standard answers or environmental feedback

---

## 7. References

The insights analyzed in this article come from the following three papers:

**[1]** Self-Distillation Enables Continual Learning — Idan Shenfeld et al. (MIT + ETH)  
Focuses on continual learning scenarios, proposing self-distillation using demonstration data to solve catastrophic forgetting problems.

**[2]** Self-Distilled Reasoner-On-Policy Self-Distillation — Siyan Zhao et al. (UCLA)  
Focuses on mathematical reasoning scenarios, utilizing ground-truth answers as privileged information, achieving 8× token efficiency.

**[3]** Reinforcement Learning via Self-Distillation-SDPO — Jonas Hübotter et al. (ETH + MIT + Stanford)  
Focuses on code generation scenarios, utilizing environmental feedback (runtime errors, etc.) for self-distillation, breaking through RLVR's information bottleneck.
