---
weight: 1
title: "Qwen3Guard's Key Technical Details"
date: 2025-09-29T12:00:00+08:00
lastmod: 2025-09-29T12:00:00+08:00
tags: ["Qwen"]
categories: ["general"]
draft: false
description: "First post in English."
lang: "en"
type: "posts"
summary: "In this post we summarize the key technical insights of Qwen3Guard …"
thumbnail: "arch.png"
author: "yellowtown"
images: []
resources:
- name: "featured-image"
  src: "arch.png"

lightgallery: true

toc:
  auto: false
---


## Why It Matters
The Qwen team introduced **Qwen3Guard**, a specialized safety model designed to improve content filtering and to integrate with reinforcement learning (RL).  
Two aspects stand out:  
- Its training recipe offers lessons for building small domain-specific classifiers (e.g., for quality, style, or topic detection).  
- Its integration with RL hints how multiple models can coordinate in agent-like systems.  

## Core Design
Qwen3Guard attaches **two classification heads** on top of Qwen3:  
- **Prompt head**: classifies whether the *query* itself is safe.  
- **Response head**: performs **token-level safety classification** during generation. This allows unsafe content to be flagged at the very first risky token—enabling real-time moderation.  

## Subtle but Important Details

1. **Safety taxonomy**: Unsafe categories are explicitly defined, each with a one-sentence description.  

![image.png](safety_cls.png)

2. **Training setup**: Pure SFT on **1.19M samples**, no DPO or RL.  

3. **Prompt structure**:  Task description, Safety policy, Unsafe category list, User dialogue and Output constraints

![image.png](prompt.png)

4. **Synthetic data via Self-Instruct**:  
    - Expanded seed data based on a safety taxonomy.  
    - Used synonym sets (e.g., *bomb/C4/TNT/black powder*) to enrich unsafe prompts.  
{{< admonition >}}
Similar to how *K2-think* uses a planner to extract key query concepts for better model understanding.  
{{< /admonition >}}
    - Built positive/negative pairs (e.g., *how to make a bomb* vs. *how to make a cake*) to avoid keyword shortcuts.  
{{< admonition >}}
Similar to *path-patching* with Xr vs. Xc examples.  
{{< /admonition >}}
    - Since standard instruct models rarely output unsafe responses, **Qwen2.5-72B-Base** was used to generate unsafe samples, alongside outputs collected from reasoning models.  

   - For annotation:  
     - Human-labeled a subset  
     - Added voting with models like **Qwen2.5-72B-Instruct** and **Qwen3-235B-A22B**  
     - Achieved **0.9 F1 score** on the human-labeled dataset.  

   - For multilingual data: translations were produced with **Qwen-MT**, and evaluated by detecting *language mixing*, *LLM judges*, and *random human sampling*.  

5. **Challenges in SFT**:  
- *Class imbalance*: “safe” dominates; “controversial” is rare.  
- *Label noise*: even human annotations are inconsistent.  

![image.png](data-1.png)

**How to Fix imbalance**:  
- Observation: the ratio of safe/unsafe in training data shifts the decision boundary.  
     - More safe → model more permissive  
     - More unsafe → model stricter  
   - For *controversial* cases:  
     1. Split data into two sets A and B.  
     2. Train two models on A with different safe/unsafe ratios (one “loose”, one “strict”).  
     3. Use them to vote on B. Disagreements → *controversial*.  
     4. Validation data was used to calibrate the balance (roughly 2:8 or 3:7).  
   - Results confirmed introducing *controversial* improved classification.  

![image.png](data-2.png)


**How Fix noise**:  
  - After resolving controversial scarcity, split data again.  
  - Train **Qwen3-32B** on one split, use it to relabel the other.  
  - This reduced noise and yielded measurable gains (~<1 point).  

![image.png](data-3.png)  

{{< admonition tip >}}
From a practical perspective: these two data-handling tricks can be skipped for the baseline version, and added later for incremental improvements. Moreover, since no end-to-end RL experiments were conducted, their final impact on RL outcomes is unclear.  
{{< /admonition >}}

6. **RL usage**: GSPO with **Qwen3-4B**, trained on 13.7k reasoning and 6.7k non-reasoning samples.  

   Two reward designs:  
   - **Guard-only**:  
     - Reward = 1 if guard predicts safe  
     - Reward = 0 if unsafe or controversial  
     - Problem: model learns to refuse answering everything (safe refusal always yields 1).  

     ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8oLl952yojkaNlap/img/45370057-7772-40f2-9769-c95add8b649e.png)  

   - **Mixed reward** (Guard + WorldPM):  
     - Unsafe → reward ≤ -10  
     - Refusal → reward ≤ -5 (often lower since WorldPM rates low helpfulness)  
     - Safe & helpful → reward ≈ WorldPM score  

     ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/da20239e-e388-426b-87d3-494fc4dcfd62.png)  

7. **General capability evaluation**: Arena-Hard-v2 (alignment), AIME-25 (math), LiveCodeBench-V6 (coding), GPQA (knowledge).  

8. **Token-level classification challenges**:  
   - Step 1: **Rollout voting**  
     - For token $S_i$, rollout completions with prefix ≤ i.  
     - Use Qwen3Guard-Gen to classify.  
     - If >85% unsafe/controversial, label token unsafe.  

     ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/edb48cb7-2467-474e-9534-1d80754f47c9.png)  

   - Issue: overestimation—sometimes safe tokens flagged unsafe due to unsafe continuations.  
   - Fix: ask **Qwen3-235B-A22B** as LLM judge on prefix to reassess.  

   - Final rule:  
     - For unsafe/controversial sample-level cases:  
       - Tokens after unsafe detection → unsafe/controversial  
       - Tokens before → safe  

     ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/2b6548f6-58f9-41a2-86ac-0214d740bc63.png)  

9. **Classification heads**: 4 in total.  
   - Query heads: safe/unsafe classification + unsafe category classification  
   - Response heads: token-level classification  

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/bd0e7dfb-fc0e-40f2-ac7f-d618c0713195.png)  

   - Query loss = sum of two cross-entropy losses  

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/b5bfb1fc-a486-4e89-8700-55247bb0ec79.png)  

   - Response loss = average cross-entropy across tokens  

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/704b4322-2510-44c4-93da-b900d50d2730.png)  

   - Trick: *q-cat* and *r-cat* losses are conditional—only applied when labels are unsafe/controversial, ignored if safe.  

## Takeaways

1. Training specialized classifiers borrows heavily from traditional ML: **ensemble, voting, distillation, cross-validation**.  
2. For distillation, a mid-sized model like **Qwen3-32B** suffices as teacher.  
3. Token-level labeling is especially interesting: it echoes Neel Nanda’s CoT interpretability work, combined with LLM-based re-judging.  
4. Large-scale reliance on LLMs for data processing—generation, labeling, filtering—is becoming standard practice.