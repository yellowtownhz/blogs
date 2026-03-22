---
title: "Video-KTR：Token 归因驱动的视频推理强化"
date: 2026-03-23T00:00:00+08:00
tags: ["强化学习", "视频理解", "Token选择", "多模态LLM", "GRPO"]
categories: ["强化学习"]
draft: false
summary: "Video-KTR 提出了一种模态感知的 Token 级强化学习框架，通过反事实扰动识别视觉、时间、熵三种关键 Token，仅对关键 Token 进行策略更新。在 Video-Holmes 基准上以 42.7% 超越 GPT-4o，为视频推理的精细化 RL 提供了新思路。"
author: "黄镇"
lang: "zh"
type: "posts"
---

## 引言：视频推理的粒度困境

视频理解任务对多模态大语言模型（MLLM）提出了独特挑战：不仅要"看懂"视觉内容，还要理解时间维度的事件演变，最后进行复杂的推理。传统的强化学习方法（如 GRPO）依赖序列级奖励信号，虽然能提升整体性能，但难以精细地塑造模型的推理能力。

核心问题在于：**不是所有的 Token 都同等重要**。当模型回答"What did the person do after entering the room?"时，"person"、"after"、"entering" 这些词直接关联到视觉感知和时间推理，而其他词则不那么关键。如果能识别出这些关键 Token，只对它们进行强化学习更新，就能实现更精准的策略塑造。

这就是 Video-KTR（Key Token Attribution for Video Reasoning）的核心思想。这篇 ICLR 2026 的工作首次将**模态感知归因**引入视频推理 RL，通过三种互补的归因信号实现 Token 级选择性更新。

## 核心方法：三种归因信号的设计

Video-KTR 的关键创新在于设计了三种互补的 Token 归因信号，分别捕捉视觉依赖、时间依赖和推理不确定性。

### 1. Visual-Aware Tokens：视觉感知归因

**目标**：识别哪些 Token 的生成依赖于视觉输入。

**方法**：反事实扰动——遮蔽视频输入后测量 logits 变化。

$$
\Delta_{\text{vis}}^i = \left| \log \text{softmax}(z_i^{\text{full}})_{y_i} - \log \text{softmax}(z_i^{\text{masked}})_{y_i} \right|
$$

其中，$z_i^{\text{full}}$ 是正常视频输入下的 logits，$z_i^{\text{masked}}$ 是遮蔽视频后的 logits。变化越大，说明该 Token 对视觉输入越敏感。

**直观理解**：当遮蔽视频后，"person"、"door"、"blue" 等描述视觉内容的词概率会显著下降，这些就是 Visual-Aware Tokens。

### 2. Temporal-Aware Tokens：时间感知归因

**目标**：识别哪些 Token 依赖于视频的时间顺序结构。

**方法**：帧打乱（Frame Shuffling）——随机打乱帧顺序后测量 logits 变化。

$$
\Delta_{\text{temp}}^i = \left| \log \text{softmax}(z_i^{\text{ordered}})_{y_i} - \log \text{softmax}(z_i^{\text{shuffled}})_{y_i} \right|
$$

**直观理解**：当帧顺序被打乱后，"first"、"then"、"appear" 等描述事件顺序的词概率会下降，这些就是 Temporal-Aware Tokens。

### 3. Entropy-Aware Tokens：熵感知归因

**目标**：捕捉模型的预测不确定性，识别推理关键点。

**方法**：计算每个 Token 的预测熵。

$$
\mathcal{H}(i) = -\sum_w p(z_i = w) \log p(z_i = w)
$$

**直观理解**：高熵 Token 如 "however"、"wait" 标记着话语转折或推理关键点，模型在这些位置的不确定性更高。

### 三种信号的互补性

消融实验揭示了三种归因信号的互补关系：

| 归因信号 | 主要词性 | 典型词汇 | 捕捉维度 |
|----------|----------|----------|----------|
| Visual-Aware | NOUN (24.8%) | person, door, blue | 感知依赖 |
| Temporal-Aware | VERB (21.2%), PRON (11.0%) | first, then, appear | 时间依赖 |
| Entropy-Aware | ADV (8.8%) | however, wait | 推理不确定性 |

关键发现：**单独使用 Temporal-Aware 在 Video-Holmes 上收益最大（+3.3），但组合 V+E+T 在所有基准上最稳定**。这说明三种信号捕捉的是不同维度的推理需求，缺一不可。

## Token 选择与策略更新

### 关键 Token 合并

三种归因信号识别的 Token 集合通过并集合并：

$$
S = S_{\text{vis}} \cup S_{\text{temp}} \cup S_{\text{ent}}
$$

实验表明，选择 Top 20% 的 Token 是最优策略，过高会引入噪声。

### 策略更新

采用二值掩码机制，只对关键 Token 进行梯度更新：

$$
J_{\text{Video-KTR}}(\theta) = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} m_{i,t} \cdot \min\left( r_{i,t} \cdot \hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \cdot \hat{A}_{i,t} \right) \right]
$$

其中 $m_{i,t} \in \{0, 1\}$ 是二值掩码，只在关键 Token 处为 1。

### 计算开销

每个 Response 需要额外 2 次前向传播：
- 1 次视觉遮蔽
- 1 次帧打乱

总体计算开销约为 Vanilla GRPO 的 3 倍，但 Rollout 数量不变（G=8）。

## 实验结果

### 主实验：超越 GPT-4o

Video-KTR 在 5 个基准上全面超越基线方法：

| 方法 | Video-Holmes | VideoMMMU | MMVU | TempCompass | VideoMME |
|------|--------------|-----------|------|-------------|----------|
| 基础模型 | 36.5% | 41.2% | 38.9% | 62.1% | 58.3% |
| GRPO | 39.8% | 44.5% | 41.2% | 65.8% | 60.1% |
| **Video-KTR** | **42.7%** | **46.8%** | **43.1%** | **68.4%** | **62.5%** |
| GPT-4o | 42.0% | - | - | - | - |

**核心突破**：Video-KTR 在 Video-Holmes 上以 42.7% 超越 GPT-4o (42.0%)。

### 消融实验关键发现

1. **硬性选择优于软加权**：Binary Top-20% 策略最优
2. **20% 更新比例最佳**：过高引入噪声
3. **扰动强度鲁棒**：帧打乱/视觉遮蔽的具体强度对结果影响有限

## Case Study：视频问答中的关键 Token 识别

以问题 "What did the person do after entering the room?" 为例：

```
1. Visual-Aware Token: "person"
   → 遮蔽视频后概率下降 0.35
   
2. Temporal-Aware Tokens: "after", "entering"
   → 打乱帧后概率下降 0.28
   
3. Entropy-Aware Token: "did"
   → 高熵 2.7

合并关键 Token 集合: {person, after, entering, did}
仅对这些 Token 进行强化学习更新
```

这个例子清晰地展示了三种归因信号如何协同工作：Visual-Aware 锁定视觉实体，Temporal-Aware 捕捉时间关系，Entropy-Aware 标记推理关键点。

## 与相关工作的对比

### 与熵选择方法的关系

Wang et al. (2025) 发现高熵 Token（"forking tokens"）是 LLM 推理的关键。Video-KTR 在此基础上引入模态感知：

| 维度 | Wang et al. 2025 | Video-KTR |
|------|------------------|-----------|
| 归因信号 | 只有熵 | 熵 + 视觉 + 时间 |
| 适用场景 | 纯文本 LLM | 视频 MLLM |
| 模态感知 | ❌ 无 | ✅ 有 |

消融实验表明，在视频推理中**单纯用熵选择会遗漏重要的时间依赖 Token**（Temporal-only 比 Entropy-only 在 Video-Holmes 上高 2.6 分）。

### 与 SDFT 的对比

SDFT（Self-Distillation from Foundation Model）通过自蒸馏利用环境反馈，Video-KTR 则利用模态归因信号。两者都是 RL 信号增强的不同路径，可以互补。

## 方法论启发

### 1. 反事实分析的有效性

通过"扰动→测量变化"揭示因果依赖，是一种简洁有效的分析方法：
- 视觉遮蔽 → 视觉依赖
- 帧打乱 → 时间依赖
- 概率分布 → 不确定性

### 2. 选择性优化的普遍价值

强化学习中的"选择性"是性能提升的关键——不是所有 Token/样本/梯度都同等重要：

| 工作 | 选择性维度 | 实现方式 |
|------|-----------|----------|
| Video-KTR | Token 选择 | 模态归因信号 |
| RAL | 注意力选择 | 注意力分布 |
| VC-STaR | 样本选择 | 视觉对比过滤 |
| SDPO | 梯度选择 | 自蒸馏 |

### 3. 可解释性价值

Token 级归因揭示了模型"关注什么"，为调试提供了抓手。高 Visual-Aware Token 集中在名词，高 Temporal-Aware Token 集中在动词和代词，这种语言学意义上的对应关系增强了方法的可信度。

## 局限性与未来方向

### 局限性

1. **计算开销**：需要额外的前向传播计算归因分数
2. **超参敏感**：Top-K 比例需要调优
3. **任务特定**：针对视频推理设计，泛化到其他模态需适配

### 未来方向

1. **自适应 K 值**：根据样本难度动态调整选择比例
2. **跨模态归因**：扩展到图像-文本等场景
3. **在线归因**：在推理时实时计算，减少训练开销
4. **与其他 RL 算法结合**：如与 RAL 的注意力优化组合

## 总结

Video-KTR 的核心贡献在于首次提出**模态感知的 Token 级强化学习**框架，通过三种互补的归因信号（视觉、时间、熵）实现精细化的策略塑造。在 Video-Holmes 上超越 GPT-4o 的结果证明了 Token 级选择性更新的有效性。

这项工作为视频推理的强化学习开辟了新方向：从粗粒度的序列级奖励到细粒度的 Token 级归因，从单一信号到多模态感知。核心洞见——**识别并聚焦于关键 Token**——具有普遍的方法论意义，有望启发更多领域的选择性优化策略。

---

**参考文献**：

- Wang, Z., et al. (2026). Video-KTR: Reinforcing Video Reasoning via Key Token Attribution. ICLR 2026.
- Wang, et al. (2025). Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning.