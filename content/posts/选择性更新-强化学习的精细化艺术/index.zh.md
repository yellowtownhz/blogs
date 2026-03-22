---
title: "选择性更新：强化学习的精细化艺术"
date: 2026-03-23T00:00:00+08:00
tags: ["强化学习", "选择性更新", "Token级优化", "Trust Region", "持续学习", "视频推理"]
categories: ["强化学习"]
draft: false
summary: "从 Video-KTR 的 Token 级选择到 SDFT 的分布级 Trust Region，探讨强化学习中'选择性更新'的两种范式及其背后的哲学意义"
author: "黄镇"
lang: "zh"
type: "posts"
---

## 1. 引言：强化学习的"选择性"困境

在强化学习应用于大语言模型的浪潮中，一个核心问题逐渐浮现：**不是所有更新都是同等重要的**。

传统的强化学习范式假设每一步、每一个样本都应该被同等对待。然而，近期的研究揭示了一个更精细的图景：

- **Video-KTR**（ICLR 2026）提出：在视频推理中，只需对 **20% 的关键 Token** 进行更新，就能超越 GPT-4o
- **Self-Distillation Enables Continual Learning**（SDFT, 2026）揭示：通过 **Trust Region 约束**，可以在学习新知识时保持分布稳定，避免灾难性遗忘

这两篇论文共同指向一个核心洞见：**选择性更新是强化学习精细化的关键**。

---

## 2. 核心问题：为什么"选择性"重要？

### 2.1 全量更新的问题

传统的强化学习更新策略面临两难：

| 更新策略 | 问题 |
|----------|------|
| 全量更新 | 噪声信号干扰学习，计算资源浪费 |
| 随机采样 | 错过关键学习机会 |
| 均匀权重 | 忽视样本/Token 的异质性 |

**核心矛盾**：信号有强弱、样本有难易、Token 有轻重。不加区分的更新策略如同"大海捞针"。

### 2.2 选择性更新的哲学

**选择性更新**的核心思想是：识别并聚焦于最关键的更新目标。

这可以从两个维度理解：

1. **Token 级选择性**：在序列中，哪些 Token 最值得学习？
2. **分布级选择性**：在参数空间中，如何保持稳定的更新方向？

Video-KTR 和 SDFT 分别从这两个维度给出了答案。

---

## 3. Token 级选择性：Video-KTR 的视角

### 3.1 问题背景

视频推理的多模态特性带来独特挑战：一个视频包含数百帧、数万 Token，但只有少数 Token 真正依赖视觉输入或时间结构。

传统 GRPO 对整个响应序列计算统一的标量奖励，忽视了这种异质性。

### 3.2 核心方法：三种归因信号

Video-KTR 提出通过**反事实扰动**识别关键 Token：

#### Visual-Aware Tokens

**目标**：识别依赖视觉输入的 Token

**方法**：遮蔽视频后测量 logits 变化

$$\Delta_{\text{vis}}^i = \left| \log \text{softmax}(z_i^{\text{full}})_{y_i} - \log \text{softmax}(z_i^{\text{masked}})_{y_i} \right|$$

**典型 Token**：名词（person, door, blue）

#### Temporal-Aware Tokens

**目标**：识别依赖时间结构的 Token

**方法**：打乱帧顺序后测量 logits 变化

$$\Delta_{\text{temp}}^i = \left| \log \text{softmax}(z_i^{\text{ordered}})_{y_i} - \log \text{softmax}(z_i^{\text{shuffled}})_{y_i} \right|$$

**典型 Token**：动词/代词（first, then, appear）

#### Entropy-Aware Tokens

**目标**：捕捉推理不确定性

**方法**：预测熵

$$\mathcal{H}(i) = -\sum_w p(z_i = w) \log p(z_i = w)$$

**典型 Token**：副词（however, wait）

### 3.3 Token 选择策略

Video-KTR 合并三种信号后，采用 **Binary Top-20%** 策略：

$$S = S_{\text{vis}} \cup S_{\text{temp}} \cup S_{\text{ent}}$$

只对关键 Token 计算策略梯度：

$$J_{\text{Video-KTR}}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} m_{i,t} \cdot \text{PPO-Loss} \right]$$

其中 $m_{i,t} \in \{0, 1\}$ 是二值掩码，仅在关键 Token 处为 1。

### 3.4 关键发现

{{< admonition success "核心结果" >}}
Video-KTR 在 Video-Holmes 基准上达到 **42.7%**，超越 GPT-4o 的 42.0%，同时仅更新 **20% 的 Token**。
{{< /admonition >}}

消融实验揭示：

| 策略 | Video-Holmes | 说明 |
|------|--------------|------|
| GRPO（全量） | 39.8% | 基线 |
| 仅时间感知 | 42.1% | 单独使用收益最大 |
| **V+E+T 组合** | **42.7%** | 所有基准最稳定 |

**语言学分析**证明三种 Token 类型互补：

| Token 类型 | 主要词性 | 捕捉维度 |
|------------|----------|----------|
| Visual-Aware | NOUN (24.8%) | 感知依赖 |
| Temporal-Aware | VERB/PRON (32.2%) | 时间依赖 |
| Entropy-Aware | ADV (8.8%) | 推理不确定性 |

---

## 4. 分布级选择性：SDFT 的视角

### 4.1 问题背景

持续学习面临**灾难性遗忘**：学习新知识时，模型会忘记旧知识。

传统 SFT 方法在演示数据上做 off-policy 训练，剧烈改变模型分布，导致遗忘。SDFT 的核心问题是：

> *当只有演示数据可用时，我们如何获得 on-policy 学习的好处？*

### 4.2 核心方法：Trust Region 约束下的自蒸馏

SDFT 将持续学习形式化为 **Trust Region 优化问题**：

$$\pi_{k+1} = \arg\max_\pi \mathbb{E}_{y\sim\pi}[r(y,x)] - \beta D_{KL}(\pi(\cdot|x) \| \pi_k(\cdot|x))$$

两个目标：
1. **最大化奖励**：学习新知识
2. **保持接近**：不偏离当前策略太远

**核心洞见**：演示条件化的模型 $\pi(y|x,c)$ 可以近似最优策略，同时满足 Trust Region 约束。

### 4.3 Self-Teacher 设计

SDFT 使用同一模型的不同条件化版本：

```
学生模型：π_θ(y|x)           # 只看到问题
教师模型：π(y|x, c)           # 看到问题 + 演示
```

**蒸馏目标**：反向 KL 散度

$$\mathcal{L}(\theta) = D_{KL}(\pi_\theta(\cdot|x) \| \pi(\cdot|x,c))$$

**关键技术**：
- **EMA 教师**：使用指数移动平均更新教师权重，防止训练崩溃
- **On-policy 采样**：在学生自己的分布上训练，减少分布不匹配

### 4.4 Trust Region 的验证

论文通过实验验证教师模型满足 Trust Region 的两个条件：

| 条件 | 验证 |
|------|------|
| **Optimality** | 100% 准确率（教师看到演示后输出正确） |
| **Minimal Deviation** | KL = 0.68 nats（vs SFT 的 1.26 nats） |

这意味着教师模型**既准确又稳定**——正是 Trust Region 约束的理想目标。

### 4.5 关键发现

{{< admonition success "核心结果" >}}
SDFT 在知识获取任务上达到 **89% 严格准确率**（vs SFT 的 80%），同时在连续 3 个任务学习后保持 **100% 归一化性能**（SFT 降至接近 0）。
{{< /admonition >}}

先前能力保持（6 个基准平均）：

| 方法 | 新任务 | 先前能力 |
|------|--------|----------|
| SFT | 80% | 53.4 - 60.2 |
| **SDFT** | **89%** | **64.5 - 65.4** |

---

## 5. 两种选择性的对比与统一

### 5.1 维度对比

| 维度 | Video-KTR | SDFT |
|------|-----------|------|
| **选择性粒度** | Token 级 | 分布级 |
| **选择依据** | 多模态归因信号 | Trust Region 约束 |
| **优化目标** | 稀疏奖励问题 | 灾难性遗忘问题 |
| **计算开销** | +2 次前向传播 | +1 次前向传播 |
| **适用场景** | 多模态推理 | 持续学习 |

### 5.2 统一视角：选择性作为归纳偏置

两种方法本质上都是在强化学习中引入**归纳偏置**：

**Video-KTR 的归纳偏置**：
- 不是所有 Token 都依赖视觉/时间输入
- 多模态推理需要识别模态特异性依赖
- 选择性更新减少噪声信号

**SDFT 的归纳偏置**：
- 学习新知识不应剧烈改变已有分布
- 演示条件化模型近似最优策略
- Trust Region 约束保证稳定学习

### 5.3 互补性

两种选择性可以组合：

1. **Token 级 + 分布级**：在关键 Token 上施加更强的 Trust Region 约束
2. **多模态持续学习**：Video-KTR 的 Token 选择机制可以集成到 SDFT 的框架中

---

## 6. 方法论启发

### 6.1 反事实分析的有效性

Video-KTR 的反事实扰动（遮蔽视频、打乱帧）揭示了一个通用方法论：

> **通过"扰动→测量变化"揭示因果依赖**

这种思路可以推广到其他模态和任务。

### 6.2 EMA 的稳定作用

SDFT 的 EMA 教师更新是防止遗忘的关键：

$$\theta_{\text{teacher}} = \alpha \cdot \theta_{\text{teacher}} + (1-\alpha) \cdot \theta_{\text{student}}$$

消融实验证明：
- 冻结基础模型作为教师：性能差
- 学生作为教师：训练发散
- **EMA 教师**：稳定且有效

### 6.3 组合优于单一

Video-KTR 的消融实验表明：
- 单独使用时间感知 Token 收益最大
- 但组合三种信号在所有基准上最稳定

这提示我们：**多种选择性信号的融合能够捕捉互补维度**。

---

## 7. 局限性与未来方向

### 7.1 Video-KTR 的局限

| 局限 | 说明 |
|------|------|
| 计算开销 | 需要额外的前向传播计算归因分数 |
| 超参敏感 | Top-K 比例需要调优 |
| 任务特定 | 针对视频推理设计，泛化需适配 |

**改进方向**：
- 自适应 K 值：根据样本难度动态调整选择比例
- 在线归因：在推理时实时计算，减少训练开销

### 7.2 SDFT 的局限

| 局限 | 说明 |
|------|------|
| ICL 依赖 | 依赖模型的上下文学习能力 |
| 超参调优 | EMA 系数 α 需要针对任务调优 |
| 演示质量 | 效果依赖于演示数据质量 |

**改进方向**：
- 自适应 EMA：根据训练阶段动态调整 α
- 主动学习：让模型自主选择最有价值的演示数据

### 7.3 统一框架的探索

一个值得探索的方向是：**将 Token 级选择性和分布级选择性统一到一个框架中**。

潜在架构：
1. 使用 Video-KTR 的归因机制识别关键 Token
2. 在关键 Token 上施加更强的 Trust Region 约束
3. 使用 SDFT 的 EMA 机制保证分布稳定性

---

## 8. 总结

Video-KTR 和 SDFT 从不同维度揭示了强化学习的精细化方向：

| 论文 | 核心贡献 | 选择性维度 |
|------|----------|------------|
| **Video-KTR** | 模态感知的 Token 级强化学习 | Token 选择 |
| **SDFT** | Trust Region 约束下的持续学习 | 分布选择 |

**共同洞见**：强化学习中的"选择性"是性能提升的关键——不是所有 Token、不是所有更新方向都同等重要。

**哲学意义**：选择性更新体现了强化学习从"暴力搜索"向"精细优化"的范式转变。正如人类学习聚焦于关键信息，机器学习也需要学会"选择"。

---

## 9. 参考文献

**[1]** Video-KTR: Reinforcing Video Reasoning via Key Token Attribution — Ziyue Wang 等 (ByteDance, NTU, NUS), ICLR 2026

**[2]** Self-Distillation Enables Continual Learning — Idan Shenfeld 等 (MIT, ETH), arXiv 2026

**[3]** Wang et al. 2025b: Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning — 启发 Video-KTR 的熵选择方法

**[4]** Schulman et al. 2017: TRPO — Trust Region Policy Optimization 的理论基础