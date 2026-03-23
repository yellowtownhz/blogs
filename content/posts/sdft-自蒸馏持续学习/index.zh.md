---
title: "SDFT：自蒸馏让持续学习成为可能"
date: 2026-03-23T00:00:00+08:00
tags: ["自蒸馏", "持续学习", "Trust-Region"]
categories: ["强化学习"]
draft: false
summary: "大模型如何在不遗忘旧知识的情况下持续学习新知识？MIT 和 ETH 团队提出的 SDFT 方法，通过自蒸馏实现了无需奖励信号的 on-policy 持续学习。"
author: "黄镇"
lang: "zh"
type: "posts"
math: true
---

## 1. 问题：大模型的"遗忘困境"

大模型部署后是静态的。GPT-4 不知道昨天发生的新闻，LLaMA 不了解上个月发布的 API。如果要在新知识上微调，模型会"灾难性遗忘"——学了新的，忘了旧的。

传统解决方案各有缺陷：

| 方法 | 问题 |
|------|------|
| **On-policy RL** | 需要奖励信号，但演示数据没有奖励 |
| **SFT 监督微调** | Off-policy，会剧烈改变模型分布 → 遗忘 |
| **Replay Buffer** | 需要存储旧数据，隐私/存储问题 |
| **参数隔离** | 模型规模随任务数增长 |

**核心问题**：当只有演示数据可用时，如何获得 on-policy 学习的好处，从而避免灾难性遗忘？

MIT 和 ETH Zurich 团队在论文《Self-Distillation Enables Continual Learning》中提出了 **SDFT** 方法，给出了一种优雅的解决方案。

---

## 2. 核心思想：用 ICL 实现自蒸馏

### 2.1 同一个模型，两个角色

SDFT 的核心洞见是：**同一个模型，通过不同的条件化方式，可以同时扮演教师和学生**。

```
学生模型：π_θ(y | x)           # 只看到问题
教师模型：π(y | x, c)           # 看到问题 + 演示
```

其中 `c` 是演示（demonstration），比如：
- 问题："2025年缅甸发生了什么灾难？"
- 演示："2025年，缅甸发生了大地震..."

教师通过 **ICL（In-Context Learning）** 能力，"理解"演示内容，生成一个更准确的分布。学生学习这个分布，从而获得新知识。

### 2.2 为什么有效？ICL 不是 SFT

关键区别在于：**ICL 生成的是"软"概率分布，而不是硬标签**。

**SFT 的问题**：
```
训练数据：(问题, 正确答案)
目标：让模型精确复制正确答案的每个 token
结果：模型分布被剧烈改变 → 遗忘
```

**ICL 的优势**：
```
教师输出：在语义等价的各种表达上都有概率
学生目标：让自己的分布**接近**教师的分布
结果：保持原有分布结构，只是"往正确方向偏移"
```

ICL 充当了**隐式奖励函数**：告诉学生"往这个方向走"，而不是"必须精确复制这个答案"。

---

## 3. 数学原理：Trust Region 约束下的最优策略

### 3.1 Trust Region 回顾

标准的 Trust Region Policy Optimization (TRPO) 公式：

$$\pi_{k+1} = \arg\max_\pi \mathbb{E}_{y \sim \pi}[r(y, x)] - \beta D_{KL}(\pi(\cdot|x) \| \pi_k(\cdot|x))$$

两个目标：
1. **最大化奖励** — 让模型变得更好
2. **保持接近** — 不要偏离当前策略太远

这个公式天然平衡了"学新"和"不忘旧"。

### 3.2 SDFT 的关键假设

论文证明了一个重要结论：**演示条件化的策略 π(y|x,c) 可以近似最优策略**。

论文通过实验验证了两个关键条件：

| 条件 | 验证结果 |
|------|----------|
| **Optimality** | 教师在看到演示后，准确率达到 100% |
| **Minimal Deviation** | 教师与基础模型的 KL 散度仅 0.68 nats（SFT 是 1.26 nats） |

这意味着：**教师恰好满足 Trust Region 的两个要求**——既是最优的，又没有偏离太远。

### 3.3 目标函数推导

从 Trust Region 可以推导出**隐式奖励函数**：

$$r(y, x, c) = \log \pi(y|x,c) - \log \pi_k(y|x)$$

SDFT 的目标函数（反向 KL 散度）：

$$\mathcal{L}(\theta) = D_{KL}(\pi_\theta(\cdot|x) \| \pi(\cdot|x,c)) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi(y|x,c)}\right]$$

**为什么是反向 KL？**

| KL 方向 | 特性 | 适用场景 |
|---------|------|----------|
| 正向 KL D(p_T \|\| p_S) | 教师覆盖学生 | 推理优化、追求确定性 |
| **反向 KL D(p_S \|\| p_T)** | 学生覆盖教师 | **持续学习、保持多样性** |

反向 KL 鼓励学生探索教师的多个模式，这对于持续学习至关重要——我们不能让模型"坍缩"到单一回答风格。

---

## 4. 关键技术：EMA 教师机制

### 4.1 为什么需要 EMA？

如果直接用学生作为教师（self-teacher），训练会崩溃：学生和教师同时变化，没有稳定的"锚点"。

论文的消融实验验证了这一点：

| 教师类型 | 效果 |
|----------|------|
| **EMA 教师** | ✅ 最佳，训练稳定 |
| 冻结基础模型 | ❌ 性能差 |
| 学生 = 教师 | ❌ 训练发散 |

### 4.2 EMA 更新公式

$$\theta_{\text{teacher}} = \alpha \cdot \theta_{\text{teacher}} + (1-\alpha) \cdot \theta_{\text{student}}$$

其中 α ∈ {0.01, 0.02, 0.05}（典型值）。

**关键点：教师不计算梯度！** 教师只用于计算 log probability，不参与反向传播。

```python
# 梯度只对学生
∇_θ L = E_{y~π_θ}[log(π_θ(y_t) / π(y_t; teacher)) × ∇_θ log π_θ(y_t)]
                                          ↑
                                      stop gradient
```

### 4.3 EMA 的作用：软锚点

EMA 教师相当于一个**缓慢移动的锚点**：
- 更新足够慢，提供稳定的参考
- 更新足够及时，能够反映学生的进步
- 无需存储旧模型，内存友好

---

## 5. 完整流程：一个知识获取的例子

### 场景：学习 2025 年缅甸地震知识

**输入**：
- 问题 x："What disaster happened in 2025 in Myanmar?"
- 演示 c："In 2025, a major earthquake struck Myanmar..."

### Step 1: 教师模型处理

教师看到完整的 Prompt：

```
<Question>What disaster happened in 2025 in Myanmar?
This is an example for a response to the question:
<Demonstration>In 2025, a major earthquake struck Myanmar, causing widespread damage...
Now answer with a response of your own, including the thinking process:
```

教师通过 ICL 生成分布 π(y|x,c)，"知道"了 2025 年缅甸地震的知识。

### Step 2: 学生模型 On-Policy 采样

学生只看到问题 x（无演示），从分布采样：

```
y ~ π_θ(·|x)
```

生成回答："In 2025, there was a significant earthquake event in Myanmar..."

### Step 3: Token 级别梯度计算

对于生成序列 y 中的每个 token t：

$$\nabla_\theta L = \sum_t \sum_{y_t \in \mathcal{V}} \log\frac{\pi_\theta(y_t|y_{<t}, x)}{\pi(y_t|y_{<t}, x, c)} \nabla_\theta \log \pi_\theta(y_t|y_{<t}, x)$$

### Step 4: 参数更新

- 更新学生参数 θ，使其分布接近教师分布
- 用 EMA 更新教师参数

### 最终效果

| 指标 | SFT | SDFT |
|------|-----|------|
| 新知识准确率（严格匹配） | 80% | **89%** |
| OOD 泛化 | 80% | **98%** |
| 先前能力保持 | 53.4-60.2 | **64.5-65.4** |

---

## 6. 实验结果：SDFT 全面超越 SFT

### 6.1 知识获取能力

论文在多个任务上测试了 SDFT：

| 任务 | SFT | SDFT | 提升 |
|------|-----|------|------|
| Wikipedia 知识（严格匹配） | 80% | **89%** | +9% |
| OOD 泛化问题 | 80% | **98%** | +18% |
| Science Q&A | 58% | **71%** | +13% |
| Tool Use | 62% | **75%** | +13% |

### 6.2 先前能力保持

这是持续学习最关键的指标：

| Benchmark | Base | SFT | SDFT |
|-----------|------|-----|------|
| HellaSwag | 65.5 | 53.4 | **64.5** |
| TruthfulQA | 65.5 | 56.0 | **65.4** |
| MMLU | 65.5 | 60.2 | **65.4** |

**结论**：SDFT 在学习新知识的同时，几乎完全保持了先前能力。SFT 则导致显著遗忘。

### 6.3 连续任务学习

论文还测试了在 3 个连续任务上学习后的性能：

- **SDFT**：所有任务保持约 100% 归一化性能
- **SFT**：先前任务性能降至接近 0

### 6.4 推理能力消融

使用 Olmo-3-7B-Think 模型测试推理能力：

| 模型 | 准确率 | Tokens |
|------|--------|--------|
| Base | 31.2% | 4612 |
| SFT | 23.5% | 3273 |
| **SDFT** | **43.7%** | 4180 |

**惊人发现**：SFT 甚至比 base 模型更差！SDFT 不仅恢复，还大幅超越 base 性能（+40%）。

---

## 7. 消融实验：关键设计决策

### 7.1 KL 估计器对比

| 估计器 | 效果 |
|--------|------|
| **解析 token 级别估计器** | ✅ 最佳性能和稳定性 |
| Token-level 估计器 | 有偏差但可用 |
| Rao-Blackwellized 估计器 | 收益不值成本 |

### 7.2 教师上下文要求

| 上下文 | 准确率 |
|--------|--------|
| **完整 text + answer** | **89%** |
| Text-only（仅问题） | 75% |
| Answer-only（仅答案） | 中等 |

**关键发现**：教师必须同时看到问题和答案才能有效指导学生。

### 7.3 Prompt 设计要点

论文强调，Prompt 必须：
1. 包含完整演示（问题 + 答案）
2. 明确指示"用自己的回答来回应"以触发 ICL 能力
3. 要求包含思考过程，促进推理能力

---

## 8. 与其他方法的对比

### 8.1 SDFT vs SFT

| 维度 | SFT | SDFT |
|------|-----|------|
| 学习方式 | Off-policy | **On-policy** |
| 目标 | 复制硬标签 | **逼近软分布** |
| 遗忘程度 | 严重 | **轻微** |
| 新知识学习 | 快但不稳定 | **快且稳定** |

### 8.2 SDFT vs 传统知识蒸馏

| 维度 | 传统蒸馏 | SDFT |
|------|----------|------|
| 教师来源 | 外部更强模型 | **自己（不同条件化）** |
| 需要预训练教师 | 是 | **否** |
| 教师更新 | 无 | **EMA** |

### 8.3 SDFT vs Video-KTR

Video-KTR 是另一篇关于持续学习的工作，采用 **Token 级选择性更新**策略：
- SDFT：通过自蒸馏实现全局的 on-policy 学习
- Video-KTR：识别关键 Token，只更新对当前任务重要的参数

两种方法可以互补：SDFT 解决"怎么学"，Video-KTR 解决"学哪些参数"。（关于 Video-KTR 的详细介绍，请期待另一篇博客。）

---

## 9. 局限与未来方向

### 9.1 当前局限

1. **ICL 依赖**：方法效果依赖模型的 ICL 能力，对小模型可能效果有限
2. **超参数调优**：EMA 系数 α 需要针对不同任务调优
3. **计算成本**：需要两次前向传播（学生采样 + 教师评分）
4. **演示质量依赖**：效果依赖于演示数据的质量

### 9.2 有趣的开放问题

1. **自适应 EMA**：能否设计动态调整 α 的策略，根据训练阶段自动调整？
2. **演示选择**：如何选择对持续学习最有价值的演示？能否引入主动学习？
3. **多模态扩展**：SDFT 能否应用于视觉-语言模型的持续学习？
4. **理论分析**：SDFT 的收敛性和泛化界是什么？

---

## 10. 总结

SDFT 提出了一个优雅的解决方案，回答了一个核心问题：

> **当只有演示数据可用时，如何获得 on-policy 学习的好处？**

答案是：**利用 ICL 能力，让模型自己教自己**。

核心贡献：
1. **证明**：演示条件化的策略在 Trust Region 约束下可以近似最优策略
2. **方法**：EMA 教师 + 反向 KL + On-policy 采样
3. **效果**：新知识学习接近 Oracle RAG，先前能力保持接近 Base 模型

这个工作代表了 LLM 训练范式的一个趋势：从"依赖外部教师"到"自我增强"。模型足够强大时，最懂它的老师，可能就是它自己。

---

## 参考文献

**[1]** Shenfeld, I., Damani, M., Hübotter, J., & Agrawal, P. (2026). *Self-Distillation Enables Continual Learning*. arXiv:2601.19897.

**[2]** Zhao, S., et al. (2026). *Self-Distilled Reasoner: On-Policy Self-Distillation for Mathematical Reasoning*. arXiv:2601.18734.

**[3]** Hübotter, J., et al. (2026). *Reinforcement Learning via Self-Distillation*. arXiv:2601.20802.