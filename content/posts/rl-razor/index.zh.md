---
title: "RL's Razor: 为什么在强化学习遗忘更少"
date: 2025-10-06T12:00:00+08:00
tags: ["强化学习", "遗忘"]
categories: ["强化学习"]
draft: false
description: "通过solid的evidence验证RL比SFT遗忘更少的原因是on-policy训练，on-policy训练自身带有一种隐式的低KL的约束，使得最终RL后模型输出分布和RL前的KL距离更小。"
summary: "为什么在强化学习遗忘更少：on-policy"
author: "黄镇"
lang: "zh"
type: "posts"
# featuredImage: "kl-divergence-comparison.png"
resources:
- name: "featured-image"
  src: "kl-divergence-comparison.png"

---


## 基本信息
机构：MIT

论文链接：[https://arxiv.org/pdf/2509.04259](https://arxiv.org/pdf/2509.04259)

会议：投稿ICLR 2026

## 关键问题
通过solid的evidence验证RL比SFT遗忘更少的原因是on-policy训练，on-policy训练自身带有一种隐式的低KL的约束，使得最终RL后模型输出分布和RL前的KL距离更小。


## 关键细节

### 1. 为什么要研究new data adaption

作者首先解释了为什么要研究new data adaption的问题：self-evolve，这一点和阿里云CEO吴泳铭在[云栖大会](http://www.news.cn/tech/20250924/c0c0f36318774c84afa44e06b2868639/c.html)上的判断是一致的。

> Despite their remarkable capabilities, today's models are largely static once deployed: they excel at tasks learned during pre-training or post-training, but are not designed to self-improve and continually acquire new capabilities. We imagine a future where deployed models are long-lived agents assisting humans in the long-term and continuously adapting to new needs. As such, models must improve and adapt to new data, environments, and objectives.

大致意思是基础模型在pre-training和post-training训练完成后就可以看作是一个静态的模型了，无法根据实际使用场景持续进化、持续获得新的能力。在将来每一个模型都是一个自我进化的agent，可以不断的根据最新的数据、最新的环境和最新的目标来进化。这一点和之前CV时代的**自适应学习**有相通之处，相同的地方都是在线上进行不断进化，不需要收集数据+微调模型+重新部署；不同之处在于自适应学习的概念只局限于数据本身的适应，这里的self-evolve并不局限于数据，而是包含数据、环境和目标的"适应"。

### 2. introduction单独一段强调研究问题

直接在introduction中用单独的一段来提出问题：RL比SFT遗忘更少的原因是什么？这一点值得在论文写作中借鉴，先发问可以让整个论文更加聚焦到要解决的关键问题上。

> This striking empirical gap raises the question: what underlying mechanism allows RL to improve on new tasks, but unlike SFT, minimally impacts the model's prior knowledge?

### 3. 为什么要研究遗忘的“机制”

还有一段话讨论了为什么已经有不少的方法在解决遗忘问题的情况下，还是要研究。因为遗忘的机制以及不同的训练方法为什么遗忘程度不同，都是未知的。

> Previous approaches to catastrophic forgetting targeted specific factors such as constraining weight updates (Kirkpatrick et al., 2017; Aljundi et al., 2018; Zenke et al., 2017), preserving learned features (Rannen et al., 2017; Hou et al., 2019), or regularizing shift in output distribution (Li & Hoiem, 2017; Stiennon et al., 2020). While these methods can reduce forgetting, they focus on its effects rather than its underlying cause. Consequently, it remains unclear what truly governs forgetting or why different training algorithms behave so differently.

这句话的思路和表述方式同样适合可解释性。

### 4. 三个核心结论

论文主体部分的标题分别叫：Reinforcement Learning Forgets Less than SFT、Smaller KL divergences lead to less forgetting、On-policy methods leads to smaller KL divergence，每一个部分都是一个关键的结论，并且前后之间逻辑相互承接、环环相扣，是非常标准、非常经典、效果非常好的 writing style。并且每一个章节最后都一个单独的take away card，再次强调本章节最后的结论。三点take away列在下面，基本上把这三点take away理解，整个论文的核心结论就比较清晰了。

*   RL is able to learn new tasks while incurring minimal forgetting, whereas SFT reaches similar new-task performance only by sacrificing prior knowledge.
    
*   Catastrophic forgetting in both SFT and RL is predicted by the KL divergence between the fine-tuned and base models on the new task.
    
*   On-policy training explains why RL maintains smaller KL divergence than SFT. Sampling from the model's own distribution keeps it close to the base model, while SFT pushes it toward arbitrary external distributions.

### 5. KL散度与遗忘的关系

验证了KL是用来度量模型遗忘程度的最好metric。注意这里的KL散度计算是在新任务/新数据上，计算的是base 模型和SFT/RL训练后模型输出的分布。这两个分布之间的KL散度越大，说明微调后的模型和base模型在新任务上的分布差异越大，这个差异和训练带来的遗忘有很强的相关性（R^2拟合系数在0.96）。

{{< admonition note "个人思考" >}}
这里逻辑上其实是有一点不自洽，或者说超出我的预期的地方。这里在新的任务上计算KL，背后潜在的假设是在新任务上的差异越大，老任务的遗忘就越大。这一点其实并不是绝对的，说不定模型本身的capacity足够大，可以做到新任务差异大的情况下、老任务的差异小。一个细节是论文讨论的SFT和RL的训练数据都只包含新任务的数据，并没有用replay，所以在replay的设置下这个论文的结论会发生什么变化还不清楚。
{{< /admonition >}}

一个比较符合直觉的做法是计算base模型和微调后模型在原始数据上的分布差异。论文对于这个地方的解释是对于基模训练的原始数据在下游是不清楚的，所以很难在基础数据上进行实验。不过论文自己本身也在ParaityMNIST上做有实验，所以至少在toy setting下这个实验其实是可以做的，但是不知道为什么这个结果没有。

### 6. Parity MNIST实验

Parity MNIST的定义是输入一个MNIST的图片，输出是图片中对应数字的奇偶。比如如果输入是一个带有字母0的图片，那么输出0/2/4/6/8都是预测正确。先在ParityMNIST和FashionMNIST训练一个3层的MLP，然后在ParityMNIST的另外一个子集上SFT和RL。

{{< admonition note "个人思考" >}}
这里对于Parity MNIST的设置其实很有细节。不知道作者这里的设置的reference来自哪里，反正如果我来设计的话，对于ParityMNIST奇偶性的设计我只能想到直接让模型输出0/1。这里的让模型输出0/2/4/6/8的好处有亮点：1）本来模型在"预训练"的时候的训练目标就是预测0-9，在"微调"的时候继续输出0/9可以保证输出的分布没有发生剧烈的变化；2）就像论文里面说的，可以体现"many distinct policies can achieve the same performance."，同样是一个带有0的输入图片，模型无论输出0、2、4、6、8，答案都是对的，这一点和推理模型路径不一定一致、但是答案是唯一的特点很像。
{{< /admonition >}}

### 7. 最优SFT分布（oracle SFT）

为了验证 KL确实可以用来度量遗忘的程度，设置了一个理想实验：对于训练数据中的每一个样本，在保证模型预测正确的情况下选择KL距离的label。通过这种方式构建的训练集SFT训练得到的模型确实遗忘更少。

**个人思考**：在LLM SFT的场景下，一个query有多个正确的candidate response，计算base model在每个response的每个token上的平均KL距离，选择一个KL最小的response作为SFT最终使用的response，这种训练是不是可以让SFT的遗忘更少。注意，这里的KL和PPL在数学上是不等价的，PPL只看GT token id的概率，KL看的是整个此表空间的概率分布。

### 8. On-policy方法导致更小的KL散度

RL和SFT相比有两个不一样的地方，一个是负样本，一个是on-policy训练。SFT只有正样本，并且数据中的response是事先标注好的，和base模型无关。而RL训练一个问题要采样多次，其中既有正样本、也有负样本，而且每个step采样都是在梯度更新之后的最新版模型上采样，response都是模型自己的回答。做了一个2 X 2的ablation实验，最终验证了关键点是在on-policy。论文对此的解释是：

> At each step, the policy samples outputs it already finds likely, then re-weights those samples according to reward, shifting probability mass toward higher-reward outcomes while suppressing lower-reward ones.

大致的意思是：RL的每个response都是模型自己的回答，这个回答发生的概率可大可小，但是至少是在模型可能回答的范围之内。而RL所做的事情就是reranking，把回答正确但是概率"相对较小"的回答往前排，把回答错误但是概率"相对较大"的回答往后排。

{{< admonition note "个人思考" >}}
第一，这个"排序"本身就能够体现RLHF和RLVR的区别和联系。两个虽然都是在做排序，但是RLHF排序的依据是人类的偏好，A比B好是人类来标注出来的；而RLVR排序的依据是正确与否，A比B好是按照答案来的。

第二，SFT严格来说也是在排序，因为输出token的概率要在整个词表空间做归一化，提高ground truth token-id的概率也是在降低其他token-id的概率。但是SFT和RL的区别在于，RL的排序是在做one-above-another的排序，只需要一个token-id的排序比另一个高就可以了；而SFT是在做一个one-above-all的排序，需要一个token-id的排序比其他所有token更高。虽然训练目标在数学上可能是等价的，但是从优化的角度来讲，两个训练目标的优化难度和优化路径是不同的，比如这篇论文里面说的RL和SFT优化目标都是新任务更好的一个区域，但是RL更偏好遗忘更少、 KL差异更好的路径。
{{< /admonition >}}

### 9. 其他metric：参数变化

首先是weight-level changes，参数差大并不代表遗忘多，参数差小并不代表遗忘少。这一点情理之外、意料之中，因为可解释性的基本假设也是某一个任务/能力的关键模块是稀疏的、而不是分布式的，那么关键模块上一点很小的改动就有可能大幅影响模型在对应任务上的表现，反过来不关键/冗余模块上较大的改动也有可能基本不影响对应任务。当然这一点结论或者假设是和任务相关的，可能一个模块在任务A上是可有可无、甚至是可以随便更新的的，但是在另一个任务B确实至关重要、一点都不能变动的。

### 10. 其他metric：表征分布变化

通过实验验证了SFT使得模型在表征空间上发生了比较大的变化，RL却没有。如下图，随着更新步数的增加，SFT前后表征分布的相似度越来越低（0.56），而RL的相似度稳定在0.94附近。具体的计算方式是从wikipedia中随机找一段文本作为probe data，只要确保这一段文本不要出现在微调的数据中就可以。然后计算base模型和微调模型的表征（embedding），具体这个表征用的是embedding还是某一层的hidden states，论文中没有明确说明，按照措辞以及一般的理解，用的应该是input embedding。计算相似度的方法是CKA （Centered Kernel Analysis），CKA一般用来衡量两个分布之间相互正交的程度，正交程度越高说明两个分布之间的相似度越低，具体的CKA方法用的是CKNNA（Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation hypothesis）。

![表征漂移](representation-drift.png)

{{< admonition note "个人思考" >}}
看上去表征空间上的drift差别挺大的，不太清楚为什么没有选择representation来作为度量，可能是因为相关性拟合的效果不太好？但至少说明表征上的drift也可以作为一种观测的方式。
{{< /admonition >}}

### 11. 其他metric：更新的稀疏性和秩

之前有论文（Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng. Reinforcement learning finetunes small subnetworks in large language models）发现RL更新的参数是稀疏的，SFT更新的参数是密集的。实验发现原因是bf16训练导致的，RL更新的一些参数复制比较小，小于bf16的精度导致向下取整为0，所以参数更新才是稀疏的。

{{< admonition note "个人思考" >}}
反过来讲，RL的梯度确实是稀疏的，虽然不是0-1的那种稀疏，但是不同参数更新的幅度是有数量级的差距的。虽然RL因为bf16精度的问题造成了参数更新是稀疏的，但是这种情况下照样RL完成了优化。RL本身并不没有带来数学上严格的参数更新稀疏的性质，从另一个角度来说，RL是不是对于参数更新的稀疏性更友好？在RL训练的过程中显式地约束这种稀疏性，和RL本身是compatible的。
{{< /admonition >}}