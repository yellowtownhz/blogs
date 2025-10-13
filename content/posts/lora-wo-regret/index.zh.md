---
# weight: 1
title: "LoRA without Regret: PEFT什么时候可以达到全量微调"
date: 2025-10-12T12:00:00+08:00
tags: ["RL", "PEFT"]
categories: ["General"]
draft: false
summary: "Thinking Machines最新博客把LoRA给整明白了"
author: "黄镇"
lang: "zh"
type: "posts"
resources:
- name: "featured-image"
  src: "cover.png"
---

原始链接：https://thinkingmachines.ai/blog/lora/

## 研究的关键问题
在SFT和RL的时候，LoRA和全量微调相比什么时候会 work、什么时候效果差不多、什么时候效果差。这个问题对于post training 的技术选型有很大参考价值。

## LoRA什么时候能达到全量微调的效果？

{{< figure src="fig1.png" alt="核心实验" width="60%" >}}
不同颜色的线代表不同的rank，横坐标是训练步数（对数尺度，纵坐标是测试集的NLL Loss，线上的每一个点代表rank=r，不同学习率在训练步数=n时的最小loss。

{{< admonition note "一句话结论" >}}
sft的时候，高rank的lora和fullft的loss曲线都和训练程度是线性的，并且重合度很高。
{{< /admonition >}}

{{< admonition tip >}}
从图上看rank=64 就基本上和fullft很接近了，实际做post-training的时候可以按照这个来。
注意：[llama factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_lora/llama3_lora_sft.yaml)的默认rank=8，和[peft](https://github.com/huggingface/peft)的默认rank=16, 都比这个推荐的值要小，实战的时候可以试着把默认rank调大。
{{< /admonition >}}

{{< admonition tip >}}
当rank过小时，loss随训练进行下降的速度会变慢。可以理解成当adapter的capacity 小于训练数据的难度时，训练会变慢。
{{< /admonition >}}

## LoRA的最优学习率是多少？
{{< figure src="fig2.png" alt="核心实验" width="60%" >}}

{{< admonition note "一句话结论" >}}
LoRA的最优学习率是全量训练的10倍
{{< /admonition >}}

{{< admonition tip >}}
对于大一点的 rank（＞32）有一个最优学习率（loss 最低），对于小一点的 rank（＜32），还是小一点的学习率更好一点，实战的时候lora 的学习率可以最大不超过全量学习率的十倍，这样比较保险。
{{< /admonition >}}

## RL+LoRA的最优rank是多少？
{{< figure src="fig3.png" alt="核心实验" width="60%" >}}

{{< admonition note "一句话结论" >}}
RL 的时候，rank=1 和全量微调能达到一样效果（学习率大十倍）
{{< /admonition >}}

注意： 这里的 RL 不确定具体的算法细节，是不是 on-policy，不过样子应该是 on policy，因为thinking machines 这个公司就是押宝on policy 的

{{< admonition note "为什么RL+rank=1就可以做到和全量微调一样的效果？" >}}
SFT + rank=1 的效果是远不如全量微调的，为什么 RL 可以达到一样的效果？博客的解释是RL 数据的信息量比SFT 小。一条response 长度等于 n 的 sft 数据的信息量是 O(n)，因为每个 token 都有一个 ground truth token，可以看作每个 token 的信息量都是 1 个单位。在 RL 里面，一条数据的信息量只有一个单位，因为只有答案对或者不对这两种可能。这一点解释不知道数学上是否严谨，但博客从这个角度来解释，因为即使是 rank=1 的 lora，它的参数量也是大于这么计算出来的信息量的，所以 lora 有这个 capacity来学习这个能力
{{< /admonition >}}

{{< admonition note "LoRA的有效学习率先慢后快" >}}
因为 lora 里面有 ab 两个矩阵，a矩阵是随机初始化，b 矩阵是全零初始化。所以在训练的早期 a矩阵的变化不会怎么影响整体的 delta w 也就是a 乘上 b，相当于是有效的学习率会比较小。在训练的后期，b 矩阵的值开始慢慢变大了，a有效学习率又提高上来了。所以，在比较少量的训练样本时建议选择更大的学习率，训练样本较多的时候学习率可以降低一下。实验经验，少量训练样本时 lora 的最佳学习率是全量微调的 15 倍，大量样本时是 10 倍
{{< /admonition >}}