---
# weight: 1
title: "Concept-Aware Finetuning"
date: 2025-10-19T12:00:00+08:00
tags: ["微调", "Multi-token Prediction"]
categories: ["General"]
draft: false
summary: "垂域的微调怎么解决概念理解的问题"
author: "黄镇"
lang: "zh"
type: "posts"
---

## 关键问题
垂域微调（比如医学/法律/代码）中，类似"ribonucleic acid"会被编码成"rib" → "on" → . . . 等5个token，next-token-prediction loss建模不到整个单词的语义。

## 关键做法
SFT的时候，比如输入第一个token “rib”，的时候在原来的预测下一个token的基础上，再加上4个辅助头来预测接下来第2-5个token。训练结束后丢掉辅助头，没有额外推理开销。（辅助头对应的是原始模型的最后一层）

## 关键细节：

### multi-token fine-tuning为什么这么难？
1. base模型是正常next-token prediction来训练的，multi-token fine-tuning会引入distribution shift，导致微调后效果比base模型还差。
2. 额外的辅助loss一般比正常的next-token prediction loss要高，直接训练会导致模型倾向于更多地针对辅助loss来优化，但是在模型推理的时候这些辅助头要丢弃掉，这种优化的方向对于推理没有提升。
3. post-training比pre-training短的多，没有那么多的compute来体现multi-token fine-tuning的价值，训练就已经结束了

### 训练没有想象的那么简单！
训练分为两个阶段：第一个阶段先冻结模型参数、只训练辅助头，第二个阶段冻结辅助头、训练模型参数。有8个关键细节：
1. 辅助头用的是正常模型的lm_head的参数来初始化，确保有一个不错的起始点；
2. 不同辅助头用的是一个共享的unembedding层，确保没有distribution shift；
3. 第一个阶段训练的时候只训练新增加的辅助头，模型参数包括unembedding都是冻结的。训练的loss函数是：$\mathcal{L}\_n = \sum_{k=2}^{n} -\alpha^{k-2} \log p_{t+k}(y_{t+k})$。loss中有一个衰减系数，负责预测token越远的辅助头对应的loss系数更低，保证训练稳定。
4. 辅助头训练用的数据来自ShareGPT、Tulu3-SFT的通用数据，但是response由original model + original head输出，可以认为是把original head的输出分布蒸馏到辅助头上。
5. 第二个阶段训练冻结辅助头和unembedding，两个阶段中unembedding都是要冻结的，我的理解是因为所有head都共享一个unmbedding矩阵，这个参数一旦更新会影响所有的head以及原始模型，训练会很不稳定，所以干脆从头到尾都冻结好了，反正预训练的时候unembedding已经有了一个不错的参数，也够用了。
6. 第二个阶段训练loss $\mathcal{L}\_n = \sum_{k=1}^{n} -\alpha^{k-1} \beta \gamma \log p_{t+k}(y_{t+k})
$，beta=0.01是一个偏小的值，gamma是一个随着训练逐渐减小的值，刚开始希望模型更多的聚焦在辅助loss上，训练后期收敛的时候希望模型保证正常的next-token loss L1最优。
7. 第一阶段训练辅助头的时候用的数据是通用的sft数据，训练之后的辅助头可能在垂域（医学/代码）场景下效果不太好；在实际场景下，可以单独拿医学/代码的数据再单独微调1个epoch辅助头，效果提升明显。
8. 作者还给了一些训练过程profiling的trick：单独monitor L1和Ln loss（不带系数的），如果训练正常的话，应当看到：a、Ln随着epoch增加而减小，说明模型确实在针对辅助loss在优化；b、Ln应该比正常的fine-tuning的时候的Ln要低，说明multi-token的loss确实是有帮助的；c、一般而言，当L2>4.0的时候，说明辅助头不可靠，这种情况下就需要在实际场景的训练集上把辅助头先微调一个epoch，然后再进行第二个阶段的训练。

### Insights：
{{< admonition tip >}}
目前Qwen3的tokenizer对于小语种的效果不好，经常一个独立的单词会被编码成2-4个token，也适用于这种multi-token建模的范畴。像地学这种有比较复杂的专业名词的领域，也可以尝试。
{{< /admonition >}}

{{< admonition tip >}}
还是有不少的trick：两个阶段训练 + 各种loss系数的设置 + 参数的初始化，这种多token预测的技术细节很多。
{{< /admonition >}}

{{< admonition tip >}}
从压缩即智能的角度，一个单词被tokenize成多个token的信息压缩是不够的，multi-token prediction就是在增加这个压缩。如果是做预训练，压缩的越好，模型智能肯定越好；但是是fine-tuning的话，就有点把不准这个带来的收益有多大了。
{{< /admonition >}}

最后附上Ilya Sutskever 2023年关于unsupervised learning的一个talk，里面最大的收获就是“压缩即智能”.
{{< youtube AKMuA_TVz3A >}}