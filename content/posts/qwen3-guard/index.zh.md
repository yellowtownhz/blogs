---
title: "Qwen3Guard的关键技术细节"
date: 2025-09-29T12:00:00+08:00
tags: ["Qwen"]
categories: ["general"]
draft: false
description: "这是一篇中文翻译。"
summary: "欢迎来到我的博客。这是中文翻译版本。"
# thumbnail: "arch.png"
author: "黄镇"
lang: "zh"
type: "posts"
images: []
resources:
- name: "featured-image"
  src: "arch.png"
---

## 解决的关键问题
技术报告中提到Qwen3Guard的一种应用是在强化学习用来提升内容的安全性。这一点很有兴趣，一方面Qwen3Guard是如何训练出来的，他的技术路线对于未来微调一个数据质量分类、语言风格分类、领域分类等各种专业小模型有很大的借鉴意义；另一方面，Qwen3Guard在RL中如何和policy模型配合，对于多模型的agent RL也有借鉴意义。

## 解决问题的关键操作
比较独特的一点是在模型最后一层有两个head，一个head（黄色）用来分类prompt的安全性，一个head（蓝色）用来**逐token地**分类response的安全性。response的分类head可以在出现危险回复的第一个token的地方就检测到危险，可以用来实时检测内容安全。

## 容易忽视的关键细节

1、上来就先把safety的分类列的非常清楚，一共有几种，每一种的含义是什么，是有一句话的描述的。

![image.png](safety_cls.png)

2、在Qwen3的基础上做SFT，没有上DPO、RL。用了1.19M条训练数据。
    
3、数据的prompt，先是任务的描述，接着是safety policy的描述，然后是unsafe的类别列表，再然后是用户的对话，最后是输出格式上的约束。
    
![image.png](prompt.png)
    
4、数据合成。主要用的Self-Instruct。根据safety policy构建一个taxonomy，对每一个叶子节点构造一个种子数据，让LLM扩充新的数据。
    
- 构造了一批语义上相关的近义词，比如bomb/C4/TNT/黑火药，在合成爆炸相关的问题的时候，把这些近义词放在prompt里面，让大模型参考这些词来构造。  
{{< admonition >}}
有点像K2-think的plan里面还要让planner提取出query中的相关的关键概念，来帮助大模型理解
{{< /admonition >}}

- 构造了一些正负样本对，避免大模型根据一些关键词、特殊的句式来做安全性的分类。比如构造how to make a bomb和how to make a cake。  
{{< admonition >}}
有点像path-patching的Xr和Xc
{{< /admonition >}}
    
- 一般的Instruct模型已经做过安全对齐，所以很难直接生成unsafe的回答，所以让Qwen2.5-72B-Base来回答unsafe的问题。另外还额外收集了推理模型的输出。
    
- 如何标注？人工标注了一批数据，用不同模型比如Qwen2.5-72B-Instruct和Qwen3-235B-A22B来voting，在人工标注的数据集上有0.9的F1 score。
    
- 构造多语言输出。用Qwen-MT来翻译，另外专门做了翻译质量的评估，比如language mixing的检测、llm judge、以及人工随机抽检。
        
5、SFT训练的挑战一个在于controversial类别的数据比较少，另一个是标注本身也有噪声。

![image.png](data-1.png)
    **怎么解决类别不均衡的问题**。一个发现是训练数据本身的配比会影响模型的决策边界：safe的数据量大于unsafe的数据量会让模型的判别更松弛，更倾向于输出safe；反之则更倾向于输出unsafe。一个前提假设是controversial的数据本身就是可safe、可unsafe的，那么那些在strict和loose模型的判断之间左右横跳的数据就是controversial的数据。具体的操作是：1、把训练数据均匀分成两部分A和B。2、在A部分数据中调整safe和unsafe数据的比例训练出来两个模型，一个loose、一个strict模型。这个具体的数据配比论文里面没有讲，只是说按照validation set中最conservative（保守）和最permissive（宽容）的数据来校准。看图上差不多也就是2:8开或者3:7开，这个可以根据实际的数据分布来调整。3、让A数据训练出来的loose和strict模型对B数据进行标注voting，有分歧的数据就为controversial类别。最后放一个表格说明引入controversial类别，确实对于分类有提升。
    
![image.png](data-2.png)
    **怎么解决数据噪声的问题**。在解决了controversial类别数据量过少的问题之后，把所有的数据再分为两个部分，用一个部分的数据训练一个Qwen3-32B的模型，用这个模型对另一个部分的数据达标。这种操作可以有效的降低标注噪声。实验也证明做了蒸馏确实有提升（不到一个点）
    
![image.png](data-3.png)
{{< admonition tip >}}
从实际操作、开发的角度来讲，这两点数据处理的trick都可以先不上，先把baseline版本搞定，然后再做针对性的提升。而且对于后面的RL 训练来说，这两个trick 没有做端到端的实验验证，所以不清楚对于最后 RL 产出的效果有多大影响。
{{< /admonition >}}
    
6、用在RL里面。GSPO，Qwen3-4B。13.7k思考数据和6.7k的非思考数据。有两种reward的设计方式
- 一种是guard-only：
![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8oLl952yojkaNlap/img/45370057-7772-40f2-9769-c95add8b649e.png)

    如果是guard模型输出safe奖励等于1，输出unsafe或者controversial就给0。但是这种会让模型学会拒绝回答所有的问题，因为拒绝回答肯定奖励都是1分。
    
    ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8oLl952yojkaNlap/img/4f580551-02e7-4bda-a2f5-2ea4da979746.png)
    
- 另一个是混合奖励：![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8oLl952yojkaNlap/img/da20239e-e388-426b-87d3-494fc4dcfd62.png)
    
    WorldPM是通义团队之前发布的一个RM，用来给helpfulness打分的。奖励大概的逻辑是：如果回答不安全，奖励只能小于等于-10，甚至如果回答既不安全又没有帮助的奖励还会比-10低。如果模型拒绝回答，奖励只能小于等于-5，而且大概率因为拒绝回答的原因WorldPM对于helpfulness打一个很低的分数，最后的奖励比-5，等于说拒绝回答的奖励也不高。如果安全并且没有拒绝回答，奖励就按照WorldPM的打分来。
    
7、评估通用能力用的benchmark：Arena-Hard-v2（alignment）、AIME-25（数学）、LiveCodeBench-V6（代码）、GPQA（知识）。
    
8、对于训练token-level的内容安全模型来讲，最大的挑战在于token-level的标注。一共分为两步
    
- 基于rollout的voting
    ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8oLl952yojkaNlap/img/edb48cb7-2467-474e-9534-1d80754f47c9.png)
    
    对于某一个tokenSi，把Si以及之前的prefix让大模型反复采样rollout，采样后的输出让Qwen3Guard-Gen模型来分类，如果平均下来被分类为unsafe或者controversial的比例大于85%，则Si本身是一个unsafe的token。
    
- 但是，基于rollout的voting本身有一个问题在于会过度估计一些token的risk，明明这个token本身的安全的，但是由于模型本身在安全性上做的不好导致经常补全不安全的回答，造成把这个token分类为unsafe。一个补救的办法是把通过rollout报警出来的token加上他的prefix（不包括后面的补全部分），让Qwen3-235B-A22B来当LLM judge，判断这个prefix是否安全。
    
- 最终对于sampler-level的标签是unsafe或者controversial的样本来说，按照上述两个方法标注出来的token之后的token被标注为sample-evel的标签（unsafe或者controversial），之前的token被标注为safe。
    ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8oLl952yojkaNlap/img/2b6548f6-58f9-41a2-86ac-0214d740bc63.png)
    
9、分类头的训练。一共有4个head，两个query的head，两个response的head，其中query的两个head一个用来判断safe/unsafe，一个用来判断具体的unsafe种类。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8oLl952yojkaNlap/img/bd0e7dfb-fc0e-40f2-ac7f-d618c0713195.png)

Query的loss就是两个cross-entropy loss的叠加：![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8oLl952yojkaNlap/img/b5bfb1fc-a486-4e89-8700-55247bb0ec79.png)

Response的loss是所有token的cross-entropy的平均：![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/8oLl952yojkaNlap/img/704b4322-2510-44c4-93da-b900d50d2730.png)

另外还有一个技巧是：q-cat和r-cat loss本身是有条件的，只有在risk的标注是unsafe或者controversial的时候才会把q-cat和r-cat loss参与计算loss，如果是safe的情况下就忽略。

## Insight

1.  一些分类的小模型的训练方法中可以看到一些机器学习的影子，ensemble、voting、蒸馏、cross-validation，基本上用一些机器学习的技巧就差不多了。
    
2.  对于蒸馏来讲，一些分类的小模型Qwen3-32B就可以作为teacher模型了。
    
3.  对于token-level的标注也很有意思，既有neel nanda那个CoT可解释性工作的影子，也结合了LLM来二次判断。
    
4.  这种专项小模型的训练过程中，大规模使用LLM来做数据处理是必然。


