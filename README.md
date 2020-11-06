
以Part 1中介绍的论文为核心，分析该论文以及引用该论文的论文，甚至引用引用该论文的论文。
# Part 1 中心论文
>2019 ACL：Multi-hop Reading Comprehension through Question Decomposition and Rescoring  
https://arxiv.org/abs/1906.02916


## 1.1 动机
把原问题分解为多个子问题，把多跳变为多个一跳。

## 1.2 问题分类
- Bridging：第一个子问题的答案在第二个子问题中扮演了主语或宾语的成分，要想回答第二个问题，首先要知道第一个问题的答案。
  - 例如：LOL S5 全球总决赛的MVP 效力于哪支队伍？
  - 子问题：分为两个子问题：“LOL S5 FMVP是谁？”、“[answer] 效力于哪支队伍？”
  - 分析：首先我们要得到第一个问题的答案“Marin”，然后带入到第二个问题中，“Marin 效力于哪支队伍？”，进而得到最终答案“SKT”。
- Intersection：原始问题的答案需要通过两个子问题的答案来取交集才能确定。
  - 例如：《爱情公寓》的哪位主演做客了LOL S10的嘉宾席？
  - 子问题：“《爱情公寓》的主演有谁？”、“谁做客了LOL S10嘉宾席？”
  - 分析：首先我们得到第一个问题的答案{娄艺潇、陈赫、孙艺洲、李金铭、王传君、邓家佳、金世佳、赵霁、李佳航、成果、万籽麟、张一铎……}、接着得到第二个问题的答案{赵品霖、李佳航、罗云熙、张彬彬……}，取得两个集合的交集得出最终答案“李佳航”。
- Comparison：原始问题就是对比两个实体在某方面的属性
  - 例如：川普和拜登谁出生的更早？
  - 子问题：“川普在何时出生？”、“拜登在何时出生？”
  - 分析：得到子问题答案后进行比较产生原始问题答案。

## 1.3 模型
### 1.3.1 分解问题
基于规则式的分解，在深度学习方便应用的比较少，就一层全连接。训练数据也比较少，400条就号称达到了不错的效果。感觉就是为了hotpotQA数据集而设计的这么一个玩意。而且它把问题分为了Bridging、Intersection以及Comparision只包含了大概92%的hotpotQA问题，剩下的8%无从考察，不能通过它给定的方法来做。  
注意：对于每一个原始问题，都会根据四类问题类型（Bridging、Intersection、Comparision以及origin）进行四次分解。

### 1.3.2 单步阅读理解问题
对于每一类分解。从所有文档中检索相关的文档，然后对于每个相关的文档d，给单步阅读理解模型输入d以及question，产生答案（span、yes、no三种）和答案的置信度。最后把答案置信度最高的答案输出作为该类分解的最终答案。  
备注：由于是化简成了单步阅读理解，所以是对在单文档上进行操作的，也就用了SQuAD数据集与hotpotQA数据集的easy部分训练了这一部分的参数。（hotpotQA-easy也是单步的）

### 1.3.3 分解评分
由于共有四种分解方式，因此会产生四个候选答案，该部分则用于判断哪个候选答案会成为最终的答案。  
- 将问题、推理类型、答案以及支撑句concat之后通过BERT得到表示，再最大池化层再sigmoid得到每一类分解方式的概率，最终取概率最大的分解方式所对应的答案为最终答案。
- 一开始就根据问题来做分类，然后仅为一开始判断出的分解类型进行后续操作。

实验结果表明第一种方式要好一点。

## 1.4 实验结果
hotpotQA(full wiki)：验证集 F1 43.26；测试集 F1 40.65  
CogQA（2019 ACL）对应的值为：49.4与48.9  
说明这个方法离CogQA还是有距离啊。

## 1.5 思考一下
- 抽取子问题这里可以改进么？有8%的问题不属于Bridging、Intersection、Comparision，那他们大概是什么情况？
- 这个方法对比与CogQA为什么会有比较大的距离？

## 1.6 被引列表
| 序号 | 论文 | 发表会议 | 是否有用 | 是否开源 |引用程度|
| :---: | :---: | :---: | :---: | :---: | :---: |
|1|[Generating Multi-hop Reasoning Questions to Improve Machine Reading Comprehension](https://dl.acm.org/doi/pdf/10.1145/3366423.3380114)|WWW 2020|0|0|弱引，只引用了观点：三类多跳问题占比92%|
|2|[Do Multi-Hop Question Answering Systems Know How to Answer the Single-Hop Sub-Questions?](https://arxiv.org/abs/2002.09919)|arXiv 2020|1（探索本质当然有用）|0|中引，作为对比模型来说明自己的研究问题：多跳QA系统能否回答单跳子问题；除此之外引用了DecompRC中划分子问题的方法|
|3|[Generating Followup Questions for Interpretable Multi-hop Question Answering](https://arxiv.org/abs/2002.12344)|arXiv 2020|0|0|没细看，没多大价值|
|4|[Multi-hop Question Answering via Reasoning Chains](https://arxiv.org/abs/1910.02610)|arXiv 2019||||
|5|[Complex Factoid Question Answering with a Free-Text Knowledge Graph](https://dl.acm.org/doi/10.1145/3366423.3380197)|WWW 2020||||
|6|[Unsupervised Question Decomposition for Question Answering](https://arxiv.org/abs/2002.09758)|EMNLP 2020||[repo](https://github.com/facebookresearch/UnsupervisedDecomposition)||
|7|[Answering Complex Open-domain Questions Through Iterative Query Generation](https://arxiv.org/abs/1910.07000)|EMNLP 2019||||
|8|[Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://arxiv.org/abs/1911.10470)|ICLR 2020||||
|9|[Self-Assembling Modular Networks for Interpretable Multi-Hop Reasoning](https://arxiv.org/abs/1909.05803)|EMNLP 2019||||
|10|[Asking Complex Questions with Multi-hop Answer-focused Reasoning](https://arxiv.org/abs/2009.07402)|arXiv 2020||||
|11|[Low-Resource Generation of Multi-hop Reasoning Questions](https://www.aclweb.org/anthology/2020.acl-main.601/)|ACL 2020||||
|12|[Logic-Guided Data Augmentation and Regularization for Consistent Question Answering](https://arxiv.org/abs/2004.10157)|ACL 2020||||
|13|[Transformer-XH: Multi-hop question answering with eXtra Hop attention](https://openreview.net/forum?id=r1eIiCNYwS)|ICLR 2020||||
|14|[Robust Question Answering Through Sub-part Alignment](https://arxiv.org/abs/2004.14648)|arXiv 2020||||
|15|[Learning to Order Sub-questions for Complex Question Answering](https://arxiv.org/abs/1911.04065)|arXiv 2019||||
|16|[Text Modular Networks: Learning to Decompose Tasks in the Language of Existing Models](https://arxiv.org/abs/2009.00751)|arXiv 2020||||
|17|[A Simple Yet Strong Pipeline for HotpotQA](https://arxiv.org/abs/2004.06753)|arXiv 2020||||
|18|[Hierarchical Graph Network for Multi-hop Question Answering](https://arxiv.org/abs/1911.03631)|arXiv 2019||||
|19|[Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval](https://arxiv.org/abs/2009.12756)|arXiv 2020||||
|20|[DDRQA: Dynamic Document Reranking for Open-domain Multi-hop Question Answering](https://arxiv.org/abs/2009.07465)|arXiv 2020||||
|21|[Multi-Step Inference for Reasoning Over Paragraphs](https://arxiv.org/abs/2004.02995)|arXiv 2020||||







# Part 2 被引论文阅读笔记
## 1. WWW 2020：Generating Multi-hop Reasoning Questions to Improve Machine Reading Comprehension
### 1.1 动机
觉得hotpotQA训练数据集比较少，所以能不能我们先产生一部分伪数据也来参与模型的训练，以此来缓解数据不充分的情况。本篇论文主要研究的就是如何生成一个需要多跳推理的问题。
### 1.2 实验
实验也比较有意思，分别测试了只使用不同比例（10%~100%）的hotpotQA训练集，用自己的方法扩充伪数据来增加训练样本依次提升性能。

## 2. arXiv 2020：Do Multi-Hop Question Answering Systems Know How to Answer the Single-Hop Sub-Questions?
### 2.1 动机
非常不错的一个工作，其研究点在于，已有的一些用于解决多跳问题的模型能否解决单跳问题么？作者通过自动+人工的方式产生了大量高质量的子问题，让能够答对复合问题的模型，去回答这些拆分复合问题之后的子问题，发现即使是SOTA复合模型在子问题上的正确率也非常低，甚至不到50%。对于一个人来说，如果他能回答对复合问题，那他肯定能回答正确该复合问题分解之后产生的子问题，而模型却答不对，因此我们可以推断出，模型很有可能就没有真正理解原文，而是通过一些清奇的线索来回答问题。所以路漫漫其修远兮。机器阅读理解的水平与人类阅读理解水平还有着非常大的距离要走。当前SQuAD数据集，机器阅读理解水平超越人类，马上就有研究“攻击”当时的SOTA模型，致使SOTA模型的准确率骤降。现在HotpotQA数据集引领模型的推理能力，也被本篇论文研究出这样训练出的模型，其实没有深层次的理解原文。 
### 2.2 模型
提出了一种新的模型包含了四个部分：文档选择、问题分类（决定问题是单跳还是多跳的）、单跳QA模型、多跳QA模型。如果问题被分类为单跳问题，则只经过单跳QA模型；如果问题被分类为多跳问题，则只经过多跳QA模型。   
但是训练单跳QA模型这里，用的训练集肯定包含了很多噪声。因为没有人工检验阶段，其实就相当于一种远程监督/弱监督的思想。  

### 2.3 实验
分别在已有的开源多跳推理RC模型（DFGN、DecompRC、CogQA）上验证。  
结果发现DFGN和DecompRC在多跳问题上的表现优于其对应的单跳问题，CogQA在分解后的单跳问题上表现的更好，但也没有好到哪去。  
于是作者又去观察了一下，那些多跳问题明明被模型回答对了，但是分解得到的子问题却没有被答对是什么情况。发现这些例子中：第二个子问题中的单词 与答案附近的单词有很高的相似度。

### 2.4 评价
我觉得该篇论文证明了现在已有的一些多跳推理模型还是没有理解文章的深入含义，不然不可能回答对复合问题却答不对子问题。赞同作者的观点：现有的一些算法在对复杂问题寻找答案时可能有的也是用了局部的匹配而没有按照推理链来。  
然而作者提出的模型，其实就是做了一个问题的分类，如果是子问题的话，直接不使用多跳QA模型，而使用单跳QA模型，感觉没啥意思，这样效果肯定有提升。  
所以综上所述，我觉得论文得到的结论是好的，但是它设计的模型根本没有解决出他观察出存在的问题，有点烂尾了给我的感觉。 

### 2.5 思考一下
- 为什么CogQA比DFGN或DecompRC在子问题上应用的效果好？
- 如何让模型能够通过推理链来习得答案而不是靠局部模式匹配？

## 3. arXiv 2020：Generating Followup Questions for Interpretable Multi-hop Question Answering
感觉这篇工作还没有做完，只针对bridge问题，可能为其设计了一个逐渐检索的过程。  
没啥大意义，除了相关工作的表述，我可能会看一下他引的paper  
| 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: |
|[Multi-hop Reading Comprehension through Question Decomposition and Rescoring](https://arxiv.org/abs/1906.02916)|ACL 2019|就是这个md讨论的核心论文|
|[Revealing the Importance of Semantic Retrieval for Machine Reading at Scale](https://www.aclweb.org/anthology/D19-1258/)|EMNLP 2019|[TODO]，只用原始问题检索了一次文档|
|[Multi-paragraph reasoning with knowledge-enhanced graph neural network](https://arxiv.org/abs/1911.02170v1)|arXiv 2019|[TODO]，只用原始问题检索了一次文档|
|[Answering while Summarizing: Multi-task Learning for Multi-hop QA with Evidence Extraction](https://www.aclweb.org/anthology/P19-1225/)|ACL 2019|[TODO]，只用原始问题检索了一次文档|
|[HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://doi.org/10.18653/v1/D18-1259)|EMNLP 2018|就原始HotpotQA数据集|
|[Answering Complex Open-domain Questions Through Iterative Query Generation](https://doi.org/10.18653/v1/D19-1261)|EMNLP 2019|[TODO]，预测 问题+第一跳证据与第二条证据（未见） 之间的最长公共子序列，以此来继续查询并检索文档|
|[Cognitive Graph for Multi-Hop Reading Comprehension at Scale](https://doi.org/10.18653/v1/P19-1259)|ACL 2019|就CogQA|
|[Multi-Hop Paragraph Retrieval for Open-Domain Question Answering](https://doi.org/10.18653/v1/P19-1222)|ACL 2019|[TODO]，训练了一个神经检索模型，利用问题和第一跳的信息来检索第二跳信息|

## 4. arXiv 2019：Multi-hop Question Answering via Reasoning Chains
### 4.1 动机
