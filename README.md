
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
|4|[Multi-hop Question Answering via Reasoning Chains](https://arxiv.org/abs/1910.02610)|arXiv 2019|0|0|弱引|
|5|[Complex Factoid Question Answering with a Free-Text Knowledge Graph](https://dl.acm.org/doi/10.1145/3366423.3380197)|WWW 2020|-|-|不是MRC，而是KBQA，所以没看|
|6|[Unsupervised Question Decomposition for Question Answering](https://arxiv.org/abs/2002.09758)|EMNLP 2020|1|[repo](https://github.com/facebookresearch/UnsupervisedDecomposition)|强引，提出了一种分解问题的方法|
|7|[Answering Complex Open-domain Questions Through Iterative Query Generation](https://arxiv.org/abs/1910.07000)|EMNLP 2019|1|[repo](https://github.com/qipeng/golden-retriever)|中引，评价了DecompRC的方法，提出了另外一种方法|
|8|[Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://arxiv.org/abs/1911.10470)|ICLR 2020|1|[repo](https://github.com/AkariAsai/learning_to_retrieve_reasoning_paths)|中引，也是不断检索文档，只不过是推理路径|
|9|[Self-Assembling Modular Networks for Interpretable Multi-Hop Reasoning](https://arxiv.org/abs/1909.05803)|EMNLP 2019|1|[repo](https://github.com/jiangycTarheel/NMN-MultiHopQA)|强引，挺有意思的一项工作，分了不同的板块去完成不同的单跳任务，也有拆分多跳问题为多个单跳问题的意图。|
|10|[Asking Complex Questions with Multi-hop Answer-focused Reasoning](https://arxiv.org/abs/2009.07402)|arXiv 2020|0|[repo](https://github.com/Shawn617/Multi-hop-NQG)|弱引，QG工作|
|11|[Low-Resource Generation of Multi-hop Reasoning Questions](https://www.aclweb.org/anthology/2020.acl-main.601/)|ACL 2020|0|0|弱引，QG工作|
|12|[Logic-Guided Data Augmentation and Regularization for Consistent Question Answering](https://arxiv.org/abs/2004.10157)|ACL 2020|1|[repo](https://github.com/AkariAsai/logic_guided_qa)|弱引，虽然这篇文章将的不是问题分解但我觉得也挺有意思，而且可能会有借鉴价值，讨论核心点是对比问题的数据增强|
|13|[Transformer-XH: Multi-hop question answering with eXtra Hop attention](https://openreview.net/forum?id=r1eIiCNYwS)|ICLR 2020|1|[repo](https://github.com/microsoft/Transformer-XH)|弱引，但我觉得这篇工作挺有意思的，让transformer在图结构上学习，评分686|
|14|[Robust Question Answering Through Sub-part Alignment](https://arxiv.org/abs/2004.14648)|arXiv 2020|0|0|弱引，仅相关工作引用了一下，本身也不是多跳推理阅读理解工作|
|15|[Learning to Order Sub-questions for Complex Question Answering](https://arxiv.org/abs/1911.04065)|arXiv 2019|1|0|中引，利用强化学习去选择最优的子问题回答顺序来得到最终答案|
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
通过对一个实例（问题和给定的文章）逐步建立推理链来推理出答案。推理链是原文中的句子所组成的序列，引导最终答案产生，推理链中相邻的两个句子一定要有一定的信息相关联，例如共享某一实体。  
感觉该篇论文没啥参考价值，为了训练模型能够抽取出推理链，先用了远程监督的思想（基于NER和共指消解）标注了伪推理链。又有错误传播，又是远程监督训练数据噪声大。实验效果感觉也没多好。  

### 4.2 有价值引文
| 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: |
|[Understanding dataset design choices for multi-hop reasoning](https://arxiv.org/abs/1904.12106)|NAACL 2019|[TODO]|
|[Compositional Questions Do Not Necessitate Multi-hop Reasoning](https://arxiv.org/abs/1906.02900)|ACL 2019(short)|[TODO]|

## 5. WWW 2020：Complex Factoid Question Answering with a Free-Text Knowledge Graph
[TODO]

## 6. EMNLP 2020：Unsupervised Question Decomposition for Question Answering
### 6.1 引文
正好是最新的一篇分解复杂问题的论文，然后根据几篇同类型论文的引文总结一下先有的关于分解复杂问题的工作：  
| 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: |
|[The Web as a Knowledge-Base for Answering Complex Questions](https://www.aclweb.org/anthology/N18-1059/)|NAACL 2018|大概看了一下，觉得论述在英文方面表述的很奇怪，在민세원的论文中（该表格的下一项论文）本篇作为引文举出，被阐述了主要区别。|
|[Multi-hop Reading Comprehension through Question Decomposition and Rescoring](https://arxiv.org/abs/1906.02916)|ACL 2019|[민세원女神](https://shmsw25.github.io/)的paper，也是本md讨论的核心paper，不多说了，膜就完事了。|
|[Unsupervised Question Decomposition for Question Answering](https://arxiv.org/abs/2002.09758)|EMNLP 2020|本篇| 
|[Break it down: A question understanding benchmark](https://arxiv.org/abs/2001.11770)|TACL 2020|introduce a Question Decomposition Meaning Representation (QDMR) to explicitly model this process（分解问题）|  
|[Complex question decomposition for semantic parsing](https://www.aclweb.org/anthology/P19-1440/)|ACL 2019|虽然不是阅读理解上的，但也是分解问题的工作|

其余觉得想看的引文：
| 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: |
|[Revealing the Importance of Semantic Retrieval for Machine Reading at Scale](https://www.aclweb.org/anthology/D19-1258/)|EMNLP 2019|[TODO]|


### 6.2 模型
1. 收集问题：  
首先作者收集了很多问题，S代表单跳问题集合，Q代表多跳问题集合。S初始化为SQuAD 2.0中的问题，Q初始化为HotpotQA中的问题。作者用Common Crawl中的以“wh”开头，以“？”结尾的句子来扩充Q和S，具体的扩充方法为：训练了一个fasttext文本分类器，将不同的问句分为“SQuAD 2.0”、“HotpotQA”以及“Common Crawl”三类，这个训练集共有60K条句子。然后将所有“Common Crawl”中问题分类结果为“SQuAD 2.0”的问题用来扩充S，将结果为“HotpotQA”的问题用来扩充Q。通过以上步骤，S从130K扩充至10.1M，Q从90K扩充至2.4M。
2. 检索子问题：  
对于一个多跳问题与子问题都用fasttext来表示（作者也尝试了用TF-IDF或BERT表示，但没有提升，这三个中fasttext表示是效果最好的）。给定一个多跳问题$q$，通过下式来选取子问题（基本思想为，子问题与原问题相关性尽可能大，且子问题集合尽可能覆盖全面）：
$$
(s_{1},s_{2}) = argmax[v^{T}_{q}v_{S1}+v^{T}_{q}v_{S2}-v^{T}_{s1}v_{S2}]
$$
3. 后处理子问题：  
由于我们得到子问题的方式是基于检索的，因此检索出的子问题与原问题可能不是关于同一个实体的。如果子问题中的实体没有出现在原问题$q$中，则将这个实体替换为$q$中的同类型（例如日期或地点）实体。  
这步对于```Seq2seq```（在测试时利用seq2seq模型生成子问题）以及```PseudoD```（在测试时直接利用上述方法来检索子问题）很重要，但是对```ONUS```模型没那么重要，因为```ONUS```要从分解后的子问题$d$重新构建出原问题$q$。
4. ONUS：  
One-to-N Unsupervised Sequence transduction（ONUS）。在模型具体做法上，利用MLM任务在Q和伪分解子问题上微调了（1 epoch）了基于transformer的seq2seq模型。然后又在```back-translation```以及```denoising objective```两个任务上微调。  
对于```denoising objective```任务，将问题$q$与对应子问题$d$都进行随机mask、丢弃部分字符、局部交换字符等加噪操作，然后让模型去复原。  
对于```back-translation```：利用分解后的子问题$d$，去产生原问题$q$。  
5. 单跳模型：  
整体模型的思路和[DecompRC](https://arxiv.org/abs/1906.02916)差不多。   
单跳QA模型的选择了```RoBERTa-large```，然后在SQuAD 2.0与HotpotQA-easy上进行预训练。预训练了两个单跳QA模型，然后进行集成。预训练好之后就当成一个黑盒不再改变参数了。  
然后对于一个多跳问题，先利用```ONUS```分解成单跳问题，接着对于一个单跳问题，将问题与不同的段落分别送至单跳QA模型，模型会在该段落下得出答案span以及有无答案的概率（yes和no也视为一种span）。所以对于一个单跳问题，产生了 段落个数 个答案，成这些答案为单跳QA问题的候选答案。然后基于这些答案和答案概率计算每个候选答案概率，再将概率最大的答案span作为子问题的最终答案。   
6. 重组模型：  
和单跳模型一致，只不过训练集的输入是子问题和子问题答案，输出是原始问题答案。每个(sub-question, sub-answer)之间用[SEP]相隔。

## 7. EMNLP 2019：Answering Complex Open-domain Questions Through Iterative Query Generation
### 7.1 动机
还是为了解决短视检索问题。在多跳问题中，有些中间实体非常重要，但它一开始没有在问题中，所以需要对初次检索得到的文档中的内容进行二次检索。于是本文提出了```GOLDEN```（Gold Entity）检索模型，每一步模型会利用之前检索得到文档内容去产生一个新的query，再利用新的query去检索文档，以更好的适应多跳问题。 

### 7.2 模型
1. query生成：  
每一次输入原问题和已经被检索到的文档，然后输出query，query是输入的一个span。可以看出query生成模型与QA模型很像，都是输入文本，输出一个目标，只不过这里的目标不是回答问题而是生成下一跳的query。自然采取的是单跳QA模型，这里选用的是```DrQA```。  
对于训练数据，比较启发式的检索已有信息（问题与之前检索到的文档，如果是第一步则只有问题）和期望在下一跳查询到的文档之间具有最高重叠率的连续跨度，将这个连续跨度作为oracle query。计算重叠率的时候测试了多种方法，例如最长公共子序列、最长公共子串、前两者的结合。  
2. 利用query检索新文档  
就正常的基于BM25的检索，之后会提高一下那些标题能够匹配到query的文档的分值，具体提高多少看匹配的情况。  
3. 生成答案  
每次query出一篇新的文档，对于hotpotQA就query两次得到两个文档即可，然后利用一个QA模型产生答案，在```BiDAF++```模型上进行了修改：（1）由于两个文档之间没有顺序性，如果用一个RNN直接将concat后的文档进行编码则表示会受到被concat的两个文档的顺序影响，因此在编码时，对两个文档分别用一个共享的RNN进行编码，从而得到两个文档的表示。（2）将原有的注意力机制都改为self-att机制。  

### 7.3 其余有价值观点
作者在评价自己的方法（利用query不断检索新的文档）与分解子问题检索文档的方法时，说到：  
>It is also more generally applicable than question decomposition approaches (Talmor and Berant, 2018; Min et al., 2019b), and does not require additional annotation for decomposition.

## 8. ICLR 2020：Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering
### 8.1 动机
还是短视检索问题，然后本篇论文作者的想法是在整个wikipedia上面去建立一个段落间的图，然后这个图中每个结点代表一个paragraph，结点之间的连线靠的是wikipedia的一些超链接和段落间的链接，这样不同paragraph之间的关系是依靠着语义建立起来的，而不是文本表面的匹配。由此来解决短视检索问题。
### 8.2 模型
1. 建立图  
没什么好说的，就是根据超链接和段落间的链接建立的。构建出的图也比较稠密。
2. 检索器  
利用question和已经检索出的文档，从**候选集**里检索下一跳需要的文档。   
先利用BERT编码**候选集**中的所有文档，然后利用已有的信息$h_{t}$（包含已检索到的文档与问题信息）计算当前步检索到别的文档的概率，以此来习得推理路径。然后根据当前步检索出的文档的表示再更新已有信息的表示$h_{t+1}$。**候选集的话在一开始是根据question基于TFIDF的方法检索到的，之后就是当前步新检索出的结点所连接的其余结点集合（这里有一个小trick，就是除了当前步新检索出的结点所连接的其余结点之外，还会加上上一次检索得到的结点列表中，除去那个被选中的结点之外，其余结点中排名最高的也会被加入。）**。在检索的过程中也用到了Beam Search，所以检索器最后会输出多条（取决于Beam size）路径之后再由阅读器处理。
$$
w_{i} = BERT_{[CLS]}(q,p_{i}) \\
p(p_{i}|h_{t}) = \sigma(w_{i}h_{t}+b) \\
h_{t+1} = RNN(h_{t},w_{i})
$$  
3. 数据增强与负采样
- 数据增强：为了训练上述检索器，首先得到一条真值推理路径$g = [p_{1},p_{2},...,p_{g}]$（通过已有的标注数据来获得，$p_{g}$代表的是一个特殊符号[EOE]表示一条路径的终端。），然后增加一条新的推理路径$g^{'} = [p_{\alpha},p_{1},p_{2},...,p{g}]$, $p_{\alpha}$属于$C_{1}$，是TFIDF分最高的一个段落，并且和$p_{1}$相联系。增加这样的新路径有利于Retriever能够在第一次通过TFIDF没有检索到正确文档的情况下，也能进而检索出正确的推理路径。  
- 负采样：两种负采样策略，基于TFIDF或基于超链接。单轮QA只用基于TFIDF，多轮则两个都用。负采样数为50。
4. 阅读器  
利用多任务学习来训练阅读器：（1）对于每条路径进行重排序、（2）阅读理解，在最有可能出现答案的路径上抽取出答案。  
（1）对于路径选择任务来说，也是利用了BERT，输入问题q与一条路径（所有文档concat）给BERT，然后利用[CLS]获取整条路径的表示，接着通过分类任务预测出这条路径的重要程度。训练数据的话是通过构建负例来训练的。      
对于阅读理解任务来说，直接使用BERT，输入concat后的文档与问题，输出答案区间，没有什么额外的特殊设计。    


## 9. EMNLP 2019：Self-Assembling Modular Networks for Interpretable Multi-Hop Reasoning
### 9.1 动机
使用模块化的思想，设计了三种不同的模块来完成不同的推理操作，每种模块只能完成一跳的推理，多个模块组合起来便可以完成一个多跳推理问题。    
具体地，利用```Controller```来预测当前步的```Modular```，有三个不同的```Modular```：```Find```用于找到子问题的答案；```Relocate```使用```Find```模组找到的答案来化简问题并继续调用```Find```来搜索化简后的问题的答案；```Compare```模组用来比较前两个```Find```找到的实体信息，并输出一个向量进而判断答案是yes or no。除此之外，还有一个比较特殊的模组```NoOp```代表没有操作，就是跳过的一次，也就是当前步不需要进行任何推理行为的意思。  
### 9.2 模型  
context经过LSTM的上下文有关表示为$h$（二维向量），问题经过LSTM的上下文有关表示为$u$（二维向量），问题经过self-attention的表示为$qv$（一维固定向量）。   
- Controller（控制器）  
  - Controller计算下一步的推理行为（即计算下一步采用哪个模组）：Controller先计算问题在第t步的表示$q_{t} = W_{1,t}qv + b_{1,t}$，然后融入之前Controller的隐层表示（下面会讲）：$cq_{t} = W_{2}[c_{t-1};q_{t}] + b_{2}$ ，最后计算下一个模组的概率分布：$p_{t,i} = softmax(W_{3}cq_{t})$。这里的$W_{3}$会将参数降维至4表示四个模组（```Find```、```Relocate```、```Compare```以及```NoOp```）
  - Controller更新当前步的隐层表示：在计算下一步推理模组时，使用到的$c_{t-1}$代表上一步的Controller隐层表示，那么当我们已经得到了下一步的推理模组时，需要更新Controller的隐层表示。具体来说，先计算$cq_{t}$（融合了问题表示与之前控制器的状态信息）对上下文$h$中每个token的权重：$e_{t,j} = W_{4}(cq_{t}u_{j}) + b_{4}$，$cv_{t} = softmax(e_{t})$，最后基于该权重来计算上下文的表示向量（一维）：$c_{t} = \sum_{j}cv_{t,j}u_{j}$。**向量$c_{t}$（1）既表示当前Controller的隐层状态，用于计算下一步模组选取的概率分布；（2）又代表着子问题的表示，因为它是在context上构建的，表示了当前步子问题**。
- Modulers
所有的模组输入都是问题表示$u$，上下文表示$h$以及子问题表示$c_{t}$。核心模组（```Find```与```Relocate```）的输出是基于bi-attention在context上构建的一个**attention map**。为了保证能够微分，且能够完成不同模组之间的交互，这个**attention map**会入栈和出栈。例如在对比问题上，我们先将两个实体/事件通过```Find```模组得到的attention map入栈，然后再使用```Compare```模组去对比，这时就要用到之前计算出的两个**attention map**，因此需要对这两个**attention map**做出栈操作。
  - ```Find```：主要功能是基于子问题表示$c_{t}$，上下文表示$h$，先生成一个更加健壮的子问题表示$h^{'} = hc_{t}$，我的理解是$c_{t}$其实已经包含了权重平均后的context的表示，是一个一维向量，然后用它跟原先上下文表示$h$中每个token的表示做乘积，那么原先权重大的token，在这里乘积也应该更大，最终得到的$h^{'}$融入了子问题表示（同样也来源于上下文表示）和上下文表示，**但最重要的是$h^{'}$是一个二维向量**，所以它可以用来跟我们的问题表示$q$进行Bi-attention计算。接着，我们将Bi-attention计算的结果（是一个在context上的attention map）入栈。
  - ```Relocate```：主要功能是用之前的```Find```模块找到的答案（本质是一个att map）来进行二跳检索，例如一个问题“A的妻子从事什么职业？”，我们先利用```Find```模块，context以及子问题表示（A的妻子）来在context上建立一个att map，这个att map代表着答案（源于context的某一处上下文），然后接着我们就可以利用```Relocate```模组，给它输入（1）第一步att map代表的答案影响后的上下文表示$h_{b}$。（2）原多跳问题表示$u$以及（3）当前步子问题表示，也就是Controller的隐层表示$c_{t}$。然后```Relocate```模组会输出另一个att map表示第二跳问题的答案（同样本质也是在context上的一个att map）。其余两个输入比较简单，我们来看一下$h_{b}$是如何构建的：首先我们将之前的att map出栈，然后利用该att map计算出答案表示：$b = \sum_{s}att_{1,s}h_{s}$，可以看出$b$是一个由context构建而成的一维向量，在这个att中，第一条问题的答案区间中的token的权重占比应当比较大，所以我们可以将$b$看成是答案的表示，然后再基于这个答案表示$b$构建一个二维的上下文表示（受第一跳问题答案影响的上下文表示）$h_{b} = hb$，这样我们就得到了$h_{b}$，最后我们调用```Find($u$, $h_{b}$, $c_{t}$)```得到一个新的att map代表二跳问题的答案，并将这个新的att map入栈。
  - ```Compare```：主要功能是在某一属性上对比两个实体/事件。一般情况下再使用```Compare```模组之前，栈内会有两个att map，代表通过```Find```模组找到了关于这两个实体/事件的这一属性。然后将这两个att map出栈加工得到对应的属性表示，再进行分类计算出yes or no即可。我们来看一下具体操作，首先出栈两个att map，然后计算得到两个实体/事件相应的属性表示：$h_{s1} = \sum_{s} att_{1,s}h_{s}$与$h_{s2} = \sum_{s} att_{2,s}h_{s}$，然后我们构建一个新的向量，融入这两个属性的表示、子问题表示$c_{t}$以及这两个属性之间基于$c_{t}$的权重差距表示：$o_{in} = [c_{t}; h_{s1}; h_{s2}; c_{t}(h_{s1}-h_{s2})]$，然后进行线性与非线性变换处理得到：$m = W_{1}(ReLU(W_{2}o_{in}))$。
  - ```NoOp```：跳过跳过，无操作，因为不可能每一步都有一个推理举动。
- Prediction Layer：
我们先把问题表示$qv$以及模组隐层状态表示$m$（好像是所有模组输出的加权平均？）concat起来预测这个问题的答案到底是span还是yes/no；如果是span的话，我们把栈内最后一个att map出栈，然后基于这个att map映射一下得到每个token为span start概率和span end概率，之后懂的都懂；如果是yes/no的话，把```compare```模组最后产生的向量$m$，然后做一个二分类预测。  


## 10. arXiv 2020:Asking Complex Questions with Multi-hop Answer-focused Reasoning
又是一个多跳问题生成的工作，貌似被EMNLP2020拒了？然后作者强调了自己的论文与[*Semantic graphs for generating deep questions.*](https://arxiv.org/abs/2004.12704)的不同

## 11. ACL 2020：Low-Resource Generation of Multi-hop Reasoning Questions
也是一篇多跳问题生成的研究，利用少量的带标签的数据与大量无标注数据训练了QG模型。  
实验时分别测试了只使用不同比例（10%~100%）的hotpotQA训练集，用自己的方法扩充伪数据来增加训练样本依次提升性能。    

## 12. ACL 2020：Logic-Guided Data Augmentation and Regularization for Consistent Question Answering
### 12.1 动机
本论文主要的动机是结合逻辑规则和神经网络模型提高**比较题答案**的准确性和一致性。使用了逻辑和语言上的知识去增强带标签的训练数据，接着使用了一个基于一致性的归一化器来训练模型。所以这篇论文主要研究的点就是对比性的推理问题（在某一个属性方面对比两个实体或事件）。  
### 12.2 模型
- 基于规则的数据增强
例如有一个训练数据$X = {q, p, a}$，（1）基于**对称一致性**增强：得到$X^{'} = {q_{sym}, p, a_{sym}}$。（2）基于**传递一致性**增强：$(q_{1}, p, a_{1}) \wedge (q_{2}, p, a_{2}) \to (q_{trans}, p, a_{trans})$，其中$q_{1}$的结果是$q_{2}$的原因。
- 训练目标   
loss分为两部分，答案预测loss以及一致性loss：$L = L_{task}(X) + L_{cons}(X, X_{aug})$；其中一致性loss又分为两部分：$L_{cons} = \lambda_{sym}L_{sym} + \lambda_{trans}L_{trans}$。   
- 退火（Annealing）：刚开始训练时只在答案预测loss上做训练，训练几轮之后再引入一致性loss。

## 13. ICLR 2020：Transformer-XH: Multi-hop question answering with eXtra Hop attention
### 13.1 动机
从名字上不难看出本论文是对`Transformer`的一个改进。以往的序列建模，不论是`Transformer`还是`Transformer-XL`都只是在文本表面的序列上进行建模的。而有些文本之间是存在着固有的结构（树或图）。例如在`Wikipedia`中，许多文档之间实际上是有联系的（体现在超链接上），以往的模型没有考虑到文本之间的图结构。这种图结构对于多跳推理阅读理解来说是非常重要的，因为多跳推理往往需要一条完整的推理链才能回答问题，而推理链就是在这样的图结构中产生的。所以本文在对文本建模时，使用到了另外一种注意力机制：e**X**tra **H**op attention。
### 13.2 模型  
直接说应用于hotpotQA的做法吧。
1. 构建图结构：基于TFIDF抽取一些文档出来，再把与问题中出现过的实体相关的文档挑出来，形成一个初选文档集合，然后对于这个集合中的每一个文档，利用BERT计算出相关度，最后保留前K个相关度的文档。接着把这K个文档相连接的文档也搞出来组成最终文档集合。然后在这个最终文档集合上构建一个图结构，每个结点代表一个文档，边代表图中的超链接关系。
2. 为每个文档中的token初始化，每个文档在输入到`Transformer-XH`之前，先tokenize，然后组成“[CLS] + question + [SEP] + anchor text + [SEP] + paragraph + [SEP]”的形式。其中anchor text指的是，指向这个文档的超链接上的文字（存在于父文档中）。然后在每层`Transformer-XH`中，所有的token都会现在本文档中进行`vanilla self-attention`（在论文中被称为in-sequence attention），然后重点来了，我们现在想让当前计算的这个文档能够get到其余和它相连接的文档的表示，具体的做法是（eXtra Hop attention）：用[CLS] token的表示获取代表本文档的query向量，然后与相连接的文档的[CLS] token表示进行self-attention。这样本文档的[CLS] token就融入了别的文档的表示。需要注意的是，在本层中[CLS] token经历了两个attention（in-sequence attention与eXtra Hop attention）的交互，所以最后还通过了拼接加线性层的方法将两者融入的信息结合起来。然后在下一层传递信息的时候，本文档中其他token的表示将会从本层和其他文档交互过的[CLS] token中也得到了来自其他文档中的信息。
3. 基于每个文档的表示，预测每个文档包含答案的概率（二分类），选取最高概率的文档接着进行span预测，再得到最终答案。

## 14. arXiv 2020：Robust Question Answering Through Sub-part Alignment
将上下文和问题利用现有的工具分解成更小的单元（predicates and corresponding arguments），然后建立图中的结点，接着利用predicate argument relation和共现关系建立边，最后将问题对齐到上下文图中来寻找答案。  
这样会获得更好的鲁棒性。在攻击实验上效果比较好。  
但本文和多跳推理无关。   

## 15. arXiv 2019：Learning to Order Sub-questions for Complex Question Answering
利用强化学习对分解负责问题得到的子问题集合进行了排序以确认子问题的回答顺序。例如对一个复杂问题“那个城市被 由JK罗琳写过的小说改编成的电影 取过景且举办过奥运会”，分解可以得到三个问题：“JK罗琳写过的小说是？”、“哈利波特在哪些城市取过景？”、“哪些城市举办过奥运会？”。显然第三个问题很广泛，如果先回答第三个子问题，搜索空间就很大，所以有必要进行这个子问题序列的研究。需要注意的是第三个子问题，不受第一个和第二个问题回答的限制，因此它是自由的。有多种子问题回答的顺序都能够得到复杂多跳问题的答案，但我们人类在做这种题的时候就会考虑到每个子问题的难度情况来选择最轻松的顺序以得到最终答案。

# Part 3 开放式问答
根据[ACL 2020 openqa Tutorial](https://github.com/danqi/acl2020-openqa-tutorial)整理  
focu事实性的基于非结构化数据（文本）的open QA  

## 1.数据集
| 类型 | 数据集 |
| :---: | :---: |
| 推理挑战 | ```Facebook bAbI```（2015）、```AI2 ARC```（2018）、```Multi-RC```（2018）|
| 多轮问答 | ```SQA```（2017）、```QuAC```（2018）、```CoQA```（2019）|
| 多跳问答 | ```HotpotQA```（2018）、```OBQA```（2018）、```QASC```（2020）|
