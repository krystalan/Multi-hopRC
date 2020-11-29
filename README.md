
# 多跳阅读理解相关论文
## 1 数据集
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600)|EMNLP 2018|`HotpotQA` **抽取式** 每个QA对有10个（distractor setting）或百万级（full wiki setting）对应的paragraph|
|2|[Constructing Datasets for Multi-hop Reading Comprehension Across Documents](https://arxiv.org/abs/1710.06481)|TACL 2018|`Wikihop`与`MedHop` **多选式** 每个QA对有数量不等（平均14，最多64）个对应的paragraph，还有候选答案集合|
|3|[Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://www.aclweb.org/anthology/D18-1260/)|EMNLP 2018|`OpenBookQA` `CommonQA` **多选式** 包含了两部分5957个多选问题（每题四个选项），1326个初级科学事实。科学事实通常不能够直接回答问题，对于一个问题，应当先检索相关的科学事实，然后加上常识（不在本数据集中提供，通常用`ConceptNet`）才能得到答案。需要多跳推理，也需要常识推理。|
|4|[Reasoning Over Paragraph Effects in Situations](https://arxiv.org/abs/1908.05852)|EMNLP MRQA Workshop 2019|`ROPES` **抽取式** 每个QA对对应两个paragraph，一个称为*background*是说明文，另一个是*situation*由众包工人创建，回答问题需要把*background*中的知识应用与*situation*才可|
|5|[DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://arxiv.org/abs/1903.00161)|NAACL 2019|`DROP` **答案不一定出现在原文中 需要通过计算、计数等操作得到** 每个QA对都有一个对应的paragraph|  
|6|[DREAM: A Challenge Data Set and Models for Dialogue-Based Reading Comprehension](https://www.aclweb.org/anthology/Q19-1014/)|TACL 2019|`Dream` **多选式** 10197个问题，并在英语考试题中收集6444个多轮多方的对话数据。多个选择题对应一个多轮多方对话数据，只有一个候选答案正确|
|7|[Looking Beyond the Surface: A Challenge Set for Reading Comprehension over Multiple Sentences](https://www.aclweb.org/anthology/N18-1023/)|NAACL 2018|`MultiRC` **多选式** 每个实例包含一个问题，多个sentences以及答案候选集，根据sentences去选择正确的答案，且可能有多个候选答案正确，每个实例的候选答案数量不一定一致|
|8|[QASC:A dataset for question answering via sentence composition](https://arxiv.org/abs/1910.11473)|AAAI 2020|`QASC` **多选式** 也是基于多个句子间的推理从候选答案中选出正确答案的数据集。含有9980个八项选择题，每个问题都被标注了两个fact sentences用来推理出最终的答案。还提供了一个包含了17M个句子的语料库，所有的fact sentences都在里面|
|9|[Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering](https://arxiv.org/abs/2010.03274)|EMNLP 2020|`eQASC`、`eQASC-perturbed`以及`eOBQA` `Textual entailment` **多选式** 这个工作其实是在研究多跳推理问题的可解释性。在`QASC`数据集的基础上，针对于每个问题又标注了多个推理链（有效或无效都有）构成了`eQASC`，接着又将推理链模板化（使推理链更加通用）构成了`eQASC-perturbed`数据集，最后为了进行out-domain test，也基于`OpenBookQA`数据集标注了推理链形成了`eOBQA`并进行测试|

## 2 数据增强
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Generating Multi-hop Reasoning Questions to Improve Machine Reading Comprehension](https://dl.acm.org/doi/pdf/10.1145/3366423.3380114)|WWW 2020|QG工作|
|2|[Asking Complex Questions with Multi-hop Answer-focused Reasoning](https://arxiv.org/abs/2009.07402)|arXiv 2020|QG工作|
|3|[Low-Resource Generation of Multi-hop Reasoning Questions](https://www.aclweb.org/anthology/2020.acl-main.601/)|ACL 2020|QG工作|
|4|[Logic-Guided Data Augmentation and Regularization for Consistent Question Answering](https://arxiv.org/abs/2004.10157)|ACL 2020|讨论核心点是对比问题的数据增强，基于对称一致性和传递一致性来增强训练样本|

## 3 在本质方面的探索
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Do Multi-Hop Question Answering Systems Know How to Answer the Single-Hop Sub-Questions?](https://arxiv.org/abs/2002.09919)|arXiv 2020|研究问题：多跳QA系统能否回答单跳子问题；除此之外引用了`DecompRC`中划分子问题的方法|
|2|[Is Graph Structure Necessary for Multi-hop Question Answering?](https://www.aclweb.org/anthology/2020.emnlp-main.583)|EMNLP 2020|`non-Open` 改进了`DFGN`模型，探索了图结构在多跳QA中是否必要，结论是：如果PTM是特征提取器的话就重要，如果微调PTM的话其实图结构不是很重要|
|3|[Do Multi-hop Readers Dream of Reasoning Chains?](https://www.aclweb.org/anthology/D19-5813)|EMNLP 2019|个人觉得这篇其实没有NAACL 2019那篇那么精彩，但也是不错的，作者也是做了一些分析实验，然后得出结论：一些单跳模型在面对`HotpotQA`问题也能做的还不错，然后推理链其实是有用的，但现在没有被挖掘的很深入，未来可能会有大用处。|
|4|[Compositional Questions Do Not Necessitate Multi-hop Reasoning](https://arxiv.org/abs/1906.02900)|ACL 2019(short)|作者发现`hotpotQA`中许多多跳问题都能够被单跳模型回答正确，于是展开了分析|
|5|[Understanding dataset design choices for multi-hop reasoning](https://arxiv.org/abs/1904.12106)|NAACL 2019|非常棒的一个工作，精彩精彩。在`HotpotQA`和`Wikihop`数据集上展开研究，发现多跳阅读理解数据集单跳也能回答对，`Wikihop`不看文章也能答对，Span式的多跳数据集优于多选式的，即使Span式的数据集中有很多问题单跳也能答对|
|6|[Is Multihop QA in DiRe Condition? Measuring and Reducing Disconnected Reasoning](https://www.aclweb.org/anthology/2020.emnlp-main.712)|EMNLP 2020|`HotpotQA`上的一个研究，也是探索了当前多跳模型到底有没有推理能力。作者将推理分为连贯推理（connected resoning）与不连贯推理。连贯推理是我们希望能够赋予模型的能力，也就是在多个文档中进行信息间的交互以此得到答案。而不连贯推理则是不交互信息就得到答案。作者设计了一些实验去探索模型通过不连贯推理所能达到的分数以此来说明现有的模型可能并没有达到我们想要赋予他们多跳推理能力的初衷。除此之外，作者还设计了一种方案将数据集进行转换以此让模型更难cheat。该工作有着非常高的借鉴价值。对于模型设计者来说，可以帮助我们鉴别自己设计的模型到底有没有连贯推理，对于数据集制造者来说，可以让其明白自己的数据集容不容易被cheat|
|7|[Avoiding Reasoning Shortcuts: Adversarial Evaluation, Training, and Model Development for Multi-Hop QA](https://www.aclweb.org/anthology/P19-1262)|ACL 2019|`non-Open` 这篇文章也揭示了`HotpotQA`数据集有些问题不用推理也能回答，他们设置了攻击实验，发现在对抗数据集（通过在答案区间以及支撑文档的标题上进行短语级别的干扰得到，这样模型如果还是是用推理捷径的话将会得到多个可能的答案，从而影响模型的表现）上现有的SOTA模型表现都会下降很多，除此之外他们设计了一个控制单元来指导模型进行多跳推理。|

## 4 逐渐检索文档（检索到了就要用）
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Answering Complex Open-domain Questions Through Iterative Query Generation](https://arxiv.org/abs/1910.07000)|EMNLP 2019|`Open` GOLDEN模型|
|2|[Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://arxiv.org/abs/1911.10470)|ICLR 2020|`Open` 不断检索文档，在整个wikipedia的文章上建立了一个图，然后一直跳|
|3|[Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval](https://arxiv.org/abs/2009.12756)|arXiv 2020|`Open` 没啥感觉，把检索文档看成序列建模问题然后beam search|
|4|[Cognitive Graph for Multi-Hop Reading Comprehension at Scale](https://doi.org/10.18653/v1/P19-1259)|ACL 2019|`Open` 就CogQA|
|5|[Unsupervised Alignment-based Iterative Evidence Retrieval for Multi-hop Question Answering](https://www.aclweb.org/anthology/2020.acl-main.414)|ACL 2020|`Open`&`non-Open` 在`MultiRC`以及`QASC`上的工作，做与问题相关的支撑句的检索。采用无监督的对其方法。每次检索会修改query，修改的规则主要是当前检索到的句子没有包含query中的那些terms，检索也使用了基本的基于词向量语义相似度的方式，属于无监督模型。但达到了很好的效果。|  
|6|[PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text](https://arxiv.org/abs/1904.09537)|EMNLP 2019|`MetaQA`、`WebQuestionsSP`以及`Complex WebQ` `Open` 从两种实体源中检索知识资源：文本语料库与知识库。作者定义了问题子图（question subgraph）这一概念，它的作用是包含足够多的与问题相关的信息然后以此来回答一个问题，该图是迭代生成的，且分为三类结点：①实体结点（存在于KB中）②文本结点（通常情况下是文本语料库中某个实体的mention）③事实结点（KB中的事实三元组）。一开始先仅利用问题中的信息来初始化问题子图，之后进行T轮迭代扩充，每次扩充会选取问题子图中的部分结点，对每一个选取到的结点，为其检索其相关的（1）文档（2）事实，对于（1）还会进一步利用实体链接模型提取其中的实体mention，对于（2）来说会提取三元组中的头尾实体。这样的构建直到问题子图可以回答问题为止，之后再进行答案预测。|
|7|[Explore, Propose, and Assemble: An Interpretable Model for Multi-Hop Reading Comprehension](https://arxiv.org/abs/1906.05210)|ACL 2019|分为三部分，三部分联合优化。Document Explore：一个级联的memory network迭代式地选取相关文档；Answer Proposer：对于推理树上的每一个从跟到结点的推理路径提出一个proposed答案；Evidence Assembler：从每一条推理路径上提取包含proposed答案的关键句，并将这些关键句结合起来以预测最终的答案。|
|8|[Multi-Hop Paragraph Retrieval for Open-Domain Question Answering](https://doi.org/10.18653/v1/P19-1222)|ACL 2019|`Open` 主要在开放式的多跳QA中做检索方面的迭代式探索尝试，提出了MUPPET(MUlti-hoP Paragraph rETrieval)，共有两个模块组成：**段落和问题编码器**以及**段落阅读器**。编码器负责获取段落的表示以及将问题编码成搜索向量，阅读器通过搜索向量基于最大化内积的方法检索相关度高的段落。在每次迭代中，搜索向量会由前几步检索得到的文档的表示所印象，所以每次迭代中的搜索向量不一样，因此才能迭代式地检索到不同的段落。|

## 5 动态检索文档（检索了不一定要用）
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[DDRQA: Dynamic Document Reranking for Open-domain Multi-hop Question Answering](https://arxiv.org/abs/2009.07465)|arXiv 2020|`Open`|

## 6 分解问题相关（其实就是逐渐检索文档的一大主流）
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[The Web as a Knowledge-Base for Answering Complex Questions](https://www.aclweb.org/anthology/N18-1059/)|NAACL 2018|大概看了一下，觉得论述在英文方面表述的很奇怪，在민세원的论文中（该表格的下一项论文）本篇作为引文举出，被阐述了主要区别。|
|2|[Multi-hop Reading Comprehension through Question Decomposition and Rescoring](https://arxiv.org/abs/1906.02916)|ACL 2019|[민세원女神](https://shmsw25.github.io/)的paper，不多说了，膜就完事了。|
|3|[Unsupervised Question Decomposition for Question Answering](https://arxiv.org/abs/2002.09758)|EMNLP 2020|定义了一种无监督的方法来分解出子问题| 
|4|[Break it down: A question understanding benchmark](https://arxiv.org/abs/2001.11770)|TACL 2020|从10个复杂问题的QA数据集中提取问题并标注分解，一共定义了13个基本分解操作，以及3个高级分解。提出了`BREAK`数据集，包含了83978个复杂问题及其对应的分解。又提出了`BreakRC`模型，将考虑问题分解应用于`HotpotQA`(fullwiki setting)中|  
|5|[Complex question decomposition for semantic parsing](https://www.aclweb.org/anthology/P19-1440/)|ACL 2019|虽然不是阅读理解上的，但也是分解问题的工作|
|6|[Learning to Order Sub-questions for Complex Question Answering](https://arxiv.org/abs/1911.04065)|arXiv 2019|利用强化学习去**选择最优的子问题回答顺序**来得到最终答案|
|7|[Generating Followup Questions for Interpretable Multi-hop Question Answering](https://arxiv.org/abs/2002.12344)|arXiv 2020|`non-Open` 也是分解问题的工作，但感觉有点简单？|

## 7 推理链
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Multi-hop Question Answering via Reasoning Chains](https://arxiv.org/abs/1910.02610)|arXiv 2019|`non-Open`|  
|2|[Exploiting Explicit Paths for Multi-hop Reading Comprehension](https://www.aclweb.org/anthology/P19-1263)|ACL 2019|`Wikihop`与`OpenBookQA` `Open`&`non-Open` 这篇工作主要的贡献在于多跳阅读理解的可解释性，为了在文本数据上达到多跳的效果，会有两种方法：GNN或者路径抽取，GNN可解释性非常差，因为它是隐式地完成信息传递。而路径抽取的方法解释性强，但如果跳数增多的话会有语义漂移问题。不过`Wikihop`或者`OpenBookQA`数据集都是两跳，所以好像不严重？然后作者就通过在问题中提取头实体，在候选答案中提取尾实体，然后在候选文档中试图抽取多个推理链，接着对推理链的实体做表示初始化然后隐式提取关系，再通过关系计算路径的表示。最后会对路径进行打分，然后根据分数得到最终的答案概率分布。我个人觉得这篇工作利用两个实体的表示去直接计算他们的关系表示这里有点粗糙了，因为两个实体之间可能存在着不止一种关系，而利用作者所给的式子则无法对这种多样的关系进行学习。|

## 8 NMN
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|0|[Deep Compositional Question Answering with Neural Module Networks](https://arxiv.org/abs/1511.02799)|CVPR 2016|NMN鼻祖，在VQA中定义了多个模组来完成不同的操作：`Attention`定位图像中某Obj的位置 `Re-attention`在att map上进行位置迁移等等|
|1|[Self-assembling modular networks for interpretable multi-hop reasoning](https://arxiv.org/abs/1909.05803)|EMNLP 2019|`non-Open` 在`HotpotQA`上的工作，入栈出栈，三个模组`Find`、`Relocate`以及`Compare`|
|2|[Neural module networks for reasoning over text](https://arxiv.org/abs/1912.04971)|ICLR 2020|`non-Open` NMN在`DROP`上的工作，设计了10个模组|
|3|[Text Modular Networks: Learning to Decompose Tasks in the Language of Existing Models](https://arxiv.org/abs/2009.00751)|arXiv 2020|`non-Open` 在`DROP`上的工作，也可以用于`HotpotQA`，两个模组`next-question generator`与`QA model`|
|4|[Multi-Step Inference for Reasoning Over Paragraphs](https://arxiv.org/abs/2004.02995)|EMNLP 2020|`non-Open` 感觉本文，在NMN上的创新度不及其余文章，然后选取的数据集也不是主流的`HotpotQA`，而是`ROPES`|

## 9 针对于多跳MRC问题的PTM改进
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Transformer-XH: Multi-hop question answering with eXtra Hop attention](https://openreview.net/forum?id=r1eIiCNYwS)|ICLR 2020|让transformer在图结构上也进行学习，评分686|

## 10 单次检索
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[A Simple Yet Strong Pipeline for HotpotQA](https://arxiv.org/abs/2004.06753)|arXiv 2020|一种非常简单的方法但达到了非常不错的效果，值得思考|
|2|[Hierarchical Graph Network for Multi-hop Question Answering](https://arxiv.org/abs/1911.03631)|EMNLP 2020|构建了一个异质图包含四类结点和七类边，利用GNN来进行多跳推理|
|3|[Select, Answer and Explain: Interpretable Multi-hop Reading Comprehension over Multiple Documents](https://arxiv.org/abs/1911.00484)|AAAI 2020|`non-Open` SAE|
|4|[Dynamically fused graph network for multi-hop reasoning](https://arxiv.org/abs/1905.06933)|ACL 2019|`non-Open` DFGN|  
|5|[Revealing the Importance of Semantic Retrieval for Machine Reading at Scale](https://www.aclweb.org/anthology/D19-1258/)|EMNLP 2019|`Open` 只用原始问题检索了一次文档，但在这一次检索中，先利用了基于TFIDF的方法筛选一遍，然后对于每个paragraph又通过语义相似度计算再筛选一遍，然后将还剩下的paragraph分解成句子，再在句子级别利用语义相似度计算得到最终所有的支撑句。最终利用支撑句和问题进行答案预测|
|6|[Multi-paragraph reasoning with knowledge-enhanced graph neural network](https://arxiv.org/abs/1911.02170v1)|arXiv 2019|`Open`&`no-Open` 主要贡献在于，从paragraphs上建立了一个KG，然后利用GNN更新结点表示，结点表示更新之后返回来更新paragraphs的表示，最后进行答案预测|

## 11.与其他任务相关联  
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Repurposing Entailment for Multi-Hop Question Answering Tasks](https://www.aclweb.org/anthology/N19-1302)|NAACL 2019|`Textual entailment` `OpenBookQA`与`MultiRC` 使用文本蕴含模型来完成多跳推理问答，模型分为两部分：相关句提取与信息聚合。相关句提取对每个候选句计算其蕴含假设（由答案和问题构成）的概率，这些概率表示每句的重要程度。信息聚合则利用上一步计算的概率为不同的句子聚合表示，最后再通过一个文本蕴含模型得到最终的蕴含概率。|   

## 12. Numerical Reasoning
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|0|[DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://arxiv.org/abs/1903.00161)|NAACL 2019|`DROP` **答案不一定出现在原文中 需要通过计算、计数等操作得到** 每个QA对都有一个对应的paragraph| 
|1*|[Numnet: Machine reading comprehension with numerical reasoning](https://arxiv.org/abs/1910.06701)|EMNLP 2019|`DROP` 从context中构建一个异质有向图出来，结点都代表context中出现的数字，分为两类：在问题中出现的数字与在段落中出现的数字。边也分为两类，根据数字间的大小关系进行连接，两类边刚好互补。|
|2*|[Question Directed Graph Attention Network for Numerical Reasoning over Text](https://arxiv.org/abs/2009.07448)|EMNLP 2020|`no-Open` 在`DROP`上的工作，所以自然不用多次检索，每个QA对给定了一个paragraph。建立了一个异质图，结点有：数字（八大类型）、实体，边有：相同类型的数字之间的边以及数字和实体在一句话内共现的边。主要的idea就是在图上引入了数字的类型以及与数字相关的实体，是针对于`NumNet`的改进|   
|3|[Neural module networks for reasoning over text](https://arxiv.org/abs/1912.04971)|ICLR 2020|`non-Open` NMN在`DROP`上的工作，设计了10个模组|
|4|[Text Modular Networks: Learning to Decompose Tasks in the Language of Existing Models](https://arxiv.org/abs/2009.00751)|arXiv 2020|`non-Open` 在`DROP`上的工作，也可以用于`HotpotQA`，两个模组`next-question generator`与`QA model`|  

(*代表仅属于本分类下的工作)

## 13.利用GNN进行多跳推理
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Multi-hop Reading Comprehension across Multiple Documents by Reasoning over Heterogeneous Graphs](https://www.aclweb.org/anthology/P19-1260/)|ACL 2019|`non-Open` 提出了HDE(Heterogeneous Document-Entity)图，图上包含了三类结点：文档结点、候选答案结点以及从文档中抽取的实体mention结点。这些结点的表示通过*co-attention*以及*self-attentive pooling*得到，在这些结点中又定义了7类边，例如：如果候选答案在某文档中出现了至少一次，那么该候选答案结点与文档结点相连。之后使用GNN-based表示更新算法为每个结点更新表示，最后由候选答案结点以及该候选答案中出现的实体mention结点共同为该候选答案进行打分。|   

## 14.与其他领域的结合
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Answering while Summarizing: Multi-task Learning for Multi-hop QA with Evidence Extraction](https://www.aclweb.org/anthology/P19-1225/)|ACL 2019|[TODO]，只用原始问题检索了一次文档|
|2|[Multi-hop Inference for Question-driven Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.547/)|EMNLP 2020|[TODO]|


## [PLAN]
| 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: |
|[Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://www.aclweb.org/anthology/2020.acl-main.412)|ACL 2020|`KGQA`|
|[Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases](https://www.aclweb.org/anthology/2020.acl-main.91)|ACL 2020|`KBQA/KGQA`|
|[Scalable Multi-Hop Relational Reasoning for Knowledge-Aware Question Answering](https://arxiv.org/abs/2005.00646)|EMNLP 2020|`CommonQA` 提出了一种结合GNN与关系路径编码的知识推理与获取知识表示的方式。先抽取知识路径，再利用改进后的GNN在路径上进行信息传播。|
|[Adapting Meta Knowledge Graph Information for Multi-Hop Reasoning over Few-Shot Relations](https://www.aclweb.org/anthology/D19-1334)|ACL 2019|`KGQA`|
|[BAG: Bi-directional Attention Entity Graph Convolutional Network for Multi-hop Reasoning Question Answering](https://www.aclweb.org/anthology/N19-1032)|NAACL 2019|[TODO]| 
|[What’s Missing: A Knowledge Gap Guided Approach for Multi-hop Question Answering](https://www.aclweb.org/anthology/D19-1281)|EMNLP 2019|`OpenBookQA` `CommonQA`|
|[Identifying Supporting Facts for Multi-hop Question Answering with Document Graph Networks](https://www.aclweb.org/anthology/D19-5306)|EMNLP 2019|[TODO]|
|[Quick and (not so) Dirty: Unsupervised Selection of Justification Sentences for Multi-hop Question Answering](https://www.aclweb.org/anthology/D19-1260)|EMNLP 2019|[TODO]|
|[Simple yet Effective Bridge Reasoning for Open-Domain Multi-Hop Question Answering](https://www.aclweb.org/anthology/D19-5806)|EMNLP 2019|[TODO]|
|[Multi-step Entity-centric Information Retrieval for Multi-Hop Question Answering](https://www.aclweb.org/anthology/D19-5816/)|EMNLP 2019|[TODO]|
|[Differentiable Reasoning over a Virtual Knowledge Base](https://openreview.net/forum?id=SJxstlHFPH)|ICLR 2020|[TODO]|
|[Multi-step Retriever-Reader Interaction for Scalable Open-domain Question Answering](https://openreview.net/forum?id=HkfPSh05K7)|ICLR 2020|[TODO]|
|[Question answering as global reasoning over semantic abstractions](https://arxiv.org/abs/1906.03672)|AAAI 2018|[TODO]|

