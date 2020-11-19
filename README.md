
# 多跳阅读理解相关论文
## 1 数据集
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600)|EMNLP 2018|`HotpotQA` 每个QA对有10个（distractor setting）或百万级（full wiki setting）对应的paragraph|
|2|[Constructing Datasets for Multi-hop Reading Comprehension Across Documents](https://arxiv.org/abs/1710.06481)|TACL 2018|`Wikihop`与`MedHop` 每个QA对有数量不等（5~20）个对应的paragraph，还有候选答案集合|
|3|[Reasoning Over Paragraph Effects in Situations](https://arxiv.org/abs/1908.05852)|EMNLP MRQA Workshop 2019|`ROPES` [TODO]|
|4|[DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://arxiv.org/abs/1903.00161)|NAACL 2019|`DROP` 每个QA对都有一个对应的paragraph|

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

## 4 逐渐检索文档（检索到了就要用）
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Answering Complex Open-domain Questions Through Iterative Query Generation](https://arxiv.org/abs/1910.07000)|EMNLP 2019|`Open` GOLDEN模型|
|2|[Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://arxiv.org/abs/1911.10470)|ICLR 2020|`Open` 不断检索文档，在整个wikipedia的文章上建立了一个图，然后一直跳|
|3|[Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval](https://arxiv.org/abs/2009.12756)|arXiv 2020|`Open` 没啥感觉，把检索文档看成序列建模问题然后beam search|
|4|[Cognitive Graph for Multi-Hop Reading Comprehension at Scale](https://doi.org/10.18653/v1/P19-1259)|ACL 2019|`Open` 就CogQA|

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
|4|[Break it down: A question understanding benchmark](https://arxiv.org/abs/2001.11770)|TACL 2020|introduce a Question Decomposition Meaning Representation (QDMR) to explicitly model this process（分解问题）|  
|5|[Complex question decomposition for semantic parsing](https://www.aclweb.org/anthology/P19-1440/)|ACL 2019|虽然不是阅读理解上的，但也是分解问题的工作|
|6|[Learning to Order Sub-questions for Complex Question Answering](https://arxiv.org/abs/1911.04065)|arXiv 2019|利用强化学习去**选择最优的子问题回答顺序**来得到最终答案|
|7|[Generating Followup Questions for Interpretable Multi-hop Question Answering](https://arxiv.org/abs/2002.12344)|arXiv 2020|`non-Open` 也是分解问题的工作，但感觉有点简单？|

## 7 句子级别的推理链
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Multi-hop Question Answering via Reasoning Chains](https://arxiv.org/abs/1910.02610)|arXiv 2019|`non-Open`|

## 8 NMN
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|0|[Neural Module Networks](https://arxiv.org/abs/1511.02799)|CVPR 2016|[TODO]，看名字都能看出来NMN的发源地。每一个模组都可以看成一个可以训练的函数。例如：在图像中定位问题词；识别图像中不同object的关系|
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
|3|[Select, Answer and Explain: Interpretable Multi-hop Reading Comprehension over Multiple Documents](https://arxiv.org/abs/1911.00484)|AAAI 2020|SAE|
|4|[Dynamically fused graph network for multi-hop reasoning](https://arxiv.org/abs/1905.06933)|ACL 2019|DFGN|

## [TODO]
| 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: |
|[Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://www.aclweb.org/anthology/2020.acl-main.412)|ACL 2020|[TODO]|
|[Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases](https://www.aclweb.org/anthology/2020.acl-main.91)|ACL 2020|[TODO]|
|[Unsupervised Alignment-based Iterative Evidence Retrieval for Multi-hop Question Answering](https://www.aclweb.org/anthology/2020.acl-main.414)|ACL 2020|[TODO]|
|[Is Graph Structure Necessary for Multi-hop Question Answering?](https://www.aclweb.org/anthology/2020.emnlp-main.583)|EMNLP 2020|[TODO]|
|[Is Multihop QA in DiRe Condition? Measuring and Reducing Disconnected Reasoning](https://www.aclweb.org/anthology/2020.emnlp-main.712)|EMNLP 2020|[TODO]|
|[Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering](https://arxiv.org/abs/2010.03274)|EMNLP 2020|[TODO]|
|[Scalable Multi-Hop Relational Reasoning for Knowledge-Aware Question Answering](https://arxiv.org/abs/2005.00646)|EMNLP 2020|[TODO]|
|[Question Directed Graph Attention Network for Numerical Reasoning over Text](https://arxiv.org/abs/2009.07448)|EMNLP 2020|[TODO]|
|[Repurposing Entailment for Multi-Hop Question Answering Tasks](https://www.aclweb.org/anthology/N19-1302)|ACL 2019|[TODO]|
|[Exploiting Explicit Paths for Multi-hop Reading Comprehension](https://www.aclweb.org/anthology/P19-1263)|ACL 2019|[TODO]|
|[Adapting Meta Knowledge Graph Information for Multi-Hop Reasoning over Few-Shot Relations](https://www.aclweb.org/anthology/D19-1334)|ACL 2019|[TODO]|
|[Answering while summarizing:Multi-task learning for multi-hop QA with evidence extraction](https://arxiv.org/abs/1905.08511)|ACL 2019|[TODO]|
|[Explore, Propose, and Assemble: An Interpretable Model for Multi-Hop Reading Comprehension](https://arxiv.org/abs/1906.05210)|ACL 2019|[TODO]|
|[Avoiding Reasoning Shortcuts: Adversarial Evaluation, Training, and Model Development for Multi-Hop QA](https://www.aclweb.org/anthology/P19-1262)|ACL 2019|[TODO]|
|[Multi-Hop Paragraph Retrieval for Open-Domain Question Answering](https://doi.org/10.18653/v1/P19-1222)|ACL 2019|[TODO]，训练了一个神经检索模型，利用问题和第一跳的信息来检索第二跳信息|
|[Multi-hop Reading Comprehension across Multiple Documents by Reasoning over Heterogeneous Graphs](https://www.aclweb.org/anthology/P19-1260/)|ACL 2019|[TODO]|
|[Answering while Summarizing: Multi-task Learning for Multi-hop QA with Evidence Extraction](https://www.aclweb.org/anthology/P19-1225/)|ACL 2019|[TODO]，只用原始问题检索了一次文档|
|[Compositional Questions Do Not Necessitate Multi-hop Reasoning](https://arxiv.org/abs/1906.02900)|ACL 2019(short)|[TODO]|
|[BAG: Bi-directional Attention Entity Graph Convolutional Network for Multi-hop Reasoning Question Answering](https://www.aclweb.org/anthology/N19-1032)|NAACL 2019|[TODO]| 
|[Understanding dataset design choices for multi-hop reasoning](https://arxiv.org/abs/1904.12106)|NAACL 2019|[TODO]|
|[TextGraphs 2019 Shared Task on Multi-Hop Inference for Explanation Regeneration](https://www.aclweb.org/anthology/D19-5309)|EMNLP 2019|[TODO]|
|[What’s Missing: A Knowledge Gap Guided Approach for Multi-hop Question Answering](https://www.aclweb.org/anthology/D19-1281)|EMNLP 2019|[TODO]|
|[Do Multi-hop Readers Dream of Reasoning Chains?](https://www.aclweb.org/anthology/D19-5813)|EMNLP 2019|[TODO]|
|[Identifying Supporting Facts for Multi-hop Question Answering with Document Graph Networks](https://www.aclweb.org/anthology/D19-5306)|EMNLP 2019|[TODO]|
|[Quick and (not so) Dirty: Unsupervised Selection of Justification Sentences for Multi-hop Question Answering](https://www.aclweb.org/anthology/D19-1260)|EMNLP 2019|[TODO]|
|[Simple yet Effective Bridge Reasoning for Open-Domain Multi-Hop Question Answering](https://www.aclweb.org/anthology/D19-5806)|EMNLP 2019|[TODO]|
|[Multi-step Entity-centric Information Retrieval for Multi-Hop Question Answering](https://www.aclweb.org/anthology/D19-5816/)|EMNLP 2019|[TODO]|
|[Revealing the Importance of Semantic Retrieval for Machine Reading at Scale](https://www.aclweb.org/anthology/D19-1258/)|EMNLP 2019|[TODO]，只用原始问题检索了一次文档|
|[Numnet: Machine reading comprehension with numerical reasoning](https://arxiv.org/abs/1910.06701)|EMNLP 2019|[TODO]|
|[Differentiable Reasoning over a Virtual Knowledge Base](https://openreview.net/forum?id=SJxstlHFPH)|ICLR 2020|[TODO]|
|[Multi-step Retriever-Reader Interaction for Scalable Open-domain Question Answering](https://openreview.net/forum?id=HkfPSh05K7)|ICLR 2020|[TODO]|
|[QASC:A dataset for question answering via sentence composition](https://arxiv.org/abs/1910.11473)|AAAI 2020|[TODO]|
|[Multi-paragraph reasoning with knowledge-enhanced graph neural network](https://arxiv.org/abs/1911.02170v1)|arXiv 2019|[TODO]，只用原始问题检索了一次文档|
|[Question answering as global reasoning over semantic abstractions](https://arxiv.org/abs/1906.03672)|AAAI 2018|[TODO]|
|[Reinforced ranker-reader for open-domain question answering](https://arxiv.org/abs/1709.00023)|AAAI 2018|[TODO]|


