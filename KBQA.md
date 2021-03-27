# KBQA相关论文

## 1. 使用多个知识库

> 知识库中通常有KG，但也有不同形式的知识源，例如非结构的文本知识。

| 序号 |                             论文                             |      发表会议       |                             备注                             |
| :--: | :----------------------------------------------------------: | :-----------------: | :----------------------------------------------------------: |
|  1   | [Natural Language QA Approaches using Reasoning with External Knowledge](https://arxiv.org/abs/2003.03446) |     arXiv 2020      | 是一篇Survey，总结了现有的利用外部知识完成QA的方法，包括数据集的整理，常用的外部知识整理还有使用外部知识的一些常用方法。知识方面，无结构知识：`Wikipedia Corpus`、`TorontoBookCorpus`、`ARC Corpus`、`WikiHow`、`RocStories`、`Story Cloze`等，结构化知识：`Yago`、`NELL`、`DBPedia`、`ConceptNet`、`WordNet`。对于无结构知识，可以考虑利用记忆网络来存储知识，对于结构化知识可以考虑利用GNN或Tree-based LSTM来存储知识。 |
|  2   | [Connecting the Dots: A Knowledgeable Path Generator for Commonsense Question Answering](https://arxiv.org/abs/2005.00691) | EMNLP 2020 Findings | 图2部分介绍了KG增强的QA模型框架，本文主要工作在于，已有的常识库，例如`ConceptNet`比较稀疏，可能仍然不能够填充从问题到正确答案的推理链，所以作者干脆直接在问题和答案中生成一条推理路径，这样的推理路径可能是KG中所没有的，以此来解决这个问题。其中生成推理路径的数据集是在KG上通过随机游走的方式得到的，并利用GPT2训练了一个路径生成模型。在`OpenBookQA`上达到了80.05(±0.68) |
|  3   | [Improving Commonsense Question Answering by Graph-based Iterative Retrieval over Multiple Knowledge Sources](https://arxiv.org/abs/2011.02705) |     COLING 2020     | 在多个知识库（`ConceptNet`、`Wikipedia`以及`Cambridge Dictionary`）中进行常识的捕获，基于图的迭代式检索，根据初始化结点（利用问题和候选答案中的concept作初始化结点）以及缩小范围化的关系在`ConceptNet`上迭代式得产生结构化图，在`Wikipedia`上进行相似度计算并保留top10文本（句子级），其中问题和候选答案中的concept均会在`Cambridge Dictionary`中查找相应的解释并拼接在后面。消融实验证明了这三个知识库都起到了作用。在`CommonseQA`上进行了验证。 |
|  4   | [Multi-Modal Answer Validation for Knowledge-Based VQA](https://arxiv.org/abs/2103.12248) |     arXiv 2021      | 数据源有`ConceptNet`、`Wikipedia`、`Google images`，先利用输入的图像和问题和现有SOTA模型（`ViLBERT`）产生候选答案，再利用候选答案进行知识检索，将检索到的知识又用于验证不同的候选答案以得出最终的预测结果。 |





