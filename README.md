
# 多跳阅读理解相关论文

## 目录
- [1.数据集](#1数据集)   
- [2.改进传统单步阅读理解方法](#2改进传统单步阅读理解方法)    
- [3.利用GNN进行多跳推理](#3利用gnn进行多跳推理)   
- [4.迭代式检索文档](#4迭代式检索文档)    
- [5.推理链](#5推理链)    
- [6.分解问题相关](#6分解问题相关)    
- [7.神经模块网络NMN](#7neural-module-networks)     
- [8.PTM for MhRC](#8针对于多跳mrc问题的ptm改进)    
- [9.数据增强](#9数据增强)    
- [10.本质方面的探索](#10在本质方面的探索)    
- [11.数学推理能力](#11numerical-reasoning)     
- [12.与其他任务相关联](#12与其他任务相关联)    
- [13.OBQA](#13obqa)    
- [14.信息检索](#14信息检索包括rerank)
- [15.可解释性研究](#15可解释性研究)
- [Leaderboard](#aleaderboard-of-hotpotqa)    

## 1.数据集
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600)|EMNLP 2018|`HotpotQA` **抽取式** 每个QA对有10个（distractor setting）或百万级（full wiki setting）对应的paragraph|
|2|[Constructing Datasets for Multi-hop Reading Comprehension Across Documents](https://arxiv.org/abs/1710.06481)|TACL 2018|`Wikihop`与`MedHop` **多选式** 每个QA对有数量不等（平均14，最多64）个对应的paragraph，还有候选答案集合|
|3|[Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering](https://www.aclweb.org/anthology/D18-1260/)|EMNLP 2018|`OpenBookQA` `CommonQA` **多选式** 包含了两部分5957个多选问题（每题四个选项），1326个初级科学事实。科学事实通常不能够直接回答问题，对于一个问题，应当先检索相关的科学事实，然后加上常识（不在本数据集中提供，通常用`ConceptNet`）才能得到答案。需要多跳推理，也需要常识推理。|
|4|[The NarrativeQA Reading Comprehension Challenge](https://arxiv.org/abs/1712.07040)|TACL 2018|`NarrativeQA` 包含来自于书本和电影剧本的1567个完整故事，数据集划分为不重叠的训练、验证和测试三个部分，共有 46765个问题答案对|
|5|[Reasoning Over Paragraph Effects in Situations](https://arxiv.org/abs/1908.05852)|EMNLP MRQA Workshop 2019|`ROPES` **抽取式** 每个QA对对应两个paragraph，一个称为*background*是说明文，另一个是*situation*由众包工人创建，回答问题需要把*background*中的知识应用与*situation*才可|
|6|[DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://arxiv.org/abs/1903.00161)|NAACL 2019|`DROP` **答案不一定出现在原文中 需要通过计算、计数等操作得到** 每个QA对都有一个对应的paragraph|  
|7|[DREAM: A Challenge Data Set and Models for Dialogue-Based Reading Comprehension](https://www.aclweb.org/anthology/Q19-1014/)|TACL 2019|`Dream` **多选式** 10197个问题，并在英语考试题中收集6444个多轮多方的对话数据。多个选择题对应一个多轮多方对话数据，只有一个候选答案正确|
|8|[Looking Beyond the Surface: A Challenge Set for Reading Comprehension over Multiple Sentences](https://www.aclweb.org/anthology/N18-1023/)|NAACL 2018|`MultiRC` **多选式** 每个实例包含一个问题，多个sentences以及答案候选集，根据sentences去选择正确的答案，且可能有多个候选答案正确，每个实例的候选答案数量不一定一致|
|9|[R4C: A Benchmark for Evaluating RC Systems to Get the Right Answer for the Right Reason](https://arxiv.org/abs/1910.04601)|ACL 2020|`R4C` **给出半结构化的derivation** 对`HotpotQA`数据集进行了标注，一共标注了约5K个实例，每个实例被标注了三个*derivation*。`R4C`觉得*Supporting Sentence*是一个非常粗粒度的概念，因为一句*Supporting Sentence*里面可能有些内容并不是推理答案所必须的，而另一部分是必须的，因此作者标注了更加细粒度的*derivation*，这是一种半结构化的形式来表达推理信息，比*Supporting Sentences*更加有挑战性，也对模型的可解释性要求更高。|
|10|[QASC:A dataset for question answering via sentence composition](https://arxiv.org/abs/1910.11473)|AAAI 2020|`QASC` **多选式** 也是基于多个句子间的推理从候选答案中选出正确答案的数据集。含有9980个八项选择题，每个问题都被标注了两个fact sentences用来推理出最终的答案。还提供了一个包含了17M个句子的语料库，所有的fact sentences都在里面|
|11|[Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering](https://arxiv.org/abs/2010.03274)|EMNLP 2020|`eQASC`、`eQASC-perturbed`以及`eOBQA` `Textual entailment` **多选式** 这个工作其实是在研究多跳推理问题的可解释性。在`QASC`数据集的基础上，针对于每个问题又标注了多个推理链（有效或无效都有）构成了`eQASC`，接着又将推理链模板化（使推理链更加通用）构成了`eQASC-perturbed`数据集，最后为了进行out-domain test，也基于`OpenBookQA`数据集标注了推理链形成了`eOBQA`并进行测试|  

## 2.改进传统单步阅读理解方法
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[A Simple Yet Strong Pipeline for HotpotQA](https://arxiv.org/abs/2004.06753)|arXiv 2020|一种非常简单的方法但达到了非常不错的效果，值得思考|



## 3.利用GNN进行多跳推理
>一般使用GNN的框架是：段落选取→编码→建图→利用GNN-based算法更新表示→答案预测。这类方法使用强大的图结构作为支撑，只经历了一次段落选取步骤，优点是对第一步检索的要求不高，缺点是可解释性非常差。

| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1*|[Multi-hop Reading Comprehension across Multiple Documents by Reasoning over Heterogeneous Graphs](https://www.aclweb.org/anthology/P19-1260/)|ACL 2019|`non-Open` 提出了HDE(Heterogeneous Document-Entity)图，图上包含了三类结点：文档结点、候选答案结点以及从文档中抽取的实体mention结点。这些结点的表示通过*co-attention*以及*self-attentive pooling*得到，在这些结点中又定义了7类边，例如：如果候选答案在某文档中出现了至少一次，那么该候选答案结点与文档结点相连。之后使用GNN-based表示更新算法为每个结点更新表示，最后由候选答案结点以及该候选答案中出现的实体mention结点共同为该候选答案进行打分。|  
|2*|[Dynamically fused graph network for multi-hop reasoning](https://arxiv.org/abs/1905.06933)|ACL 2019|`non-Open` DFGN|   
|3*|[Identifying Supporting Facts for Multi-hop Question Answering with Document Graph Networks](https://www.aclweb.org/anthology/D19-5306)|EMNLP 2019|`HotpotQA` 构建了DGN(Document Graph Network)并在上面传递信息以及识别supporting fact。Document Grpah包含两类结点：段落结点以及句子结点。以及两类边：如果句子存在于某文档中，则该句子结点与文档结点相连。如果一个文档中的实体被另一个文档所引用，则这两个文档之间相连。（注意没有句子与句子之间的边，因为会大大增加模型的复杂度且带来不了显著提升）。在构建Document Graph之后有一个过滤的步骤，根据问题，对所有段落中的每句话去计算其与问题的相似度。最终选取topk个**句子**。这些句子与文档之间构成原先Document Graph中的一个子图。然后结点的表示与问题的表示进行*Bi-Linear Attention*与*Self-Attention*得到结点的初始化表示，之后利用GNN系列算法更新表示，最终对句子结点进行supporting fact的预测。|
|4*|[BAG: Bi-directional Attention Entity Graph Convolutional Network for Multi-hop Reasoning Question Answering](https://www.aclweb.org/anthology/N19-1032)|NAACL 2019|`non-Open` `Wikihop` 建立了一个比较简单的图，图上的结点都是实体结点，共有两类边：不同段落间相同实体之间的边以及同一段落任意两个实体结点之间也存在一条边。之后使用`Glove`、`ELMo`、`NER`以及`POS`来做特征的初始化，然后使用GCN去更新表示，最后对每一个实体结点进行其为答案的概率预测。| 
|5*|[Multi-paragraph reasoning with knowledge-enhanced graph neural network](https://arxiv.org/abs/1911.02170v1)|arXiv 2019|`Open`&`no-Open` 主要贡献在于，从paragraphs上建立了一个KG，然后利用GNN更新结点表示，结点表示更新之后返回来更新paragraphs的表示，最后进行答案预测|
|6*|[Hierarchical Graph Network for Multi-hop Question Answering](https://arxiv.org/abs/1911.03631)|EMNLP 2020|构建了一个异质图包含四类结点和七类边，利用GNN来进行多跳推理|
|7|[Is Graph Structure Necessary for Multi-hop Question Answering?](https://www.aclweb.org/anthology/2020.emnlp-main.583)|EMNLP 2020|`non-Open` 改进了`DFGN`模型，探索了图结构在多跳QA中是否必要，结论是：如果PTM是特征提取器的话就重要，如果微调PTM的话其实图结构不是很重要| 
|8*|[Select, Answer and Explain: Interpretable Multi-hop Reading Comprehension over Multiple Documents](https://arxiv.org/abs/1911.00484)|AAAI 2020|`non-Open` SAE|

(*代表仅属于本分类下的工作)

## 4.迭代式检索文档
>多次检索文档，可解释性优于GNN-based models。其中PullNet也使用到了GNN算法。这里的分类认为：多次检索文档，需要不断更新query然后以此来不断地进行迭代式检测。注意这种方法包括两大流派：生成推理链以及分解问题，这两个流派将在之后阐述，先介绍迭代式检索文档中不属于这两个流派的工作。

| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Explore, Propose, and Assemble: An Interpretable Model for Multi-Hop Reading Comprehension](https://arxiv.org/abs/1906.05210)|ACL 2019|分为三部分，三部分联合优化。Document Explore：一个级联的memory network迭代式地选取相关文档；Answer Proposer：对于推理树上的每一个从跟到结点的推理路径提出一个proposed答案；Evidence Assembler：从每一条推理路径上提取包含proposed答案的关键句，并将这些关键句结合起来以预测最终的答案。|
|2|[Multi-Hop Paragraph Retrieval for Open-Domain Question Answering](https://doi.org/10.18653/v1/P19-1222)|ACL 2019|`Open` 主要在开放式的多跳QA中做检索方面的迭代式探索尝试，提出了MUPPET(MUlti-hoP Paragraph rETrieval)，共有两个模块组成：**段落和问题编码器**以及**段落阅读器**。编码器负责获取段落的表示以及将问题编码成搜索向量，阅读器通过搜索向量基于最大化内积的方法检索相关度高的段落。在每次迭代中，搜索向量会由前几步检索得到的文档的表示所影响，所以每次迭代中的搜索向量不一样，因此才能迭代式地检索到不同的段落。|
|3|[Revealing the Importance of Semantic Retrieval for Machine Reading at Scale](https://www.aclweb.org/anthology/D19-1258/)|EMNLP 2019|`Open` 只用原始问题检索了一次文档，但在这一次检索中，先利用了基于TFIDF的方法筛选一遍，然后对于每个paragraph又通过语义相似度计算再筛选一遍，然后将还剩下的paragraph分解成句子，再在句子级别利用语义相似度计算得到最终所有的支撑句。最终利用支撑句和问题进行答案预测。虽然只检索了一次，但其实在第一次的基础上进行了后续的多次筛选与更加细粒度的检索。| 
|4|[Unsupervised Alignment-based Iterative Evidence Retrieval for Multi-hop Question Answering](https://www.aclweb.org/anthology/2020.acl-main.414)|ACL 2020|`Open`&`non-Open` 在`MultiRC`以及`QASC`上的工作，做与问题相关的支撑句的检索。采用无监督的对其方法。每次检索会修改query，修改的规则主要是当前检索到的句子没有包含query中的那些terms，检索也使用了基本的基于词向量语义相似度的方式，属于无监督模型。但达到了很好的效果。|  
|5|[Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval](https://arxiv.org/abs/2009.12756)|arXiv 2020|`Open` 没啥感觉，把检索文档看成序列建模问题然后beam search|
|6|[DDRQA: Dynamic Document Reranking for Open-domain Multi-hop Question Answering](https://arxiv.org/abs/2009.07465)|arXiv 2020|`Open` 动态检索文档，只有最终确定的才进入到下一步|  

## 5.推理链
> 最终形成一条推理链。

| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Exploiting Explicit Paths for Multi-hop Reading Comprehension](https://www.aclweb.org/anthology/P19-1263)|ACL 2019|`Wikihop`与`OpenBookQA` `Open`&`non-Open` 这篇工作主要的贡献在于多跳阅读理解的可解释性，为了在文本数据上达到多跳的效果，会有两种方法：GNN或者路径抽取，GNN可解释性非常差，因为它是隐式地完成信息传递。而路径抽取的方法解释性强，但如果跳数增多的话会有语义漂移问题。不过`Wikihop`或者`OpenBookQA`数据集都是两跳，所以好像不严重？然后作者就通过在问题中提取头实体，在候选答案中提取尾实体，然后在**全部候选文档**中抽取多个推理链，接着对推理链的实体做表示初始化然后隐式提取关系，再通过关系计算路径的表示。最后会对路径进行打分，然后根据分数得到最终的答案概率分布。我个人觉得这篇工作利用两个实体的表示去直接计算他们的关系表示这里有点粗糙了，因为两个实体之间可能存在着不止一种关系，而利用作者所给的式子则无法对这种多样的关系进行学习。|
|2|[Cognitive Graph for Multi-Hop Reading Comprehension at Scale](https://doi.org/10.18653/v1/P19-1259)|ACL 2019|`Open` 就CogQA|
|3|[Multi-step Entity-centric Information Retrieval for Multi-Hop Question Answering](https://www.aclweb.org/anthology/D19-5816/)|EMNLP 2019|也是在检索文档上做了探索。从第一次检索到的文档上分析并利用其中的实体信息进行后续的多跳分析。|
|4|[Simple yet Effective Bridge Reasoning for Open-Domain Multi-Hop Question Answering](https://www.aclweb.org/anthology/D19-5806)|EMNLP 2019|`Open`&`non-Open` 在短视检索上进行探索的工作，`Bridge Reasoner`输入的是开始文档（通过IR得到），然后进行span预测，预测出桥梁实体，进而再产生候选答案段落，`Passage Reader`提取最终答案。|  
|5|[PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text](https://arxiv.org/abs/1904.09537)|EMNLP 2019|`MetaQA`、`WebQuestionsSP`以及`Complex WebQ` `Open` 从两种实体源中检索知识资源：文本语料库与知识库。作者定义了问题子图（question subgraph）这一概念，它的作用是包含足够多的与问题相关的信息然后以此来回答一个问题，该图是迭代生成的，且分为三类结点：①实体结点（存在于KB中）②文本结点（通常情况下是文本语料库中某个实体的mention）③事实结点（KB中的事实三元组）。一开始先仅利用问题中的信息来初始化问题子图，之后进行T轮迭代扩充，每次扩充会选取问题子图中的部分结点，对每一个选取到的结点，为其检索其相关的（1）文档（2）事实，对于（1）还会进一步利用实体链接模型提取其中的实体mention，对于（2）来说会提取三元组中的头尾实体。这样的构建直到问题子图可以回答问题为止，之后再进行答案预测。|
|6|[Answering Complex Open-domain Questions Through Iterative Query Generation](https://arxiv.org/abs/1910.07000)|EMNLP 2019|`Open` GOLDEN模型|
|7|[Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://arxiv.org/abs/1911.10470)|ICLR 2020|`Open` 不断检索文档，在整个wikipedia的文章上建立了一个图，然后一直跳|


## 6.分解问题相关
> 可解释性也很强，但是分解问题需要额外的工作，如何训练模型合适地分解问题是一个challenge。

| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[The Web as a Knowledge-Base for Answering Complex Questions](https://www.aclweb.org/anthology/N18-1059/)|NAACL 2018|大概看了一下，觉得论述在英文方面表述的很奇怪，在민세원的论文中（该表格的下一项论文）本篇作为引文举出，被阐述了主要区别。|
|2|[Multi-hop Reading Comprehension through Question Decomposition and Rescoring](https://arxiv.org/abs/1906.02916)|ACL 2019|[민세원女神](https://shmsw25.github.io/)的paper，不多说了，膜就完事了。|
|3|[Complex question decomposition for semantic parsing](https://www.aclweb.org/anthology/P19-1440/)|ACL 2019|虽然不是阅读理解上的，但也是分解问题的工作|
|4|[Learning to Order Sub-questions for Complex Question Answering](https://arxiv.org/abs/1911.04065)|arXiv 2019|利用强化学习去**选择最优的子问题回答顺序**来得到最终答案|
|5|[Unsupervised Question Decomposition for Question Answering](https://arxiv.org/abs/2002.09758)|EMNLP 2020|定义了一种无监督的方法来分解出子问题| 
|6|[Break it down: A question understanding benchmark](https://arxiv.org/abs/2001.11770)|TACL 2020|从10个复杂问题的QA数据集中提取问题并标注分解，一共定义了13个基本分解操作，以及3个高级分解。提出了`BREAK`数据集，包含了83978个复杂问题及其对应的分解。又提出了`BreakRC`模型，将考虑问题分解应用于`HotpotQA`(fullwiki setting)中|  
|7|[Generating Followup Questions for Interpretable Multi-hop Question Answering](https://arxiv.org/abs/2002.12344)|arXiv 2020|`non-Open` 也是分解问题的工作，但感觉有点简单？|


## 7.Neural Module Networks
>最先在VQA中使用，后也引用至多跳阅读理解领域，肥肠复杂，因为要定义好肥肠多的模组功能以及每个模组分别在什么场景下出现。

| 序号 | 论文 | 发表会议 | 备注 | repo |
| :---: | :---: | :---: | :---: |:---: |
|1|[Deep Compositional Question Answering with Neural Module Networks](https://arxiv.org/abs/1511.02799)|CVPR 2016|NMN鼻祖，在VQA中定义了多个模组来完成不同的操作：`Attention`定位图像中某Obj的位置 `Re-attention`在att map上进行位置迁移等等||
|2|[Learning to Compose Neural Networks for Question Answering](https://arxiv.org/abs/1601.01705)|NAACL 2016|||
|3|[Learning to Reason: End-to-End Module Networks for Visual Question Answering](https://arxiv.org/abs/1704.05526)|ICCV 2017|`N2NMN`||
|4|[Inferring and Executing Programs for Visual Reasoning](https://arxiv.org/abs/1705.03633)|ICCV 2017|`PG+EE`||
|5|[Using Syntax to Ground Referring Expressions in Natural Images](https://arxiv.org/abs/1805.10547)|AAAI 2018|`GroundNet`||
|6|[Transparency by Design: Closing the Gap Between Performance and Interpretability in Visual Reasoning](https://arxiv.org/abs/1803.05268)|CVPR 2018|`TbD`||
|7|[Explainable neural computation via stack neural module networks](https://arxiv.org/abs/1807.08556)|ECCV 2018| use continuous and soft layout prediction and maintain a differentiable stack-based data structure to store the predicted modules’ output. This approach to optimize the modular network is shown to be superior to using a Reinforcement Learning approach which makes hard module decisions.|[repo](https://github.com/ronghanghu/snmn)|
|8|[Neural Compositional Denotational Semantics for Question Answering](https://arxiv.org/abs/1808.09942)|EMNLP 2018 | | | 
|9|[Routing Networks: Adaptive Selection of Non-Linear Functions for Multi-Task Learning](https://openreview.net/forum?id=ry8dvM-R-)|ICLR 2018|||
|10|[Modular Networks: Learning to Decompose Neural Computation](https://arxiv.org/abs/1811.05249)|NIPS 2018|||
|11|[Self-assembling modular networks for interpretable multi-hop reasoning](https://arxiv.org/abs/1909.05803)|EMNLP 2019|`non-Open` 在`HotpotQA`上的工作，入栈出栈，三个模组`Find`、`Relocate`以及`Compare`|[repo](https://github.com/jiangycTarheel/NMN-MultiHopQA)|
|12|[Explore, Propose, and Assemble: An Interpretable Model for Multi-Hop Reading Comprehension](https://arxiv.org/abs/1906.05210)| ACL 2019 |分为三部分，三部分联合优化。Document Explore：一个级联的memory network迭代式地选取相关文档；Answer Proposer：对于推理树上的每一个从跟到结点的推理路径提出一个proposed答案；Evidence Assembler：从每一条推理路径上提取包含proposed答案的关键句，并将这些关键句结合起来以预测最终的答案。| |
|13|[Multi-Step Inference for Reasoning Over Paragraphs](https://arxiv.org/abs/2004.02995)|EMNLP 2020|`non-Open` 感觉本文，在NMN上的创新度不及其余文章，然后选取的数据集也不是主流的`HotpotQA`，而是`ROPES`||
|14|[Neural module networks for reasoning over text](https://arxiv.org/abs/1912.04971)|ICLR 2020|`non-Open` NMN在`DROP`上的工作，设计了10个模组||
|15|[Text Modular Networks: Learning to Decompose Tasks in the Language of Existing Models](https://arxiv.org/abs/2009.00751)|arXiv 2020|`non-Open` 在`DROP`上的工作，也可以用于`HotpotQA`，两个模组`next-question generator`与`QA model`||





## 8.针对于多跳MRC问题的PTM改进
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Transformer-XH: Multi-hop question answering with eXtra Hop attention](https://openreview.net/forum?id=r1eIiCNYwS)|ICLR 2020|让transformer在图结构上也进行学习，评分686|
|2|[CogLTX: Applying BERT to Long Texts](http://keg.cs.tsinghua.edu.cn/jietang/publications/NIPS20-Ding-et-al-CogLTX.pdf)|NIPS 2020|在长文本上应用BERT，并且使得BERT有一定的推理能力，重点句迭代选取最后保留对于当前任务最相关的句子，进行下一轮计算。|

## 9.数据增强
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Avoiding Reasoning Shortcuts: Adversarial Evaluation, Training, and Model Development for Multi-Hop QA](https://www.aclweb.org/anthology/P19-1262)|ACL 2019|`non-Open` 这篇文章也揭示了`HotpotQA`数据集有些问题不用推理也能回答，他们设置了攻击实验，发现在对抗数据集（通过在答案区间以及支撑文档的标题上进行短语级别的干扰得到，这样模型如果还是是用推理捷径的话将会得到多个可能的答案，从而影响模型的表现）上现有的SOTA模型表现都会下降很多，除此之外他们设计了一个控制单元来指导模型进行多跳推理。|  
|2*|[Low-Resource Generation of Multi-hop Reasoning Questions](https://www.aclweb.org/anthology/2020.acl-main.601/)|ACL 2020|QG工作|
|3*|[Logic-Guided Data Augmentation and Regularization for Consistent Question Answering](https://arxiv.org/abs/2004.10157)|ACL 2020|讨论核心点是对比问题的数据增强，基于对称一致性和传递一致性来增强训练样本|
|4*|[Generating Multi-hop Reasoning Questions to Improve Machine Reading Comprehension](https://dl.acm.org/doi/pdf/10.1145/3366423.3380114)|WWW 2020|QG工作|
|5*|[Asking Complex Questions with Multi-hop Answer-focused Reasoning](https://arxiv.org/abs/2009.07402)|arXiv 2020|QG工作|

(*代表仅属于本分类下的工作)


## 10.在本质方面的探索
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Compositional Questions Do Not Necessitate Multi-hop Reasoning](https://arxiv.org/abs/1906.02900)|ACL 2019(short)|作者发现`hotpotQA`中许多多跳问题都能够被单跳模型回答正确，于是展开了分析|
|2|[Avoiding Reasoning Shortcuts: Adversarial Evaluation, Training, and Model Development for Multi-Hop QA](https://www.aclweb.org/anthology/P19-1262)|ACL 2019|`non-Open` 这篇文章也揭示了`HotpotQA`数据集有些问题不用推理也能回答，他们设置了攻击实验，发现在对抗数据集（通过在答案区间以及支撑文档的标题上进行短语级别的干扰得到，这样模型如果还是是用推理捷径的话将会得到多个可能的答案，从而影响模型的表现）上现有的SOTA模型表现都会下降很多，除此之外他们设计了一个控制单元来指导模型进行多跳推理。|
|3|[Do Multi-hop Readers Dream of Reasoning Chains?](https://www.aclweb.org/anthology/D19-5813)|EMNLP 2019|个人觉得这篇其实没有NAACL 2019那篇那么精彩，但也是不错的，作者也是做了一些分析实验，然后得出结论：一些单跳模型在面对`HotpotQA`问题也能做的还不错，然后推理链其实是有用的，但现在没有被挖掘的很深入，未来可能会有大用处。|
|4|[Understanding dataset design choices for multi-hop reasoning](https://arxiv.org/abs/1904.12106)|NAACL 2019|非常棒的一个工作，精彩精彩。在`HotpotQA`和`Wikihop`数据集上展开研究，发现多跳阅读理解数据集单跳也能回答对，`Wikihop`不看文章也能答对，Span式的多跳数据集优于多选式的，即使Span式的数据集中有很多问题单跳也能答对|
|5|[Is Graph Structure Necessary for Multi-hop Question Answering?](https://www.aclweb.org/anthology/2020.emnlp-main.583)|EMNLP 2020|`non-Open` 改进了`DFGN`模型，探索了图结构在多跳QA中是否必要，结论是：如果PTM是特征提取器的话就重要，如果微调PTM的话其实图结构不是很重要|
|6|[Is Multihop QA in DiRe Condition? Measuring and Reducing Disconnected Reasoning](https://www.aclweb.org/anthology/2020.emnlp-main.712)|EMNLP 2020|`HotpotQA`上的一个研究，也是探索了当前多跳模型到底有没有推理能力。作者将推理分为连贯推理（connected resoning）与不连贯推理。连贯推理是我们希望能够赋予模型的能力，也就是在多个文档中进行信息间的交互以此得到答案。而不连贯推理则是不交互信息就得到答案。作者设计了一些实验去探索模型通过不连贯推理所能达到的分数以此来说明现有的模型可能并没有达到我们想要赋予他们多跳推理能力的初衷。除此之外，作者还设计了一种方案将数据集进行转换以此让模型更难cheat。该工作有着非常高的借鉴价值。对于模型设计者来说，可以帮助我们鉴别自己设计的模型到底有没有连贯推理，对于数据集制造者来说，可以让其明白自己的数据集容不容易被cheat|
|7|[Do Multi-Hop Question Answering Systems Know How to Answer the Single-Hop Sub-Questions?](https://arxiv.org/abs/2002.09919)|arXiv 2020|研究问题：多跳QA系统能否回答单跳子问题；除此之外引用了`DecompRC`中划分子问题的方法|


## 11.Numerical Reasoning
>在DROP数据集上的一些研究，主要要求模型具有计数，数学运算等能力。

| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|0|[DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://arxiv.org/abs/1903.00161)|NAACL 2019|`DROP` **答案不一定出现在原文中 需要通过计算、计数等操作得到** 每个QA对都有一个对应的paragraph| 
|1*|[Numnet: Machine reading comprehension with numerical reasoning](https://arxiv.org/abs/1910.06701)|EMNLP 2019|`DROP` 从context中构建一个异质有向图出来，结点都代表context中出现的数字，分为两类：在问题中出现的数字与在段落中出现的数字。边也分为两类，根据数字间的大小关系进行连接，两类边刚好互补。|
|2*|[A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning](https://www.aclweb.org/anthology/D19-1170/)|EMNLP 2019|[TODO] 没`NumNet`好|
|3*|[Giving BERT a Calculator: Finding Operations and Arguments with Reading Comprehension](https://www.aclweb.org/anthology/D19-1609/)|EMNLP 2019|[TODO] 没`NumNet`好|
|4*|[Injecting Numerical Reasoning Skills into Language Models](https://arxiv.org/abs/2004.04487)|ACL 2020|[TODO] 没`NumNet`好|
|5*|[Question Directed Graph Attention Network for Numerical Reasoning over Text](https://arxiv.org/abs/2009.07448)|EMNLP 2020|`no-Open` 在`DROP`上的工作，所以自然不用多次检索，每个QA对给定了一个paragraph。建立了一个异质图，结点有：数字（八大类型）、实体，边有：相同类型的数字之间的边以及数字和实体在一句话内共现的边。主要的idea就是在图上引入了数字的类型以及与数字相关的实体，是针对于`NumNet`的改进|   
|6|[Neural module networks for reasoning over text](https://arxiv.org/abs/1912.04971)|ICLR 2020|`non-Open` NMN在`DROP`上的工作，设计了10个模组|
|7*|[Neural Symbolic Reader: Scalable Integration of Distributed and Symbolic Representations for Reading Comprehension](https://openreview.net/forum?id=ryxjnREFwH)|ICIR 2020|[TODO] 没`NumNet`好|
|8|[Text Modular Networks: Learning to Decompose Tasks in the Language of Existing Models](https://arxiv.org/abs/2009.00751)|arXiv 2020|`non-Open` 在`DROP`上的工作，也可以用于`HotpotQA`，两个模组`next-question generator`与`QA model`|  

(*代表仅属于本分类下的工作)


## 12.与其他任务相关联  
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Answering while Summarizing: Multi-task Learning for Multi-hop QA with Evidence Extraction](https://www.aclweb.org/anthology/P19-1225/)|ACL 2019|`多跳阅读理解`、`抽取式文本摘要`以及`文本蕴含` 可解释性研究，受抽取式文本摘要的灵感，提出了QFE(Query Focused Extractor)模型。创新度不是很高，和原始的HotpotQA baseline挺像的。只不过多了一个支撑句预测层，该层就是他设计的QFE模型|
|2|[Repurposing Entailment for Multi-Hop Question Answering Tasks](https://www.aclweb.org/anthology/N19-1302)|NAACL 2019|`Textual entailment` `OpenBookQA`与`MultiRC` 使用文本蕴含模型来完成多跳推理问答，模型分为两部分：相关句提取与信息聚合。相关句提取对每个候选句计算其蕴含假设（由答案和问题构成）的概率，这些概率表示每句的重要程度。信息聚合则利用上一步计算的概率为不同的句子聚合表示，最后再通过一个文本蕴含模型得到最终的蕴含概率。|   
|3|[A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/abs/1910.11476)|ACL 2020|[TODO]|
|4|[Multi-hop Inference for Question-driven Summarization](https://www.aclweb.org/anthology/2020.emnlp-main.547/)|EMNLP 2020|[TODO]|
|5|[A Joint Training Dual-MRC Framework for Aspect Based Sentiment Analysis](https://arxiv.org/abs/2101.00816)|AAAI 2021|[TODO]|


## 13.OBQA
> 这个模块主要总结一下在OpenBookQA上的已有工作

| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|0|[Natural Language QA Approaches using Reasoning with External Knowledge](https://arxiv.org/abs/2003.03446)|arXiv 2020| 是一篇Survey，总结了现有的利用外部知识完成QA的方法，包括数据集的整理，常用的外部知识整理还有使用外部知识的一些常用方法。知识方面，无结构知识：`Wikipedia Corpus`、`TorontoBookCorpus`、`ARC Corpus`、`WikiHow`、`RocStories`、`Story Cloze`等，结构化知识：`Yago`、`NELL`、`DBPedia`、`ConceptNet`、`WordNet`。对于无结构知识，可以考虑利用记忆网络来存储知识，对于结构化知识可以考虑利用GNN或Tree-based LSTM来存储知识。 |
|1| [Careful Selection of Knowledge to Solve Open Book Question Answering](https://arxiv.org/abs/1907.10738)| ACL 2019|一个比较复杂的工作，先基于问题产生hypothesis，再基于hypo检索相关fact，利用fact以及hypo之间的差异去检索missing knowledge。接着根据fact以及missing knowledge去综合为每个候选选项打分，最终分数高的为答案选项。可以看出该篇工作设计的流程较为复杂，达到了6个modular，取得的效果也一般，在OBQA测试集上达到了72的acc|
|2|[Enhancing Pre-Trained Language Representations with Rich Knowledge for Machine Reading Comprehension](https://pdfs.semanticscholar.org/e704/6bf945ad6326537a1ac78a96fd2f45acc900.pdf?_ga=2.8782581.1586501600.1609492089-1005375774.1592035873) | ACL 2019 | 只有related work部分有用，总结的挺好的 |
|3| [Improving Question Answering with External Knowledge](https://arxiv.org/abs/1902.00993) | MRQA@EMNLP 2019 | 本文主要利用了无结构化的外部知识（wikipedia）来提升OBQA的效果，设定了不同的方法来吸收open-domain(wiki)与in-domain(其余数据集)中的sentences与corpus中检索得到的sentences，一起输入至模型，并预测答案，在OBQA测试集上达到了68的acc |
|4|[KagNet: Knowledge-Aware Graph Networks for Commonsense Reasoning](https://arxiv.org/abs/1909.02151)| EMNLP 2019 |一篇非常有借鉴意义的工作，虽然是在`CommonsenseQA`上的工作，但其提出的框架已经被利用在不同的常识QA中了。首先根据QA对建立schema graph，再利用GCN更新图中结点的表示，之后利用LSTM去编码不同的路径信息，再使用attention机制综合所有路径的信息得到graph vector，最终使用graph vector来计算QA对的成立概率。|
|5|[What’s Missing: A Knowledge Gap Guided Approach for Multi-hop Question Answering](https://www.aclweb.org/anthology/D19-1281)|EMNLP 2019|idea是：可以检测到的fact与候选答案之间存在gap。对于一个问题，检索到fact之后，对fact进行key span预测，基于key span会去`ConceptNet`以及`ARC Corpus`中检索缺失的知识，再进行QA。标注了key span以及gap类型信息，在训练的过程当中进行多任务学习：gap类型预测以及答案预测。最终在`OBQA`测试集上达到了64.41的acc| 
|6| [Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering](https://arxiv.org/abs/2010.03274) | EMNLP 2020 | 在`OBQA`部分数据的基础上人工添加了推理链形成`eOBQA`数据集 |
|7|[Scalable Multi-Hop Relational Reasoning for Knowledge-Aware Question Answering](https://arxiv.org/abs/2005.00646)|EMNLP 2020|`CommonQA` 提出了一种结合GNN与关系路径编码的知识推理与获取知识表示的方式。先抽取知识路径，再利用改进后的GNN在路径上进行信息传播。|
|8| [Connecting the Dots: A Knowledgeable Path Generator for Commonsense Question Answering](https://arxiv.org/abs/2005.00691) | EMNLP 2020 Findings | 图2部分介绍了KG增强的QA模型框架，本文主要工作在于，已有的常识库，例如`ConceptNet`比较稀疏，可能仍然不能够填充从问题到正确答案的推理链，所以作者干脆直接在问题和答案中生成一条推理路径，这样的推理路径可能是KG中所没有的，以此来解决这个问题。其中生成推理路径的数据集是在KG上通过随机游走的方式得到的，并利用GPT2训练了一个路径生成模型。在`OpenBookQA`上达到了80.05(±0.68) |
|9|[UnifiedQA: Crossing Format Boundaries With a Single QA System](http://danielkhashabi.com/files/2020_unifiedqa/unifiedqa.pdf)|EMNLP 2020 Findings | OBQA SOTA模型，87.2acc，[TODO] |
|10|[Do Transformers Dream of Inference, or Can Pretrained Generative Models Learn Implicit Inferential Rules?](https://www.semanticscholar.org/paper/Do-Transformers-Dream-of-Inference%2C-or-Can-Models-Liang-Surdeanu/bd2239d6cea24604ff3687d37f3d475f6d7b12bc) | EMNLP Workshop 2020 | 简单的看了一下这篇工作，现有的一些预训练语言模型虽然在QA问题方面产生了良好的表现，但却拥有非常差的可解释性，本文以`T5`为例，在`QASC`数据集上进行了探索，研究预训练语言模型是否具有良好的推理性能，设定的任务为：给定premise statements，让模型生成combined statement（和`QASC`模型的标注过程有关）。最终的结论显示，PTM能完成一些简单的推理，但当问题比较复杂时，贴进真实世界时以及需要尝试时，模型的表现效果并不好。 |
|11| [Improving Commonsense Question Answering by Graph-based Iterative Retrieval over Multiple Knowledge Sources](https://arxiv.org/abs/2011.02705) | COLING 2020 | 在多个知识库（`ConceptNet`、`Wikipedia`以及`Cambridge Dictionary`）中进行常识的捕获，基于图的迭代式检索，根据初始化结点（利用问题和候选答案中的concept作初始化结点）以及缩小范围化的关系在`ConceptNet`上迭代式得产生结构化图，在`Wikipedia`上进行相似度计算并保留top10文本（句子级），其中问题和候选答案中的concept均会在`Cambridge Dictionary`中查找相应的解释并拼接在后面。消融实验证明了这三个知识库都起到了作用。在`CommonseQA`上进行了验证。|
|12| [Designing Templates for Eliciting Commonsense Knowledge from Pretrained Sequence-to-Sequence Models](https://www.aclweb.org/anthology/2020.coling-main.307/) | COLING 2020(short) | 设计模板，利用预训练语言模型当中捕获的隐式知识来完成多跳推理，选用了T5语言模型，将多项选择问题建模成NLI（自然语言推断）问题，设计模板将原问题和答案拆分成前提与假设，判断是否能通过前提来推断出假设。在`OBQA`的测试集上达到了83.2的acc | 
|13|[Learning Contextualized Knowledge Structures for Commonsense Reasoning](https://arxiv.org/abs/2010.12873) | arXiv 2020 | 是对`Connecting the Dots`那篇的一个改进，提出了`Hybrid Graph Network`，指出现有的一些工作基本都利用了KG来完成常识推理，可是KG是稀疏的，可能会存在着一些关系的缺漏，从而导致最终的效果不太理想。`Connecting the Dots`那篇文章中，会对任意两个`Concept`生成一条伪路径，但如果两个`Concept`相距过长，则生成的伪路径并不可靠，也很难用已有的预定义关系集合来描述长距离的两个`Concept`，除此之外，在`ConceptNet`上抽取得到的fact与生成出的fact，可能与context（question and candidate answer）的中心主题不一样，这些都限制了模型的表现。因此本文提出了`HGN`，这个方法会在所有的question mentioned concept与所有的answer mentioned concept之间建立一个全连接的图，然后如果`ConceptNet`中存在的关系，则用`ConceptNet`上学习到的关系表示来初始化边，如果不在，则先利用路径生成（也是基于GPT2，并在推理阶段固定GPT2的参数）方法产生表示，再将表示进行维度变换操作，以得到最终表示。接着在图上利用GNN的一个变体更新表示，并在最后基于注意力机制聚合全图信息，最终预测出question-candidate answer的概率。目前已开源。 |
|14|[Knowledge Fusion and Semantic Knowledge Ranking for Open Domain Question Answering](https://arxiv.org/abs/2004.03101)| arXiv 2020 | 挺特别的一个工作，没有使用KG在`OBQA`测试集上达到了80.0的acc，信息检索部分，知识库使用的是`OBQA`、`QASC`以及`ARC`三个数据集中自带的非结构化文本知识。检索采用了`Elasticsearch`（一个基于`Lucene`的搜索引擎），检索本身采取了二次检索的方案，第一步使用question+answer(candidate)来检索，第二步使用第一步检索的fact与第一步query之间的差集来检索。二次检索完后，使用了基于BERT的相似度计算模型，其实就是一个rerank，最后在知识融合的QA模型里（融合指的是检索到的知识与预训练语言模型本身蕴含的知识）利用了（1）[CLS_Ai] Ui Q [SEP] Ai [SEP] （2）[CLS_C] Q [SEP] A1 ... [SEP] C [SEP]，获得对应的CLS token embedding，其中Ui代表与候选答案Ai相关的fact，C代表与问题本身相关的，所有候选答案都共用的fact（一开始使用了分类模型，将每一个fact分成了这两类），最后利用[CLS_Ai]与[CLS_C]共同计算答案Ai的概率。|
|15| [Context Modeling with Evidence Filter for Multiple Choice Question Answering](https://arxiv.org/abs/2010.02649) | arXiv 2020 | 该文基于对`OpenBookQA`的观察，针对于支撑句提出了两个假设：（1）如果一个句子与四个选项的关联度都差不多，那么这个句子很有可能对推理答案没有用。（2）如果一个句子与一个选项的关联度高，而与其余的选项关联度低，那么该句很有可能是支撑句。以往的工作都是将不同的选项进行独立判断，所以没有使用到这两个启发式的假设。作者提出的模型中先独立的抽取了支撑句，然后根据支撑句与其余选项的关联进行调整以得到最终的支撑句。由于本文没有使用任何外部知识（例如`ConceptNet`）所以效果自然不好，在`OpenBookQA`测试集上仅达到了65.6的acc |
|16|[Benchmarking Knowledge-Enhanced Commonsense Question Answering via Knowledge-to-Text Transformation](https://arxiv.org/abs/2101.00797)|AAAI 2021|[TODO]|
|17|[Understanding Few-Shot Commonsense Knowledge Models](https://arxiv.org/abs/2101.00297)|arXiv 2021||

## 14.信息检索（包括rerank）
> 因为许多数据集要面临检索过程，例如open-domain question answering，本部分记录一下自己看到过的主要在做检索工作的文章。

| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1| The Probabilistic Relevance Framework: BM25 and Beyond | 2009 | `BM25`，传统信息检索的结晶，直接用[Lucene Java Implementation](https://lucene.apache.org/)即可 |
|2| [Multilingual Universal Sentence Encoder for Semantic Retrieval](https://arxiv.org/abs/1907.04307) | ACL 2020 | 提出了`USE-QA`，是一个基于transformer的信息检索系统，效果算是基于transformer的深度语义检索中比较好的。 |
|3| [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) | EMNLP 2020 | 陈丹琦与민세원，提出了一个基于BERT的检索系统`DPR`（[repo](https://github.com/facebookresearch/DPR)），在OpenQA场景下超越了传统IR（`BM25`或`TFIDF`），采用的方法也比较简单，对所有语料库中的passage进行编码，对于一个问题，用另一个编码器进行编码，之后使用MIPS算法计算top-k passage作为检索结果。为了使得问题编码后的向量与其相关的passage之间的内积尽可能的大，对两个编码器（用了两个独立的BERT）进行了训练，在负采样上使用了一些技巧。 |
|4| [Using the Hammer Only on Nails: A Hybrid Method for Evidence Retrieval for Question Answering](https://arxiv.org/abs/2009.10791) | arXiv 2020 | 提出了一个混合的检索系统，综合了`BM25`（传统的信息检索方法）和`USE-QA`（基于transformer的信息检索方法）。出发点是由于基于transformer的信息检索方法忽略了词表面token overlap的这一有效信号，所以想让两种方法进行结合。结合的方法其实非常简单，就是对一个query，先利用传统IR检索系统（例如`BM25`）进行检索，得到的top分值如果大于一个阈值，则相信检索结果，若小于该阈值，则选择不相信传统IR的检索结果，转而使用基于transformer的检索系统。 |
|5|[Differentiable Open-Ended Commonsense Reasoning](https://arxiv.org/abs/2010.14439)|arXiv 2020|林禹臣，提出了开放式常识推理任务，以往的常识推理数据集大多都是候选答案式的，但这样的设定离真实场景还有距离。修改了原有的数据集，并提出了`DrFact`模型，可以在文本语料库上进行可微的多跳推理。其中使用到了`GenericsKB`知识语料库|
|6|[Learning Dense Representations of Phrases at Scale](https://arxiv.org/abs/2012.12624) | arXiv 2020 | 陈丹琦，一个phrase级别的信息检索工作，旨在为大量的phrase学习到一个稠密的表示，本文提出的方法`DensePhrase`卖点在于速度快，其准确度方面和`DPR`之间存在一点差距。为了学习到稠密的phrase表示，对于每一个候选phrase都会利用QG模型生成一个问题，以此来训练。在训练过程当中，还是用了蒸馏方法，在负采样方面也使用了一些技巧。 |
|7|[Reader-Guided Passage Reranking for Open-Domain Question Answering](https://arxiv.org/abs/2101.00294)|arXiv 2021| rerank上的工作，号称不用任何训练就可以直接提升检索的概率（在rerank后） [TODO] |

## 15.可解释性研究
| 序号 | 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: | :---: |
|1|[Exploiting Explicit Paths for Multi-hop Reading Comprehension](https://www.aclweb.org/anthology/P19-1263)|ACL 2019|`Wikihop`与`OpenBookQA` `Open`&`non-Open` 这篇工作主要的贡献在于多跳阅读理解的可解释性，为了在文本数据上达到多跳的效果，会有两种方法：GNN或者路径抽取，GNN可解释性非常差，因为它是隐式地完成信息传递。而路径抽取的方法解释性强，但如果跳数增多的话会有语义漂移问题。不过`Wikihop`或者`OpenBookQA`数据集都是两跳，所以好像不严重？然后作者就通过在问题中提取头实体，在候选答案中提取尾实体，然后在**全部候选文档**中抽取多个推理链，接着对推理链的实体做表示初始化然后隐式提取关系，再通过关系计算路径的表示。最后会对路径进行打分，然后根据分数得到最终的答案概率分布。我个人觉得这篇工作利用两个实体的表示去直接计算他们的关系表示这里有点粗糙了，因为两个实体之间可能存在着不止一种关系，而利用作者所给的式子则无法对这种多样的关系进行学习。|
|2|[Answering while Summarizing: Multi-task Learning for Multi-hop QA with Evidence Extraction](https://www.aclweb.org/anthology/P19-1225/)|ACL 2019|`多跳阅读理解`、`抽取式文本摘要`以及`文本蕴含` 可解释性研究，受抽取式文本摘要的灵感，提出了QFE(Query Focused Extractor)模型。创新度不是很高，和原始的HotpotQA baseline挺像的。只不过多了一个支撑句预测层，该层就是他设计的QFE模型|
|3|[R4C: A Benchmark for Evaluating RC Systems to Get the Right Answer for the Right Reason](https://arxiv.org/abs/1910.04601)|ACL 2020|`R4C` **给出半结构化的derivation** 对`HotpotQA`数据集进行了标注，一共标注了约5K个实例，每个实例被标注了三个*derivation*。`R4C`觉得*Supporting Sentence*是一个非常粗粒度的概念，因为一句*Supporting Sentence*里面可能有些内容并不是推理答案所必须的，而另一部分是必须的，因此作者标注了更加细粒度的*derivation*，这是一种半结构化的形式来表达推理信息，比*Supporting Sentences*更加有挑战性，也对模型的可解释性要求更高。|
|4|[Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering](https://arxiv.org/abs/2010.03274)|EMNLP 2020|`eQASC`、`eQASC-perturbed`以及`eOBQA` `Textual entailment` **多选式** 这个工作其实是在研究多跳推理问题的可解释性。在`QASC`数据集的基础上，针对于每个问题又标注了多个推理链（有效或无效都有）构成了`eQASC`，接着又将推理链模板化（使推理链更加通用）构成了`eQASC-perturbed`数据集，最后为了进行out-domain test，也基于`OpenBookQA`数据集标注了推理链形成了`eOBQA`并进行测试|  
|5*|[Unsupervised Explanation Generation for Machine Reading Comprehension](https://arxiv.org/abs/2011.06737)|arXiv 2020|可解释性研究，崔神新作，在MRC中通过无监督的方式来让模型输出推理所需的句子。只有少数的数据集有支撑句标注信号，例如`HotpotQA`，大量的数据集上都没有这样的支撑句监督信号，所以作者基于假设：**系统可以使用原有上下文中的少量信息来达到和原来推理类似的结论** 设计了循环动态门控机制（Recursive Dynamic Gating）。在模型的整体框架方面，采用了知识蒸馏的部分思想，分为Teacher模型和Student模型，其中Teacher模型接受原有问题、候选答案以及上下文来学习预测答案，而Student模型则意旨用更少量的上下文信息也能够达到和Teacher模型类似的效果。最终使用Teacher模型来预测答案，用Student模型来预测支撑句。那么如何使用更少量的信息呢？这里就用到了循环动态门控机制，该机制可以使BERT类模型（在本文中采用了ALBERT）中每一层transformer的输出在传递到下一层transformer中进行衰减。具体细节就不赘述了。在Loss方面使用了答案预测的loss，模型蒸馏的loss以及一个余弦相似度loss用来使模型更好的学习循环动态门控机制中的参数|


(*代表仅属于本分类下的工作)

## [PLAN]
| 论文 | 发表会议 | 备注 |
| :---: | :---: | :---: |
|[Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://www.aclweb.org/anthology/2020.acl-main.412)|ACL 2020|`KGQA`|
|[Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases](https://www.aclweb.org/anthology/2020.acl-main.91)|ACL 2020|`KBQA/KGQA`|
|[Adapting Meta Knowledge Graph Information for Multi-Hop Reasoning over Few-Shot Relations](https://www.aclweb.org/anthology/D19-1334)|ACL 2019|`KGQA`|
|[Quick and (not so) Dirty: Unsupervised Selection of Justification Sentences for Multi-hop Question Answering](https://www.aclweb.org/anthology/D19-1260)|EMNLP 2019|`ARC`、`MultiRC` 提出了一种无监督的支撑句选取方法，基本思路是最大化选取句子的相关度，最小化选取supporting fact的overlap以及最大化问题和答案的覆盖度。|
|[Differentiable Reasoning over a Virtual Knowledge Base](https://openreview.net/forum?id=SJxstlHFPH)|ICLR 2020|在`HotpotQA`上效果不是很好，主要在`MetaQA`达到了比较好的效果，让模型在纯文本的语料库上也能够模仿在知识库上做检索。所以是Virtual Knowledge Base|
|[Multi-step Retriever-Reader Interaction for Scalable Open-domain Question Answering](https://openreview.net/forum?id=HkfPSh05K7)|ICLR 2019|[TODO]|
|[Question answering as global reasoning over semantic abstractions](https://arxiv.org/abs/1906.03672)|AAAI 2018|[TODO]|

## A.Leaderboard of HotpotQA
### A.1 ANS (distractor setting)
| 模型 | 对应论文 | 所属类别 | EM(Test) | F1(Test) | EM(dev) | F1(dev) |TOP|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **QUARK** (arXiv 2020) | 2.1 | （1） | - | - | **67.75** | **81.21** | TOP-3 |
| DFGN (ACL 2019) | 3.2 | （2） | 56.31 | 69.69 | - | - | |
| KGNN (arXiv 2019) | 3.5 | （2） | 50.81 | 65.75 | - | - | |
| **HGN** (EMNLP 2020) | 3.6 | （2） | **69.22** | **82.19** | - | - | TOP-1 |
| **C2F Reader** (EMNLP 2020) | 3.7 | （2） | **67.98** | **81.24** | - | - | TOP-2 |
| **SAE** (AAAI 2020) | 3.8 | （2） | **66.92** | **79.62** | **67.70** | **80.75** | TOP-4 | 
| ICLR 2020 | 5.7 | （4） | - | - | 81.2 | 68.0 | |
| DECOMPRC (ACL 2019) | 6.2 | （5） | - | 70.57 | - | - | |
| **Unsupervised** (EMNLP 2020) | 6.5 | （5） | **66.33** | **79.34** | - | **80.1** | TOP-5 |
| TheirNMN (EMNLP 2019) | 7.1 | （6） | 49.58 | 62.71 | 50.67 | 63.35 | |
| MODULARQA (arXiv 2020) | 7.4 | （6） | - | - | - | 61.8 | |


### A.2 ANS (fullwiki setting)
| 模型 | 对应论文 | 所属类别 | EM(Test) | F1(Test) |TOP|
| :---: | :---: | :---: | :---: | :---: | :---: |
| **QUARK (arXiv 2020)** | 2.2 | （1） | **55.50** | **67.51** | TOP-4 |
| KGNN (arXiv 2019) | 3.5 | （2） | 27.65 | 37.19 | |
| **HGN (EMNLP 2020)** | 3.6 | （2） | **59.74** | **71.41** | TOP-3 |
| MUPPET (ACL 2019) | 4.2 | （3） | 31.07 | 40.42 | |
| whole pip (EMNLP 2019) | 4.3 | （3） | 45.30 | 57.30 | |
| **DDRQA (arXiv 2020)** | 4.6 | （3） | **62.9** | **76.9** | TOP-1 |
| CogQA (ACL 2019) | 5.2 | （4） | 37.1 | 48.9 | |
| GOLDEN (EMNLP 2019) | 5.6 | （4） | 37.92 | 48.58 | |
| **ICLR 2020** | 5.7 | （4） | **60.0** | **73.0** | TOP-2 |
| **Transformer-XH (ICLR 2020)** | 8.1 | （7） | **51.6** | **64.1** | TOP-5 |

### A.3 所属类别及入选TOP数量：  
（1）基于单步阅读理解模型改进：1 （其中一个同时入选两个setting）   
（2）基于GNN：3 （其中一个同时入选两个setting）  
（3）迭代式文档检索：1  
（4）推理链：1   
（5）分解问题：1  
（6）NMN：0   
（7）PTM：1   
