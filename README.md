# text-representation

- Text representation works, such as : paper, code, review, datasets, blogs, thesis and so on. 

- This repository is created for the researchers who are interested in text representation. The scope of text representation including: word embedding, sentence embedding, exploration experiments and RNN module (or some else like VAE). this repository will give the lastest and full works from the everywhere I touch on. Hope your reply and suggestion, thanks.

## 目录 Table of contents

* [文本表示简介 Introduction to text representation](#0文本表示简介)

* [研究领域与最新相关文章 Research latest articles by area](#1latest)

* [理论与综述文章 Theoretical and survey papers](#3理论与综述文章)

* [相关代码 Available codes](#4代码)

* [文本表示领域代表性研究学者 Scholars](#5文本表示领域代表性研究学者)

* [相关的硕博士论文 Thesis](#6文本表示领域相关的硕博士论文)

* [常用公开数据集及算法结果 Datasets and benchmark](#7公开数据集及实验结果)

* [比赛 Challenges and competitions](#8比赛)

* [其他 Miscellaneous](#其他)

* [Contributing](#contributing)

## 0. 文本表示简介 (Introduction to text representation)



## 1. 研究领域与最新相关文章 (Research latest articles by area)
### Word representation

> Active Learning

- AAAI2017. Active Discriminative Text Representation Learning. Ye Zhang et.al
    
    文章使用主动学习来学习word embedding，主动学习的关键是：挑选样例使得模型收益最大，主要有一下三种方法：
    - 随机挑选
    - 不确定性采样，一般使用熵
    - 期望梯度长度

    文章主要使用第三种方法，分为了两种：使得word embedding 梯度最大的样例来学习。2. 使得softmax分类线性层参数梯度最大的样例学习。最后对比了主动学习上面所提到的几种方法在文本分类中的performance。


> Subword

- TACL2017-Enriching Word Vectors with Subword Information
    本篇文章通过建立一个 Subword 词典，来丰富word的信息，比如where的subword是<wh, whe, her, ere, re>这样的，词的表示是embedding(word) + \sum embedding(n-gram)，使用Skip-gram来训练词向量，是一个无监督训练模型。
    实验证明：
    - 在越大的数据集上，embedding的维度可以适当放大。
    - 加入n-grams真的很有用，特别是数据小时，相较word2vec优势更加明显。
    - 在不同语言下都比较好用，特别是那些rich language。
    - 在句法形态评估任务上效果较为显著。
    
- EMNLP2017 Mimicking Word Embeddings using Subword RNNs

    佐治亚理工的工作，很easy也很work的想法，文章主要用subword的信息来解决OOV问题训练subword的表示方法：把char embedding表示的word和pre-train的word embedding做欧式距离损失，最后直接用char embedding来表示单词本文中使用LSTM来融合char做word embedding

- EMNLP2018 Adapting Word Embeddings to New Languages with Morphological and Phonological Subword Representations
**待完成**

> Ambiguity
 
 - ICLR2017-Multimodal Word Distributions
 
    文章提出使用GMM分布来学习词的多个语义(对GMD不太了解的我，没太看懂)

- EMNLP2017 Outta Control: Laws of Semantic Change and Inherent Biases in Word Representation Models
    文章主要研究歧义词的词意的改变和时间的关系。
    通过研究历史学的语料(1990-1999)，文中发现，词意的改变并不是跟时间有完全的关系
    - 词频与词意的改变是负相关的
    - 歧义的多少与词意的改变正相关
    - 词在相应类别中的代表性与词意改变负相关
    
- ICLR2017 Beyond Bilingual: Multi-sense Word Embeddings using Multilingual Context
    
    过往的一词多义问题通过context来进行解决。近来也有学者发现使用多语言解决一词多义问题，因为对于相同单词不同含义在其他语言中可能对应着不同的单词，但是之前的方法大多存在固定一个多义词含义数量的问题。本文通过multi-view Bayesian non-parametric方法在多语言上解决之前的限制。
    
> Ngram

- EMNLP2017 Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics
    文章提出了使用Ngram来训练skip-gram的word embedidng，把每个Ngram都作为一个unique的embedding，在训练word2Ngram时，不仅中心word要预测
    周边的Ngram，同时也要预测周边的word。Ngram2Ngram相同。实验对比了word和Ngram的不同组合形式(Ngram2Ngram相同, )训练出来的Ngram embedding，
    还有使用不同的训练方法训练出来的embedding包括SGNS, GloVe, PPMI, and SVD
    实验结果在word similarity和analogy表明，Ngram训练是有优势的，同时在neiberhoods中，ngram同样可以学习出近义词，甚至学的更好

- ENMLP2017 Nonsymbolic Text Representation
    文章主要是做了一个探索性的实验，目的是证明，在不用空格分词的情况下，在某些实验上也能得到还可以的表现。

- EMNLP2017 Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features
    文章提出一种新的Sentence Embedding模型Sent2Vec，使用了Ngram来辅助无监督Sentence Embedding的任务。具体做法是对句子的Ngram(包括unigram)求和然后求平均来表示句子，通过预测缺失的单词来训练词向量。同时文章也提出了方法屏蔽词频的影响。
    实验在GLUE的数据集上对比了一些常见的sentence表示的方法，并没有在所有的任务中表现SOA。
    
- ICLR2018 A NEW METHOD OF REGION EMBEDDING FOR TEXT CLASSIFICATION

    百度研究院2018的工作。在文本分类中，词袋模型有它的限制。他对单词的表示没有包含单词的顺序信息。N-gram解决了这一问题，但n-gram也有局限: 当n较大时，通常导致数据缺失。在新模型中，region embedding有两部分组成：单词本身的embedding有向量表示，联系上下文的embedding由词向量与上下文关联的矩阵表示。文本看作是region embedding的集合。在文本分类的实验中(主要是Yelp Dataset)大多数实验有提升
    
> Transfer learning

- Discourse-Based Objectives for Fast Unsupervised Sentence Representation Learning
本文使用无监督句子学习方式，迁移sentence encoder。设计了3个预训练目标来预训练已有的句子编码器（Sentence encoder），当模型经过这3个预训练后在针对目标任务进行训练
1. 句子顺序是否正确
2. 是否是当前句子的下一句
3. Conjunction Prediction，取具有连接词开头的句子，然后去掉连接词，让模型去预测
三个任务可以一起学习，也可以pipeline的学习。实验在几个简单的分类数据集上证明了在速度上有提升，且在分类acc上有限度的下降。

> Others

- ICLR2019 Adaptive Input Representations for Neural Language Modeling

    FB2019的工作。文章主要借鉴adaptive softmax的想法，使用词的词频对词进行聚簇，文中设定5个簇，每个簇的embedding用不同维度去表示。然后对每一个簇设置一个矩阵，目的是把词映射到相同的维度。目的是去对高频的Word embedding进行表示能力的提升，较少低频词的过拟合。实验在BILLION WORD、WIKITEXT-103对比了CharCNN,Subword的模型，有perplexity提升。

### Sentence representation

> Context Information

- EMNLP2017 A Deep Neural Network Sentence Level Classification Method with Context Information

    文章主要使用了LSTM+CNN的结构，同时使用FOFE编码上下文句子来辅助句子分类任务。 这里面对的问题是，对于句子分类任务，文中把这个句子称作是Focus，其他的句子分Left、Right。 因为句子分类任务是没有上下文的，实验中作者是通过查找数据集来源，来补充上下文句子。 文章的实验主要证明了FOFE or Context信息有用。

### Exploration experiments

### RNN module (or some else like VAE)


## 2. 理论与综述文章(Theoretical and survey papers)


## 3. 相关代码 (Available codes)

## 4. 文本表示领域代表性研究学者 (Scholars)

## 5. 相关的硕博士论文 (Thesis)

## 6. 常用公开数据集及算法结果 (Datasets and benchmark)

## 7. 比赛 (Challenges and competitions)

## 8. 其他 (Miscellaneous)


