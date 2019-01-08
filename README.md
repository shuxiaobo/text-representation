# text-representation

* Text representation works, such as : paper, code, review, datasets, blogs, thesis and so on. 

* This repository is created for the researchers who are interested in text representation. The scope of text representation including: word embedding, sentence embedding, exploration experiments and RNN module (or some else like VAE). this repository will give the lastest and full works from the everywhere I touch on. Hope your reply and suggestion, thanks.

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

> Transfer learning

- Discourse-Based Objectives
for Fast Unsupervised Sentence Representation Learning
本文使用无监督句子学习方式，迁移sentence encoder。设计了3个预训练目标来预训练已有的句子编码器（Sentence encoder），当模型经过这3个预训练后在针对目标任务进行训练
1. 句子顺序是否正确
2. 是否是当前句子的下一句
3. Conjunction Prediction，取具有连接词开头的句子，然后去掉连接词，让模型去预测
可以多任务学习，也可以pipeline学习 。实验在几个简单的分类数据集上证明了在速度上有提升，且在分类acc上有限度的下降。

> Subword

- TACL2017-Enriching Word Vectors with Subword Information
    本篇文章通过建立一个 Subword 词典，来丰富word的信息，比如where的subword是<wh, whe, her, ere, re>这样的，词的表示是embedding(word) + \sum embedding(n-gram)，使用Skip-gram来训练词向量，是一个无监督训练模型。
    实验证明：
    - 在越大的数据集上，embedding的维度可以适当放大。
    - 加入n-grams真的很有用，特别是数据小时，相较word2vec优势更加明显。
    - 在不同语言下都比较好用，特别是那些rich language。
    - 在句法形态评估任务上效果较为显著。
    
- EMNLP2017 Mimicking Word Embeddings using Subword RNNs
    佐治亚理工的工作，很easy也很work的想法，文章主要用subword的信息来解决OOV问题
训练subword的表示方法：把char embedding表示的word和pre-train的word embedding做欧式距离损失，最后直接用char embedding来表示单词
本文中使用LSTM来融合char做word embedding 

> Ambiguity
 - ICLR2017-Multimodal Word Distributions
 
    文章提出使用GMM分布来学习词的多个语义
    
> Ngram

- EMNLP2017 Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics
    文章提出了使用Ngram来训练skip-gram的word embedidng，把每个Ngram都作为一个unique的embedding，在训练word2Ngram时，不仅中心word要预测
    周边的Ngram，同时也要预测周边的word。Ngram2Ngram相同。实验对比了word和Ngram的不同组合形式(Ngram2Ngram相同, )训练出来的Ngram embedding，
    还有使用不同的训练方法训练出来的embedding包括SGNS, GloVe, PPMI, and SVD




### Sentence representation

### Exploration experiments

### RNN module (or some else like VAE)


## 2. 理论与综述文章(Theoretical and survey papers)


## 3. 相关代码 (Available codes)

## 4. 文本表示领域代表性研究学者 (Scholars)

## 5. 相关的硕博士论文 (Thesis)

## 6. 常用公开数据集及算法结果 (Datasets and benchmark)

## 7. 比赛 (Challenges and competitions)

## 8. 其他 (Miscellaneous)


