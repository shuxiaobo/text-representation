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
### Word representation (#18)

> Active Learning

- AAAI2017. Active Discriminative Text Representation Learning. Ye Zhang et.al
    
    文章使用主动学习来学习word embedding，主动学习的关键是：挑选样例使得模型收益最大，主要有一下三种方法：
    - 随机挑选
    - 不确定性采样，一般使用熵
    - 期望梯度长度

    文章主要使用第三种方法，分为了两种：使得word embedding 梯度最大的样例来学习。2. 使得softmax分类线性层参数梯度最大的样例学习。最后对比了主动学习上面所提到的几种方法在文本分类中的performance。


> Subword && OOV

- TACL2017-Enriching Word Vectors with Subword Information
    本篇文章通过建立一个 Subword 词典，来丰富word的信息，比如where的subword是<wh, whe, her, ere, re>这样的，词的表示是embedding(word) + \sum embedding(n-gram)，使用Skip-gram来训练词向量，是一个无监督训练模型。
    实验证明：
    - 在越大的数据集上，embedding的维度可以适当放大。
    - 加入n-grams真的很有用，特别是数据小时，相较word2vec优势更加明显。
    - 在不同语言下都比较好用，特别是那些rich language。
    - 在句法形态评估任务上效果较为显著。
    
- EMNLP2017 Mimicking Word Embeddings using Subword RNNs

    佐治亚理工的工作，很easy也很work的想法，文章主要用subword的信息来解决OOV问题训练subword的表示方法：把char embedding表示的word和pre-train的word embedding做欧式距离损失，最后直接用char embedding来表示单词本文中使用LSTM来融合char做word embedding

- EMNLP2018 Generalizing Word Embeddings using Bag of Subwords
    之前的word3vec使用的是上下文单词来训练。本篇文章把词看做是character n-grams，使用CBOW-like的方法，在更好的训练word embedding的同时可以用来解决OOV问题。最后的实验结果表明，本文提出来的方法在POS和Similarity任务中表现的SOA。

- EMNLP2018 Learning Better Internal Structure of Words for Sequence Labeling
    以往的利用char embeddin模型里，没有详细的说明哪一种结构适合于不同颗粒度的表示的结合。本篇文章对比了之前的几种CNN模型，比较了他们的优缺点，并提出一种新的funnel-shaped CNN with no down-sample，这种模型可以学习到更好的句子结构表示。实验在POS,NER,Syntactic chunking中表现的非常不错。


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
    
- EMNLP2018 Leveraging Gloss Knowledge in Neural Word Sense Disambiguation by Hierarchical Co-Attention
    本文主要是利用Word Sense Disambiguation（WSD）的数据集来学习词的歧义表示。WSD数据集里面会给出歧义词想对应的词意的解释。在模型上，使用Co-attention来捕捉word和sentence的信息。实验结果在一些POS和WSD问题上表现的很好。
    
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
    
> Transfer learning && Contextual

- Discourse-Based Objectives for Fast Unsupervised Sentence Representation Learning
本文使用无监督句子学习方式，迁移sentence encoder。设计了3个预训练目标来预训练已有的句子编码器（Sentence encoder），当模型经过这3个预训练后在针对目标任务进行训练
    * 句子顺序是否正确
    * 是否是当前句子的下一句
    * Conjunction Prediction，取具有连接词开头的句子，然后去掉连接词，让模型去预测
三个任务可以一起学习，也可以pipeline的学习。实验在几个简单的分类数据集上证明了在速度上有提升，且在分类acc上有限度的下降。

- NAACL2018 Contextualized Word Representations for Reading Comprehension

    大多数的Reading Comprehension论文都集中精力在Q和D的interact attention上。文章做了实验，使用Contextual表示的Q和D要在结果中表现的更好。
    
- NAACL2018**BESTPAPER** Deep contextualized word representations (ELMO)
    过往的word2vec方法是利用词语的共现来训练word embedding，它们大多没有融入上下文的信息。这一次我们利用language model(LM)的方法结合RNN+char embedding训练出来的RNN和char embedding迁移到其他下游任务中都有巨大的提升。这是一种可以利用大规模语料的无监督方法。缺点也显而易见，Contextual没有利用双向的语言信息，同时没考虑词频信息。
    Note: 建议结合论文：Dissecting Contextual Word Embeddings: Architecture and Representation来理解ELMo。

- ICLR2017 Learned in Translation: Contextualized Word Vectors (CoVe)
    在Image领域，很多问题的解决得益于迁移学习。同样的想法应用在NLP领域。作者认为能理解上下文的模型是可以应用迁移学习的，比如说在机器翻译中。文章使用en2ge的翻译任务训练处encoder然后迁移到其他任务中。把CoVe用到了语义情感分析、问题分类、文本蕴含、问答等多种NLP任务上，这些模型的性能都得到了提升。

> Co-occurance

- EMNLP2017 Dict2vec : Learning Word Embeddings using Lexical Dictionaries
    Word2Vec的训练方式是利用上下文的词共现来训练词。本文提出利用词典的词解释来作为共现语料训练词。文中定义了Strong Pair，意为双方的解释里都出现了对方。然后只出现一方的叫Weak Pair。结合Skip-gram来训练。本篇文章也算是提出了一种新奇的解决思路。
    

- ICLR2019 CBOW IS NOT ALL YOU NEED: COMBINING CBOW WITH THE COMPOSITIONAL MATRIX SPACE MODEL
    连续词袋模型(CBOW)的一个缺点是没法掌握词序，本文提出一个新的word embedding的初始化方法和新的Loss，结合CBOW方法一起训练。最后的实验结果表明这种训练方法在大多分类和STS任务中都比CBOW要好。同时也能比其他初始化方法记住更多的信息。

- EMNLP2018 Quantifying Context Overlap for Training Word Embeddings
    Word2Vec的训练方式是利用上下文的词共现来训练词，但是忽略了训练上下文词之间的语义相似度。本文在Glove上做出修改，不使用共现的次数作为损失函数，而是使用两个词之间Point-wise Mutual Information (PMI)的交集词的最小PMI值作为损失函数。实验在Word similarity and analogy results都表现的很好，同样几个分类数据集也表现的不错。(对比Glove,SGNS)

> Others

- ICLR2019 Adaptive Input Representations for Neural Language Modeling

    FB2019的工作。文章主要借鉴adaptive softmax的想法，使用词的词频对词进行聚簇，文中设定5个簇，每个簇的embedding用不同维度去表示。然后对每一个簇设置一个矩阵，目的是把词映射到相同的维度。目的是去对高频的Word embedding进行表示能力的提升，较少低频词的过拟合。实验在BILLION WORD、WIKITEXT-103对比了CharCNN,Subword的模型，有perplexity提升。
    
- ACL2018 Joint Embedding of Words and Labels for Text Classification
    文章提出来一个把word和label用在一起来表示文本的想法，融合的方法是使用attention。该想法用在分类问题中，在几个分类分体中表现的还不错。

### Sentence representation (#1)

> Context Information

- EMNLP2017 A Deep Neural Network Sentence Level Classification Method with Context Information

    文章主要使用了LSTM+CNN的结构，同时使用FOFE编码上下文句子来辅助句子分类任务。 这里面对的问题是，对于句子分类任务，文中把这个句子称作是Focus，其他的句子分Left、Right。 因为句子分类任务是没有上下文的，实验中作者是通过查找数据集来源，来补充上下文句子。 文章的实验主要证明了FOFE or Context信息有用。
    

### Exploration experiments

### RNN module (or some else like VAE)


## 2. 理论与综述文章(Theoretical and survey papers)


## 3. 相关代码 (Available codes)

## 4. 文本表示领域代表性研究学者 (Scholars)
word2vec 作者 Tomas Mikolov
1、Efficient Estimation of Word Representation in Vector Space, 2013

2、Distributed Representations of Sentences and Documents, 2014

3、Enriching Word Vectors with Subword Information, 2016



## 5. 相关的硕博士论文 (Thesis)

## 6. 常用公开数据集及算法结果 (Datasets and benchmark)

## 7. 比赛 (Challenges and competitions)

## 8. 其他 (Miscellaneous)






