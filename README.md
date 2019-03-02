# text-representation

- Text representation works, such as : paper, code, review, datasets, blogs, thesis and so on. 

- This repository is created for the researchers who are interested in text representation. The scope of text representation including: word embedding, sentence embedding, exploration experiments and RNN module (or some else like VAE). this repository will give the latest and full works from the everywhere I touch on. Hope your reply and suggestion, thanks.

## 目录 Table of contents

* [文本表示简介 (Introduction to text representation)](#0文本表示简介-introduction-to-text-representation)

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

> Active Learning (#1)

- AAAI2017. [Active Discriminative Text Representation Learning. Ye Zhang et.al](https://arxiv.org/pdf/1606.04212.pdf)
    
    文章使用主动学习来学习word embedding，主动学习的关键是：挑选样例使得模型收益最大，主要有一下三种方法：
    - 随机挑选
    - 不确定性采样，一般使用熵
    - 期望梯度长度

    文章主要使用第三种方法，分为了两种：使得word embedding 梯度最大的样例来学习。2. 使得softmax分类线性层参数梯度最大的样例学习。最后对比了主动学习上面所提到的几种方法在文本分类中的performance。


> Subword && OOV (#5)

- TACL2017 [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1707.06961.pdf)
    本篇文章通过建立一个 Subword 词典，来丰富word的信息，比如where的subword是<wh, whe, her, ere, re>这样的，词的表示是embedding(word) + \sum embedding(n-gram)，使用Skip-gram来训练词向量，是一个无监督训练模型。
    实验证明：
    - 在越大的数据集上，embedding的维度可以适当放大。
    - 加入n-grams真的很有用，特别是数据小时，相较word2vec优势更加明显。
    - 在不同语言下都比较好用，特别是那些rich language。
    - 在句法形态评估任务上效果较为显著。
    
- EMNLP2017 [Mimicking Word Embeddings using Subword RNNs](https://arxiv.org/pdf/1707.06961.pdf)

    佐治亚理工的工作，很easy也很work的想法，文章主要用subword的信息来解决OOV问题训练subword的表示方法：把char embedding表示的word和pre-train的word embedding做欧式距离损失，最后直接用char embedding来表示单词本文中使用LSTM来融合char做word embedding。

- EMNLP2018 [Generalizing Word Embeddings using Bag of Subwords](https://arxiv.org/pdf/1809.04259.pdf)
    之前的word3vec使用的是上下文单词来训练。本篇文章把词看做是character n-grams，使用CBOW-like的方法，在更好的训练word embedding的同时可以用来解决OOV问题。最后的实验结果表明，本文提出来的方法在POS和Similarity任务中表现的SOA。

- EMNLP2018 [pLearning Better Internal Structure of Words for Sequence Labeling](http://aclweb.org/anthology/D18-1279)
    以往的利用char embeddin模型里，没有详细的说明哪一种结构适合于不同颗粒度的表示的结合。本篇文章对比了之前的几种CNN模型，比较了他们的优缺点，并提出一种新的funnel-shaped CNN with no down-sample，这种模型可以学习到更好的句子结构表示。实验在POS,NER,Syntactic chunking中表现的非常不错。

- ACL2018 [Compositional Representation of Morphologically-Rich Input for Neural Machine Translation](http://www.aclweb.org/anthology/P18-2049)
    
    本文主要提出来使用triple-char 过 rnn的表示作为subword的表示方法比LMVR, BPE效果在translation表现的更好。
    
- AAAI2019 [Learning Semantic Representations for Novel Words: Leveraging Both Form and Context
Timo](https://arxiv.org/abs/1811.03866.pdf)

    本文主要通过预训练用来解决OOV问题，通过char n-gram和word context分别做简单的平均来求得最后的词表示，损失采用pre-train的word embedding做MSE损失。

- EMNLP2018 [Adapting Word Embeddings to New Languages with Morphological and Phonological Subword Representations](https://arxiv.org/pdf/1808.09500.pdf)
**待完成**

> Ambiguity (#4)
 
 - ICLR2017 [Multimodal Word Distributions](https://arxiv.org/pdf/1704.08424.pdf)
 
    文章提出使用GMM分布来学习词的多个语义(对GMD不太了解的我，没太看懂)

- EMNLP2017 Outta Control: Laws of Semantic Change and Inherent Biases in Word Representation Models
    文章主要研究歧义词的词意的改变和时间的关系。
    通过研究历史学的语料(1990-1999)，文中发现，词意的改变并不是跟时间有完全的关系
    - 词频与词意的改变是负相关的
    - 歧义的多少与词意的改变正相关
    - 词在相应类别中的代表性与词意改变负相关
    
- ACL2017 [Beyond Bilingual: Multi-sense Word Embeddings using Multilingual Context](http://www.aclweb.org/anthology/W17-2613)
    
    过往的一词多义问题通过context来进行解决。近来也有学者发现使用多语言解决一词多义问题，因为对于相同单词不同含义在其他语言中可能对应着不同的单词，但是之前的方法大多存在固定一个多义词含义数量的问题。本文通过multi-view Bayesian non-parametric方法在多语言上解决之前的限制。
    
- EMNLP2018 [Leveraging Gloss Knowledge in Neural Word Sense Disambiguation by Hierarchical Co-Attention](http://aclweb.org/anthology/D18-1170)
    本文主要是利用Word Sense Disambiguation（WSD）的数据集来学习词的歧义表示。WSD数据集里面会给出歧义词想对应的词意的解释。在模型上，使用Co-attention来捕捉word和sentence的信息。实验结果在一些POS和WSD问题上表现的很好。
    
> Ngram (#4)

- EMNLP2017 [Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics](http://aclweb.org/anthology/D17-1023)
    文章提出了使用Ngram来训练skip-gram的word embedidng，把每个Ngram都作为一个unique的embedding，在训练word2Ngram时，不仅中心word要预测
    周边的Ngram，同时也要预测周边的word。Ngram2Ngram相同。实验对比了word和Ngram的不同组合形式(Ngram2Ngram相同, )训练出来的Ngram embedding，
    还有使用不同的训练方法训练出来的embedding包括SGNS, GloVe, PPMI, and SVD
    实验结果在word similarity和analogy表明，Ngram训练是有优势的，同时在neiberhoods中，ngram同样可以学习出近义词，甚至学的更好。

- ENMLP2017 [Nonsymbolic Text Representation](https://arxiv.org/pdf/1610.00479.pdf)
    文章主要是做了一个探索性的实验，目的是证明，在不用空格分词的情况下，在某些实验上也能得到还可以的表现。

- EMNLP2017 [Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features](https://arxiv.org/pdf/1703.02507.pdf)
    文章提出一种新的Sentence Embedding模型Sent2Vec，使用了Ngram来辅助无监督Sentence Embedding的任务。具体做法是对句子的Ngram(包括unigram)求和然后求平均来表示句子，通过预测缺失的单词来训练词向量。同时文章也提出了方法屏蔽词频的影响。
    实验在GLUE的数据集上对比了一些常见的sentence表示的方法，并没有在所有的任务中表现SOA。
    
- ICLR2018 [A NEW METHOD OF REGION EMBEDDING FOR TEXT CLASSIFICATION](https://openreview.net/pdf?id=BkSDMA36Z)

    百度研究院2018的工作。在文本分类中，词袋模型有它的限制。他对单词的表示没有包含单词的顺序信息。N-gram解决了这一问题，但n-gram也有局限: 当n较大时，通常导致数据缺失。在新模型中，region embedding有两部分组成：单词本身的embedding有向量表示，联系上下文的embedding由词向量与上下文关联的矩阵表示。文本看作是region embedding的集合。在文本分类的实验中(主要是Yelp Dataset)大多数实验有提升。
    
> Transfer learning && Contextual (#4)

- [Discourse-Based Objectives for Fast Unsupervised Sentence Representation Learning](https://arxiv.org/pdf/1705.00557.pdf)
本文使用无监督句子学习方式，迁移sentence encoder。设计了3个预训练目标来预训练已有的句子编码器（Sentence encoder），当模型经过这3个预训练后在针对目标任务进行训练
    * 句子顺序是否正确
    * 是否是当前句子的下一句
    * Conjunction Prediction，取具有连接词开头的句子，然后去掉连接词，让模型去预测
三个任务可以一起学习，也可以pipeline的学习。实验在几个简单的分类数据集上证明了在速度上有提升，且在分类acc上有限度的下降。

- NAACL2018 [Contextualized Word Representations for Reading Comprehension](https://arxiv.org/pdf/1712.03609.pdf)

    大多数的Reading Comprehension论文都集中精力在Q和D的interact attention上。文章做了实验，使用Contextual表示的Q和D要在结果中表现的更好。
    
- NAACL2018**BESTPAPER** [Deep contextualized word representations (ELMO)](https://arxiv.org/pdf/1802.05365.pdf)
    过往的word2vec方法是利用词语的共现来训练word embedding，它们大多没有融入上下文的信息。这一次我们利用language model(LM)的方法结合RNN+char embedding训练出来的RNN和char embedding迁移到其他下游任务中都有巨大的提升。这是一种可以利用大规模语料的无监督方法。缺点也显而易见，Contextual没有利用双向的语言信息，同时没考虑词频信息。
    Note: 建议结合论文：Dissecting Contextual Word Embeddings: Architecture and Representation来理解ELMo。

- ICLR2017 [Learned in Translation: Contextualized Word Vectors (CoVe)](https://arxiv.org/pdf/1708.00107.pdf)
    在Image领域，很多问题的解决得益于迁移学习。同样的想法应用在NLP领域。作者认为能理解上下文的模型是可以应用迁移学习的，比如说在机器翻译中。文章使用en2ge的翻译任务训练处encoder然后迁移到其他任务中。把CoVe用到了语义情感分析、问题分类、文本蕴含、问答等多种NLP任务上，这些模型的性能都得到了提升。

> Co-occurance (#6)

- EMNLP2017 [Dict2vec : Learning Word Embeddings using Lexical Dictionaries](https://www.aclweb.org/anthology/D17-1024)
    Word2Vec的训练方式是利用上下文的词共现来训练词。本文提出利用词典的词解释来作为共现语料训练词。文中定义了Strong Pair，意为双方的解释里都出现了对方。然后只出现一方的叫Weak Pair。结合Skip-gram来训练。本篇文章也算是提出了一种新奇的解决思路。
    
- ICLR2019 [CBOW IS NOT ALL YOU NEED: COMBINING CBOW WITH THE COMPOSITIONAL MATRIX SPACE MODEL](https://openreview.net/pdf?id=H1MgjoR9tQ)
    连续词袋模型(CBOW)的一个缺点是没法掌握词序，本文提出一个新的word embedding的初始化方法和新的Loss，结合CBOW方法一起训练。最后的实验结果表明这种训练方法在大多分类和STS任务中都比CBOW要好。同时也能比其他初始化方法记住更多的信息。

- EMNLP2018 [Quantifying Context Overlap for Training Word Embeddings](http://aclweb.org/anthology/D18-1057)
    Word2Vec的训练方式是利用上下文的词共现来训练词，但是忽略了训练上下文词之间的语义相似度。本文在Glove上做出修改，不使用共现的次数作为损失函数，而是使用两个词之间Point-wise Mutual Information (PMI)的交集词的最小PMI值作为损失函数。实验在Word similarity and analogy results都表现的很好，同样几个分类数据集也表现的不错。(对比Glove,SGNS)。
    
- ACL2018 [SemAxis: A Lightweight Framework to Characterize Domain-Specific Word Semantics Beyond Sentiment](https://arxiv.org/pdf/1806.05521.pdf)
    通常，语义会随着上下文语境的变化而变化。本文提出来一种新方法，把词按照近义词来聚类，两个类别之间尽量是反义，每一个类别计算出一个Semantic Axis，计算方式是，求出每个类别的中心点，然后两个类别中心点相减就可以得到一个SemAxis。词的类别是从ConceptNet选出来的729个。然后把word embedding使用cosine计算与每一个轴的相似度，组成一个新的向量。 

- IJCAI2018 [Approximating Word Ranking and Negative Sampling for Word Embedding](https://www.ijcai.org/proceedings/2018/0569.pdf)
    一篇比较好的文章,分析了自己解决问题的思路。作者首先分析了以往的CBOW和SGNS的损失函数只是一个局部最优解。主要集中精力与解决负采样问题，作者把负采样问题看成了word rank问题，实质是一个margin negetive sampling问题，在提出rank之后又遇到大样本梯度爆炸问题。作者都文中所提出来的方法，在Analogy和similarity实验中表现很好。

- IJCAI2018 [Complementary Learning of Word Embeddings](https://www.ijcai.org/proceedings/2018/0607.pdf)
    CBOW和SGNS被认为是两个互补的任务。本文主要使用2 agent的RL,分别去学习CBOW和SGNS，Reward是预测出来的概率log值。实验在文本相似度任务上表现不错。
    
- NAACL2015 [Two/Too Simple Adaptations of Word2Vec for Syntax Problems](http://www.cs.cmu.edu/~lingwang/papers/naacl2015.pdf)

    文章对CBOW和SG模型做了位置相关的修改，在CBOW上，采用stack上下文单词的方法替代SUM。在SG上，在窗口内，采用每个单词单独一个映射矩阵的方法来预测输出。最后的实验结果表明，有一点点的提高在POS和DP上。

- NIPS2014 [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf)

    Omer Levy和Yoav Goldberg的工作，文章对word2vec方法进行了理论性的分析，证明的结论是word2vec实质上是对word-context PMI矩阵的分解。文章的公式还算好懂，[参考文章](http://www.readventurer.com/?p=755#comment-2286㜹)，最近两年有许多利用PMI来改进word2vec方法的。


> Negative sampling
    
- WSDM2016 WordRank: [Learning Word Embeddings via Robust Ranking](https://aclweb.org/anthology/D16-1063)
    本文第一次提出把word2vec的训练作为一种rank方式，使用距离作为rank的指标。思想是通过构造rank指示函数的上界优化目标，同样使用了平滑的技术。文章对比了几种连续的可以作为rank目标的函数，结果DCG表现最好。

- ACL2018 [Batch IS NOT Heavy: Learning Word Representations From All Samples](https://www.comp.nus.edu.sg/~xiangnan/papers/acl18-word-embedding.pdf)
    对于以往的word2vec模型，都是采用负采样做损失函数，负采样的好坏决定了，学习质量。本文避开负样例的选取问题，选取所有的词作为负样例，使用PMI作为label加上回归损失的形式，同时利用数学的trick的方法解决训练时间问题。实验效果比传统的word2vec,glove在word simi和analogy要好。文章把只选取SGNS作为w2v没有CBOW

- WSDM2017 [Improving Negative Sampling for Word Representation using Self-embedded Features](https://arxiv.org/pdf/1710.09805.pdf)

    对于正常的频率负采样方法，不包含语义信息，文章从梯度和实例上证明这样的方法存在缺陷。文章的改进办法是里用取embedding最近的几个词向量来做负样例。同时文章也提出了a fast adaptive and feature-dependent sampling algorithm进行采样。实验结果在word similarity上对比CBOW和SG提升了不少。
    
- ACL 2018 [GNEG: Graph-Based Negative Sampling for word2vec](http://www.aclweb.org/anthology/P18-2090)
    
    本文利用Graph来解决频率负采样无法包含采样语义的问题。通过构建词语共现图，然后random walk来挑选负采样单词。结果在训练时间和sim, analogy任务上都有提升。
    

> Others (#5)

- ICLR2019 [Adaptive Input Representations for Neural Language Modeling](https://arxiv.org/pdf/1809.10853.pdf)

    FB2019的工作。文章主要借鉴adaptive softmax的想法，使用词的词频对词进行聚簇，文中设定5个簇，每个簇的embedding用不同维度去表示。然后对每一个簇设置一个矩阵，目的是把词映射到相同的维度。目的是去对高频的Word embedding进行表示能力的提升，较少低频词的过拟合。实验在BILLION WORD、WIKITEXT-103对比了CharCNN,Subword的模型，有perplexity提升。
    
- ACL2018 [Joint Embedding of Words and Labels for Text Classification](https://arxiv.org/pdf/1805.04174.pdf)
    文章提出来一个把word和label用在一起来表示文本的想法，融合的方法是使用attention。该想法用在分类问题中，在几个分类分体中表现的还不错。

- IJCAI2018 [Think Globally, Embed Locally — Locally Linear Meta-embedding of Words](https://arxiv.org/pdf/1709.06671.pdf)
    不同的word embedding训练方法，训练出来的词向量都各自有优势，本文是想用线性fusion，来融合不同的embedding。其主要思想是：对不同的word embedding训练方法，一个客观的好坏评价是neiberhoods。所以本文认为不同embedding的neiborhoods都应该相同，但是neiborhoods内的单词应该有各自的权重。最后作者通过构造新的损失函数来完成想法。在Semantic similarity， Word analogy detection， Relation classification， Short-text classification表现的都比融入之前的方法好。
    
- axiv2018 [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/pdf/1804.07461.pdf)
    本文介绍了NLP领域的一些基础任务的数据集和task，被称为General Language Understanding Evaluation (GLUE),并且对于一些模型通常会出现的问题做了自己的NLI数据集。在后面实验中做了些基于transfer learning的baseline，主要是elmo-like的实验，同时也在自己提出来的NLI数据集上测试，发现对比人工的结果还相差甚远。
    
- ICLR2019 UNDERSTANDING COMPOSITION OF WORD EMBED- DINGS VIA TENSOR DECOMPOSITION
**难**

- ACL 2017 [Skip-Gram – Zipf + Uniform = Vector Additivity Alex](http://www.aclweb.org/anthology/P17-1007)
    
    本文实际上证明的是单词分布在满足Uniform分布时用Skip-Gram模型训练出来的词向量具有Vector Additivity的性质(比如“man” - “woman” = “king” - “queen”)。而实际上我们的单词是满足Zipf分布的，所以这篇文章的证明其实和实际是不符合的，但是确实解释了词向量的一些性质。
词向量作为Deep Learning for NLP的基石已经是广为人熟知了，它的本质是认为如果两个词的上下文是一样的，那么可以认为这两个词是一样的。Mikolov在13年提出word2vec的时候写出了惊艳的等式“man” - “woman” = “king” - “queen”，也就是说将两个词向量相加得到一个向量，与这个向量最接近的词向量所代表的单词的意思往往是这个两个词意思的组合。这是一个很有意义的结论，也就是词向量将语义映射到了欧式空间。这样我们就可以用欧式空间的度量来衡量语义。在Standford CS224n课程中就有一个小插曲。如下图所示这里的tie可能表示球赛的平局，也可能表示领带，还可能表示绳子打结。那它的词向量究竟在哪里呢？虽然相似的词被映射到邻近的位置，但该论文证明词向量是所有义项的平均。


### Sentence representation (#1)

> Context Information (#1)

- EMNLP2017 [A Deep Neural Network Sentence Level Classification Method with Context Information](https://arxiv.org/pdf/1809.00934.pdf)

    文章主要使用了LSTM+CNN的结构，同时使用FOFE编码上下文句子来辅助句子分类任务。 这里面对的问题是，对于句子分类任务，文中把这个句子称作是Focus，其他的句子分Left、Right。 因为句子分类任务是没有上下文的，实验中作者是通过查找数据集来源，来补充上下文句子。 文章的实验主要证明了FOFE or Context信息有用。
    
> Multi-task learning

- ICML2008 [A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning](https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf)
    
    本文是第一个用DNN的方法训练多任务学习，将language model, pos-tag, NER, chunk, SRL（Semantic Role Labeling ）统一到一个框架中，最终利用language model, pos-tag, NER, chunk等任务中自动学习的feature来提升SRL的性能，实验结果与state-of-the-art相当。

- Arxiv 2018 [Multi-Task Deep Neural Networks for Natural Language Understanding Xiaodong](https://arxiv.org/pdf/1901.11504.pdf)
    本文是MSRA的文章，没有什么特别的创新点，利用了BERT的预训练，加上分类任务，文本相似性任务，句子对分类任务，做多任务学习。在多数任务中表现良好。
    
- Arxiv 2018 [Discourse-Based Objectives for Fast Unsupervised Sentence Representation Learning Yacine](https://arxiv.org/pdf/1705.00557.pdf)

    本文通过设置三个句子无监督任务来进行预训练：1.句子顺序是否正确。2. 是否是当前句子的下一句。3.预测句子之间的连接词。模型没有创新，实验在文本分类中表现良好。训练时间更少些。



### Exploration experiments (#1)

- ICLR2019 [LOOKING FOR ELMO’S FRIENDS: SENTENCE-LEVEL PRETRAINING BEYOND LANGUAGE MODELING](https://arxiv.org/pdf/1812.10860.pdf)
    用于自然语言处理任务（如翻译、问答和解析）的最先进模型都具有旨在提取每个输入句子含义和内容表征的组件。这些句子编码器组件通常直接针对目标任务进行训练。这种方法可以在数据丰富的任务上发挥作用，并在一些狭义定义的基准上达到人类水平，但它只适用于少数具有数百万训练数据样本的 NLP 任务。这引起人们对预训练句子编码的兴趣：我们有充分的理由相信，可以利用外部数据和训练信号来有效地预训练这些编码器，因为它们主要用于捕获句子含义而不是任何特定于任务的技能。并且我们已经看到了预训练方法在词嵌入和图像识别相关领域中获得的成功。

    更具体地说，最近的四篇论文表明，预训练句子编码器可以在 NLP 任务上获得非常强的性能。首先，McCann 等人 (2017) 表明来自神经机器翻译系统的 BiLSTM 编码器可以在其他地方有效地重用。Howard & Ruder (2018)、Peters 等 (2018)、 Radford 等 (2018) 表明，通过生成式语言建模（LM）以无监督方式预训练的各种编码器也是有效的。然而，每篇论文都使用自己的评估方法，不清楚哪个预训练任务最有效，或者是否可以有效地组合多个预训练任务；在句子到向量编码的相关设置中，使用多个标注数据集的多任务学习已经产生了鲁棒的当前最佳结果。

    本文试图系统地解决这些问题。研究者在 17 种不同的预训练任务、几个简单的基线以及这些任务的几种组合上训练可重用的句子编码器，所有这些都使用受 ELMo 扩展的单个模型架构和过程，用于预训练和迁移。然后，研究者根据 GLUE 基准测试中的 9 个目标语言理解任务评估这些编码器，他们共得到了 40 个句子编码器和 360 个已训练模型。然后，研究者测量目标任务的性能相关性，并绘制了评估训练数据量对每个预训练和目标任务的影响的学习曲线。

    实验结果表明语言建模是其中最有效的一个预训练任务，预训练期间的多任务学习可以提供进一步的增益，并在固定句子编码器上得到新的当前最佳结果。然而，ELMo 式的预训练也有令人担忧的地方，研究者预训练模型并将其用于目标任务时没有进一步微调，这是脆弱的并且存在严重限制： (i) 一般的基线表征和最好的预训练编码器几乎能达到相同的表现，不同的预训练任务之间的差别可能非常小。(ii) 不同的目标任务在它们受益最多的预训练方面存在显著差异，并且多任务预训练不足以避免这个问题并提供通用的预训练编码器。
**这篇文章摘自机器之心**


### RNN module (or some else like VAE)


## 2. 理论与综述文章(Theoretical and survey papers)

- [Word Embeddings: A Survey](https://arxiv.org/pdf/1901.09069.pdf)
    
    这篇综述把word embedding从两个解决问题的方面来分，两个方面又分了主要问题和解决方式
    1. 基于预测的模型
        1. 问题：训练时间长
            1. negative sampling
            2. hierachical softmax
            3. 前后位置的单词提升收敛速度
        2. 问题：频率问题
    2. 基于count共现矩阵的模型
        1. 问题：频率问题
    文章发在了arxiv上，文章只写了10页，还有很多工作没有总结在上面。

## 3. 相关代码 (Available codes)

## 4. 文本表示领域代表性研究学者 (Scholars)
**word2vec 作者 Tomas Mikolov**

    1、Efficient Estimation of Word Representation in Vector Space, 2013
    
    2、Distributed Representations of Sentences and Documents, 2014
    
    3、Enriching Word Vectors with Subword Information, 2016

**[Process in NLP](https://nlpprogress.com/) 系统作者 Sebastian Ruder**


## 5. 相关的硕博士论文 (Thesis)

## 6. 常用公开数据集及算法结果 (Datasets and benchmark)

## 7. 比赛 (Challenges and competitions)

### In Process
1. Google AI of Kaggle: [Gendered Pronoun Resolution](https://www.kaggle.com/c/gendered-pronoun-resolution)
        Timeline: April 8, 2019 - Entry deadline. You must accept the competition rules before this date in order to compete.
    
    April 8, 2019 - Team Merger deadline. This is the last day participants may join or merge teams.
    
    April 15, 2019 - Stage 1 Submission & Model upload deadline*
    
    April 16, 2019 - Stage 2 begins. New test set uploaded.
    
    April 22, 2019 - Stage 2 ends & Final submission deadline. Stage 2 final submissions selection must be completed by this date.
    
    May 3, 2019 - Description Paper Submission deadline.** Winners Obligations deadline.
    
    May 24, 2019 - Description Paper Reviews and Acceptance announcements.
    
    June 7, 2019 - Camera-ready Papers deadline.
    
    August 2, 2019 - ACL (Association for Computational Linguistics) 2019 Workshop on Gender Bias in Natural Language Processing.
    代词性别消歧任务，给一个句子，包含两个实体名称一个代词，区别这个代词所指向的实体。

### Over


### To Go
1. NLPCC (TBD)
2. [2019语言与智能技术竞赛---百度&CCF](http://lic2019.ccf.org.cn/)

    Registration Begin: **Now**
    Time Begin: **April 1**
    本届比赛设立了三个任务：
    1. 机器阅读理解任务：让机器阅读文本然后回答和阅读内容相关的问题， 阅读理解旨在使机器具备理解自然语言的能力。
    2. 知识驱动对话：知识驱动对话是一种人机对话任务，机器根据知识信息构建的图谱进行主动聊天， 旨在使机器具备模拟人类用语言进行信息传递的能力。
    3. 信息抽取任务：信息抽取是指从自然语言文本中抽取实体、属性、关系等信息， 旨在使机器具备从海量文本自动构建知识的能力。

## 8. 其他 (Miscellaneous)
[Transfer learning with language model 20181031](https://drive.google.com/file/d/1kmNAwrSlFYo0cN_DcURMOArBwe9FxWxR/view)：Ruder在一次meetup上面讲的关于LM的内容，包括对现在LM的分析和未来的展望。





