# TENER: Adapting Transformer Encoder for Named Entity Recognition
## 0. intro
- author: Hang Yan,Bocao Deng, Xiaonan Li, Xipeng Qiu
- organization: School of Computer Science, Fudan University,
Shanghai Key Laboratory of Intelligent Information Processing, Fudan University

## 1. translation and interpretion of n.
### 1.1 abstract
> Bidirectional long short-term memory networks
(BiLSTMs) have been widely used
as an encoder for named entity recognition
(NER) task.

> Recently, the fully-connected
self-attention architecture (aka Transformer) is
broadly adopted in various natural language
processing (NLP) tasks owing to its parallelism
and advantage in modeling the long range
context. 

Transformer在nlp的任务中如今很常用

> Nevertheless, the performance
of the ***vanilla*** Transformer in NER is not as
good as it is in other NLP tasks. In this
paper, we propose TENER, a NER architecture
***adopting adapted Transformer Encoder*** to
model the character-level features and wordlevel
features. 

vanilla: 香草的，普通的，香草

普通的tranformer不好用，在ner中表现不太好，所以提出了一种新的transformer结构去模拟字符和单词级别的特征。


> By incorporating the directionaware,
distance-aware and un-scaled attention,
we prove the Transformer-like encoder is
just as effective for NER as other NLP tasks.
Experiments on six NER datasets show that
TENER achieves superior performance than
the prevailing BiLSTM-based models.

方向感知、距离感知和未缩放的attention机制结合，类似transformer的编码器还是有效的。

### 1.2 introduction

> ***The named entity recognition (NER) is the task of
finding the start and end of an entity in a sentence
and assigning a class for this entity.*** NER has been
widely studied in the field of natural language processing
(NLP) because of its potential assistance
in question generation (Zhou et al., 2017), relation
extraction (Miwa and Bansal, 2016), and coreference
resolution (Fragkou, 2017). Since (Collobert
et al., 2011), various neural models have been introduced
to avoid hand-crafted features (Huang
et al., 2015; Ma and Hovy, 2016; Lample et al.,
2016).

NER是在句子中找到一个实体的首尾并为这个实体标记类别的任务。由于NER在问题生成、关系析取，指代消解中有潜在的帮助，因此它在自然语言处理领域中被广泛研究。自2011年不同的神经网络模型被应用在该任务中，以减少人工特征的使用。

>NER is usually viewed as a sequence labeling task, the neural models usually contain three components:
***word embedding layer, context encoder layer, and decoder layer*** (Huang et al., 2015; Ma and Hovy, 2016; Lample et al., 2016; Chiu and Nichols, 2016; Chen et al., 2019; Zhang et al., 2018; Gui et al., 2019b). 
The difference between various NER models mainly lies in the variance in these components.

NER经常被看作序列标记任务，这里神经模型通常包括以下几个部分：词嵌入层，上下文编码层，以及解码层。不同的NER模型主要是这几个组成部分的不同。


>Recurrent Neural Networks (RNNs) are widely
employed in NLP tasks due to its sequential characteristic,
which is aligned well with language.
Specifically, bidirectional long short-term memory
networks (BiLSTM) (Hochreiter and Schmidhuber, 1997) is one of the most widely used RNN
structures. (Huang et al., 2015) was the first one
to apply the BiLSTM and Conditional Random
Fields (CRF) (Lafferty et al., 2001) to sequence
labeling tasks. Owing to BiLSTM’s high power to
learn the contextual representation of words, it has
been adopted by the majority of NER models as
the encoder (Ma and Hovy, 2016; Lample et al.,
2016; Zhang et al., 2018; Gui et al., 2019b).

RNN由于其序列特征在NLP任务中得到广泛应用，这与语言的使用很一致。具体地，BiLSTM 双向长短期记忆神经元就是其中广泛应用的一种RNN网络结构。Huang第一次应用了双向长短期记忆网络和条件随机场在序列标记任务中，由于长短期记忆网络能够学习代表前后逻辑关系的单词，它也已经被大部分的NER模型当做编码器使用。


>Recently, Transformer (Vaswani et al., 2017) began to prevail in various NLP tasks, like machine translation (Vaswani et al., 2017), language modeling (Radford et al., 2018), and pretraining models (Devlin et al., 2018). 
The Transformer encoder ***adopts a fully-connected self-attention structure to model the long-range context***, which is the weakness of RNNs. 
Moreover, Transformer has better parallelism ability than RNNs. 
However, in the NER task, Transformer encoder has been reported to perform poorly (Guo et al., 2019), our experiments also confirm this result. 
Therefore, it is intriguing to explore the reason why Transformer does not work well in NER task.

近期，transformer开始不同的NLP应用中占优势，像机器翻译、语言建模和预训练模型。
transformer编码器使用了一个全连接的self attention结构去模拟长文本，这正是rnn的弱项。
此外，transformer比RNN也有更好的并行能力。然而，在NER任务中，Transformer编码器表现较差。
因此，探索transformer编码器在NER任务中表现不佳的原因也是很令人感兴趣的。

>In this paper, we analyze the properties of
Transformer and propose two specific improvements
for NER.

在这篇论文中，我们分析了Transformer的特征，并依据NER任务提出了两种特定的改进。

>The first is that the sinusoidal position embedding used in the vanilla Transformer is aware of distance but unaware of the directionality.
In addition, this property will lose when used in the vanilla Transformer. 
However, both the direction and distance information are important in the NER task.
For example in Fig 1, words after “in” are more likely to be a location or time than words before it, and words before “Inc.” are mostly likely to be of the entity type “ORG”. Besides, an entity is a continuous span of words.
Therefore, the awareness of distance might help the word better recognizes its neighbor. 
To endow the Transformer with the ability of direction- and distanceawareness,
***we adopt the relative positional encoding (Shaw et al., 2018; Huang et al., 2019; Dai et al., 2019). instead of the absolute position encoding.***
We propose a revised relative positional encoding that uses fewer parameters and performs better.

首先是在普通Transformer中使用的正弦曲线位置嵌入，它对位置敏感但方向不敏感。
此外，在被普通Transformer使用的时候，这种特性会丢失。
但是，方向和距离的信息在NER任务中都很重要。
例如，举例子。in后面很可能是地点或者时间而不是单词，之前的单词inc很有可能是实体类型ORG.
此外，实体还可能是连续的单词。
因此，距离感知能够帮助一个单词更好地识别出它的近邻。
为了赋予这个trransformer方向和距离敏感的能力，我们采取了一种相对位置编码来代替绝对位置编码。
我们提出了一种改进的相对位置编码，它使用更少的参数并且表现更好。


>The second is an empirical finding. 
The attention distribution of the vanilla Transformer is scaled and smooth. 
But for NER, a sparse attention is suitable since not all words are necessary to be attended. 
Given a current word, a few contextual words are enough to judge its label. 
The smooth attention could include some noisy information.
Therefore, ***we abandon the scale factor of dot-production attention and use an un-scaled and sharp attention.***

第二点则是实验发现。
普通transformer的attention分布是有比例且平滑的。
但是对于NER人物，一个稀疏的attention分布可能会更合适，因为并不是所有单词都有必要被看到。
给出一个当前的单词，一些前后关系词，对判断它的标签而言已经足够。
平滑的attention可能会包含很多的噪声信息。
因此，我们抛弃了点产品attention的比例因此，并使用一个未缩放且锋利的attention。

>With the above improvements, we can greatly boost the performance of Transformer encoder for NER.

通过上面的改进，我们极大地提升了Transformer编码器在NER任务上的表现效果。

>Other than only using Transformer to model the word-level context, we also tried to apply it as a character encoder to model word representation with character-level information. 
The previous work has proved that character encoder is necessary to capture the character-level features and alleviate the out-of-vocabulary (OOV) problem (Lample et al., 2016;
Ma and Hovy, 2016; Chiu and Nichols, 2016; Xin et al., 2018). 
In NER, CNN is commonly used as the character encoder.
However, we argue that CNN is also not perfect for representing character-level information, 
because the receptive field of CNN is limited, and the kernel size of the CNN character encoder is usually 3, which means it cannot correctly recognize 2-gram or 4-gram patterns. 
Although we can deliberately design different kernels, ***CNN still cannot solve patterns with discontinuous characters***, such as “un..ily” in “unhappily” and “unnecessarily”.
Instead, ***the Transformer-based character encoder shall not only fully make use of the concurrence power of GPUs, but also have the potentiality to recognize different n-grams and even discontinuous patterns. ***
Therefore, in this paper, we also try to use Transformer as the character encoder, and we compare four kinds of character encoders.

**该段介绍字符编码器**

除了使用Transformer来模拟单词级别的上下文，我们同样将它应用在字符级别的特征以及缓和词汇量过大的问题。
在NER任务重，CNN通常作为字符编码器使用。
然而，我们主张CNN在代表字符级别的信息时仍然不完美，因为CNN的感受野是受限的，CNN字符编码器的内核大小通常是3，这意味着，它不能准确的识别2元或者4元模型。
尽管我们能够刻意设计出不同大小的内核，但CNN仍然不能用不连续字符解决模型问题，例如。。。。。。。。。
取而代之的是，基于transformer的字符编码器可能不仅重复利用了GPU的合并算力，同时也有识别出n元甚至是不连续模型的n元问题的潜力。
因此，在这篇文章中，我们尝试使用Transformer作为字符编码器，并且我们比较四种不同的字符编码器。

>In summary, to improve the performance of the Transformer-based model in the NER task, we explicitly utilize the directional relative positional encoding, reduce the number of parameters and sharp the attention distribution. 
After the adaptation, the performance raises a lot, making our model even performs better than BiLSTM based models. 
Furthermore, in the six NER datasets, we achieve state-of-the-art performance among models without considering the pre-trained language models or designed features.

总之，为了提高基于transformer的模型在NER任务中的表现，我们明确使用了方向相对位置编码，减少参数数量和锐化attention的分布。
在改良后，网络效果提升了很多，这使我们的模型比基于BiLSTM的模型效果好很多。
而且，在六个NER的数据集上，在没有考虑预训练语言模型和设计特征的情况下，我们取得了极好的效果。

## 1.2 Related Work

### 1.2.1 Neural Architecture for NER

>Collobert et al. (2011) utilized the Multi-Layer Perceptron (MLP) and CNN to avoid using task specific features to tackle different sequence labeling tasks, such as Chunking, Part-of-Speech (POS) and NER. 
In (Huang et al., 2015), BiLSTM-CRF was introduced to solve sequence labeling questions. Since then, the BiLSTM has been extensively used in the field of NER (Chiu and Nichols, 2016; Dong et al., 2016; Yang et al., 2018; Ma and Hovy, 2016).

C使用多层感知机和CNN来避免使用特定特征处理不同的序列标记任务，例如分块、词性分析和NER。
2015，BiLSTM-CRF被用在解决序列标记问题上，从那时起，BiLSTM已经在NER领域内被广泛应用。

>Despite BiLSTM’s great success in the NER task, it has to compute token representations one by one, which massively hinders full exploitation of GPU’s parallelism. 
Therefore, CNN has been proposed by (Strubell et al., 2017; Gui et al., 2019a) to encode words concurrently. 
In order to enlarge the receptive field of CNNs, (Strubell et al., 2017) used iterative dilated CNNs (IDCNN).

虽然BiLSTM在NER任务中取得了巨大的成功，但它需要计算一个个地计算令牌表示，这极大地拖慢了GPU的利用效率。
因此，CNN被提议同时编码单词，为了扩大CNN的感受野，S使用膨胀CNN

>Since the word shape information, such as the capitalization and n-gram, is important in recognizing named entities, 
CNN and BiLSTM have been used to extract character-level information (Chiu and Nichols, 2016; Lample et al., 2016; Ma and Hovy, 2016; Strubell et al., 2017; Chen et al., 2019).

由于词类型信息，例如n元和大写，在识别命名实体中是很重要的，CNN和BiLSTM都被用来提取字符级别的信息。

>Almost all neural-based NER models used pretrained word embeddings, like Word2vec and Glove (Pennington et al., 2014; Mikolov et al., 2013). 
And when contextual word embeddings are combined, the performance of NER models will boost a lot (Peters et al., 2017, 2018; Akbik et al., 2018). 
ELMo introduced by (Peters et al., 2018) used the CNN character encoder and BiLSTM languagemodels to get contextualized word representations.
Except for the BiLSTM based pre-trained models, BERT was based on Transformer (Devlin et al., 2018).

几乎所有基于神经元的NER模型都是用预训练的词嵌入，例如word2vec和Glove.
并且当上下文的词嵌入被结合起来，NER模型的表现会提高很多。
P介绍了Elmo，使用CNN字符编码器和BiLSTM语言模型去获得上下文相关的词表示。
除了基于BiLSTM的预训练模型，BERT也基于Transformer。

### 1.2.2 Transformer


## 2. important things

### 2.1

提出了两点改进

1. 相对位置编码的transformer
2. 通过实验发现得到稀疏且锐利的attention更好用。

同时改进了字符编码器。