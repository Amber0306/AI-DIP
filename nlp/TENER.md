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

>The named entity recognition (NER) is the task of
finding the start and end of an entity in a sentence
and assigning a class for this entity. NER has been
widely studied in the field of natural language processing
(NLP) because of its potential assistance
in question generation (Zhou et al., 2017), relation
extraction (Miwa and Bansal, 2016), and coreference
resolution (Fragkou, 2017). Since (Collobert
et al., 2011), various neural models have been introduced
to avoid hand-crafted features (Huang
et al., 2015; Ma and Hovy, 2016; Lample et al.,
2016).

>NER is usually viewed as a sequence labeling
task, the neural models usually contain three components:
word embedding layer, context encoder
layer, and decoder layer (Huang et al., 2015; Ma and Hovy, 2016; Lample et al., 2016; Chiu and
Nichols, 2016; Chen et al., 2019; Zhang et al.,
2018; Gui et al., 2019b). The difference between
various NER models mainly lies in the variance in
these components.

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

>Recently, Transformer (Vaswani et al., 2017)
began to prevail in various NLP tasks, like machine
translation (Vaswani et al., 2017), language
modeling (Radford et al., 2018), and pretraining
models (Devlin et al., 2018). The Transformer
encoder adopts a fully-connected self-attention
structure to model the long-range context, which
is the weakness of RNNs. Moreover, Transformer
has better parallelism ability than RNNs. However,
in the NER task, Transformer encoder has
been reported to perform poorly (Guo et al., 2019),
our experiments also confirm this result. Therefore,
it is intriguing to explore the reason why
Transformer does not work well in NER task.

>In this paper, we analyze the properties of
Transformer and propose two specific improvements
for NER.

>The first is that the sinusoidal position embedding
used in the vanilla Transformer is aware of
distance but unaware of the directionality. In addition,
this property will lose when used in the vanilla Transformer. However, both the direction
and distance information are important in the NER
task. For example in Fig 1, words after “in” are
more likely to be a location or time than words before
it, and words before “Inc.” are mostly likely
to be of the entity type “ORG”. Besides, an entity
is a continuous span of words. Therefore, the
awareness of distance might help the word better
recognizes its neighbor. To endow the Transformer
with the ability of direction- and distanceawareness,
we adopt the relative positional encoding
(Shaw et al., 2018; Huang et al., 2019; Dai
et al., 2019). instead of the absolute position encoding.
We propose a revised relative positional
encoding that uses fewer parameters and performs
better.

>The second is an empirical finding. The attention
distribution of the vanilla Transformer is
scaled and smooth. But for NER, a sparse attention
is suitable since not all words are necessary
to be attended. Given a current word, a few contextual
words are enough to judge its label. The
smooth attention could include some noisy information.
Therefore, we abandon the scale factor of
dot-production attention and use an un-scaled and
sharp attention.

>With the above improvements, we can greatly
boost the performance of Transformer encoder for
NER.

>Other than only using Transformer to model
the word-level context, we also tried to apply it
as a character encoder to model word representation
with character-level information. The previous
work has proved that character encoder is
necessary to capture the character-level features
and alleviate the out-of-vocabulary (OOV) problem
(Lample et al., 2016; Ma and Hovy, 2016;
Chiu and Nichols, 2016; Xin et al., 2018). In NER,
CNN is commonly used as the character encoder.
However, we argue that CNN is also not perfect
for representing character-level information, because because
the receptive field of CNN is limited, and the
kernel size of the CNN character encoder is usually
3, which means it cannot correctly recognize
2-gram or 4-gram patterns. Although we can deliberately
design different kernels, CNN still cannot
solve patterns with discontinuous characters,
such as “un..ily” in “unhappily” and “unnecessarily”.
Instead, the Transformer-based character encoder
shall not only fully make use of the concurrence
power of GPUs, but also have the potentiality
to recognize different n-grams and even discontinuous
patterns. Therefore, in this paper, we also
try to use Transformer as the character encoder,
and we compare four kinds of character encoders.

>In summary, to improve the performance of the
Transformer-based model in the NER task, we explicitly
utilize the directional relative positional
encoding, reduce the number of parameters and
sharp the attention distribution. After the adaptation,
the performance raises a lot, making our
model even performs better than BiLSTM based
models. Furthermore, in the six NER datasets, we
achieve state-of-the-art performance among models
without considering the pre-trained language
models or designed features.


## 2. important things
