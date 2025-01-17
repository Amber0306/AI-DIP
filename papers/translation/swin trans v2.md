# 0.摘要

大规模的自然语言处理模式被证明在没有饱和迹象的情况下显著提高了语言任务的绩效。它们还表现出像人类一样令人惊叹的few-shot能力。本文旨在探索计算机视觉中的大规模模型。

我们解决了大视觉模型训练和应用中的三个主要问题，包括训练的不稳定性，预训练和微调之间的分辨率差距，以及对标记数据的饥饿。

提出了三种主要的训练方法：

1. 结合余弦注意的残差后范数训练方法，以提高训练的稳定性；
2. 对数间隔连续位置偏移法，有效地将低分辨率图像的预训练模型转移到高分辨率输入的下游任务；
3. 自监督预训练方法，SimMIM，以减少对大量标记图像的需求。

通过这些技术，本文成功地训练了一个30亿参数的Swin Transformer V2模型，这是迄今为止最大的密集视觉模型，并使其能够对分辨率高达1,536×1,536的图像进行训练。它在4个具有代表性的视觉任务上创造了新的性能纪录，包括ImageNet-V2图像分类、COCO目标检测、ADE20K语义分割和Kinetics-400视频动作分类。还要注意的是，我们的训练比谷歌10亿级视觉模型的训练效率高得多，后者消耗的标签数据少40倍，训练时间少40倍。代码可以在微软的https://github.com//Swin-Transformer上找到。

> few-shot learning 小样本学习

# 1. introduction

扩大语言模型已经取得了令人难以置信的成功。它显著提高了模型在语言任务[19，24，49，50，52，53]上的表现，并且该模型展示了与人类相似的惊人的少发能力[7]。自从拥有3.4亿个参数的BERT大型模型[19]以来，语言模型在几年内迅速扩大了1000多倍，达到5300亿个密集参数[50]和1.6万亿个稀疏参数[24]。这些大型语言模型还被发现具有越来越强的少射击能力，类似于人类在广泛的语言任务中的智力[7]。

另一方面，视觉模型的放大一直落后于人。虽然人们早就认识到，较大的视觉模型通常在视觉任务中表现更好[29，60]，但绝对模型大小最近才能达到约10-20亿个参数[17，27，39，56，80]。更重要的是，与大语言模型不同，现有的大视觉模型仅适用于图像分类任务[17，56，80]。

为了成功地训练大型通用视觉模型，我们需要解决几个关键问题。

首先，我们对大型视觉模型的实验揭示了训练中的不稳定性问题。我们发现，在大型模型中，跨层激活幅度的差异变得显著更大。仔细观察原始架构，会发现这是由直接添加回主分支的剩余单元的输出引起的。结果是激活值逐层累积，因此较深层的振幅明显大于早期层的振幅。

为了解决这个问题，我们提出了一种新的标准化配置，称为res-post-norm，它将LN层从每个残差单元的开始移动到后端，如图1所示。我们发现这种新的配置在网络层产生的激活值要温和得多。我们还提出了一个比例余弦注意来取代以前的点积注意。比例余弦注意使得计算与块输入的幅度无关，并且注意值不太可能陷入极端。在我们的实验中，提出的两种技术不仅使训练过程更加稳定，而且提高了精度，特别是对于较大的模型。

其次，许多下游视觉任务，如对象检测和语义分割，需要高分辨率的输入图像或大的注意窗口。低分辨率预训练和高分辨率微调之间的窗口大小变化可能相当大。当前常见的做法是对位置偏差图进行双三次插值[22，46]。这种简单的修复有点特别，结果通常是次优的。我们引入了对数间隔连续位置偏差(对数CPB ),它通过在对数间隔坐标输入上应用一个小的元网络来产生任意坐标范围的偏差值。由于元网络采用任何坐标，预训练的模型将能够通过共享元网络的权重来自由地跨窗口大小转移。我们方法的一个关键设计是将坐标转换到对数空间，使得即使当目标窗口大小明显大于预训练时，外推比率也可以很低。模型容量和分辨率的扩大还会导致现有视觉模型的GPU内存消耗过高。为了解决内存问题，我们结合了几个重要的技术，包括零优化器[54]，激活检查点[12]和顺序自我注意计算的新实现。通过这些技术，大型模型和分辨率的GPU内存消耗显著降低，而对训练速度的影响甚微。

利用上述技术，我们成功训练了一个30亿Swin的Transformer模型，并使用Nvidia A100-40G GPU有效地将其转移到图像分辨率高达1536×1536的各种视觉任务中。在我们的模型预训练中，我们还采用自我监督预训练来减少对超大标记数据的依赖。与以前的实践(JFT3B)相比，标记数据减少了40倍，30亿模型在广泛的视觉基准上实现了最先进的精度。具体来说，它在ImageNet-V2图像分类验证集[55]上获得了84.0%的顶级准确性，在COCO对象检测测试开发集上获得了63.1 / 54.4框/掩模AP，在ADE20K语义分割上获得了59.9 mIoU，在Kinetics-400视频动作分类上获得了86.8%的顶级准确性，比原始Swin Transformers中的最佳数字高了+NA%、+4.4/+3.3、+6.3和+1.9[46，400]

通过扩大视觉模型的容量和分辨率，使其在一般视觉任务中表现出色，就像一个好的语言模型在一般NLP任务中的表现一样，我们的目标是在这个方向上刺激更多的研究，以便我们最终可以弥合视觉和语言模型之间的容量差距，并促进这两个领域的联合建模。

# 2. Related Work

### Language networks and scaling up

自[65]的开创性工作以来，Transformer一直服务于标准网络.

对这种架构进行扩展的探索已经开始，有效的自我监督学习方法的发明加速了这一进程，如掩蔽或自回归语言建模[19，52]，扩展法则的发现进一步推动了这一进程[36]

从那时起，语言模型的容量在几年内急剧增加了1000多倍，从BERT-340M到威震天-图灵-530B [7，49，50，53]和稀疏开关-变压器-1.6T [24]。随着容量的增加，各种语言基准的准确性得到了显著提高。零发或少发成绩也显著提高[7]，这是人类一般智能的一个基础。

### Vision networks and scaling up

CNN长期以来一直是标准的计算机视觉网络[40，41]。自AlexNet [40]以来，架构变得更加深入和庞大，这极大地推进了各种视觉任务，并在很大程度上推动了计算机视觉中深度学习的浪潮，如VGG [60]，谷歌网[62]和ResNet citehe2015resnet

在过去的两年中，CNN架构已经进一步扩大到大约10亿个参数[27，39]，然而，绝对性能可能并不令人鼓舞，这可能是由于CNN架构中的感应偏差限制了建模能力。

> inductive bias感应偏差？

去年变形金刚开始陆续接手一个又一个具有代表性的视觉基准，包括ImageNet1K图像级分类基准[22]、COCO区域级物体检测基准[46]、ADE20K像素级语义分割基准[46、83]、Kinetics-400视频动作分类基准[2]等

自从这些工作以来，已经提出了许多视觉Transformer变体来在相对小的范围内提高精度[14，21，34，42，63，68，71，75，77，78，82]。只有少数作品尝试放大视觉变形金刚[17，56，80]。然而，它们依赖于具有分类标签的巨大图像数据集，即JFT-3B，并且仅应用于图像分类问题。

### Transferring across window / kernel resolution

对于CNN，以前的工作通常在预训练和微调期间固定内核大小。全局视觉转换器，如ViT [22]，全局计算注意力，等效的注意力窗口大小与增加的输入图像分辨率成线性比例。对于局部视觉转换器架构，如Swin Transformer [46]，窗口大小可以是固定的，也可以在微调期间改变。允许可变的窗口大小在使用中更方便，以便被可能可变的整个特征图整除，并调整感受野以获得更好的准确性。为了处理预训练和微调之间的可变窗口大小，双三次插值是以前的常见做法[22，46]。在本文中，我们提出了一种对数间隔连续位置偏差方法(Log-CPB)，该方法在低分辨率下更平滑地转移预训练的模型权重，以处理更高分辨率的窗口。

### Study on bias terms

在NLP中，相对位置偏差方法被证明是有益的[53]，与原始变换器中使用的绝对位置嵌入相比[65]

在计算机视觉中，相对位置偏差方法更常用[31，46，75]，这可能是因为视觉信号的空间关系在视觉建模中起着更重要的作用。通常的做法是直接学习偏差值作为模型权重。也有一些著作专门研究如何设置和学习偏差术语[38，69]。

### Continuous convolution and variants

我们的对数CPB方法也与连续卷积和变体的早期工作相关[30，45，58，67]，其利用元网络来处理不规则数据点。我们的对数CPB方法受到了这些努力的启发，同时解决了一个不同的问题，即在任意窗口大小之间传递视觉变压器中的相对位置偏差。我们还提出了对数空间坐标，以减轻在大尺寸变化之间转换时外推的困难。

# 3. Swin Transformer V2

## 3.1 A brief review of swin transformer

Swin Transformer是一款通用计算机视觉主干，在各种粒度识别任务中取得了强大的性能，如区域级对象检测、像素级语义分割和图像级图像分类。Swin Transformer的主要思想是在vanilla Transformer编码器中引入几个重要的视觉先验，包括层次、局部性和平移不变性，它结合了两者的优势:基本的Transformer单元具有强大的建模能力，视觉先验使其对各种视觉任务友好。

### Normalization configuration

众所周知，标准化技术[3，35，64，70]对于稳定训练更深层次的架构至关重要。最初的Swin转换器继承了语言转换器[52]和vanilla ViT [22]中的惯例，在没有深入研究的情况下使用了预规范化配置，如图1所示。在下面的小节中，我们将研究这种默认的规范化配置1。

### Relative position bias

ta是原始Swin转换器中的一个关键组件，它引入了一个附加的参数偏差项来编码自关注计算中的几何关系:

Attention(Q, K, V ) = SoftMax(QKT /√d + B)V,

其中B ∈ RM 2×M 2是每个磁头的相对位置偏差项；q，K，V ∈ RM 2×d是查询、键和值矩阵；d是查询/键维度，M 2是一个窗口中的面片数。相对位置偏差编码视觉元素的相对空间配置，并在各种视觉任务中表现出关键作用，特别是对于密集识别任务，如物体检测。

在Swin变压器中，沿各轴的相对位置在[M+1，M-1]范围内，相对位置偏置参数化为偏置矩阵B∈R(2M 1)×(2M 1)，B中的元素取自B

当在不同的窗口尺寸之间转换时，在预训练中学习的相对位置偏差矩阵被用于在通过双三次插值的微调中初始化不同尺寸的偏差矩阵。

> 双三次插值又称立方卷积插值。三次卷积插值是一种更加复杂的插值方式。该算法利用待[采样](https://so.csdn.net/so/search?q=采样&spm=1001.2101.3001.7020)点周围16个点的灰度值作三次插值，不仅考虑到4  个直接相邻点的灰度影响，而且考虑到各邻点间灰度值变化率的影响。三次运算可以得到更接近高分辨率图像的放大效果，但也导致了运算量的急剧增加。这种算法需要选取插值基函数来拟合数据，其最常用的插值基函数所示
> $$
> W(x)= \begin{cases}(a+2)|x|^{3}-(a+3)|x|^{2}+1 & \text { for }|x| \leq 1 \\ a|x|^{3}-5 a|x|^{2}+8 a|x|-4 a & \text { for } 1<|x|<2 \\ 0 & \text { otherwise }\end{cases}
> $$

### Issues in scaling up model capacity and window resolution

当我们提高Swin变压器的容量和窗口分辨率时，我们观察到两个问题。

#### 扩大模型容量时的不稳定性问题

如图2所示，当我们将原始Swin变压器模型从小尺寸放大到大尺寸时，更深层的激活值会显著增加。具有最高和最低振幅的层之间的差异已经达到104的极值。当我们将其进一步放大到巨大的规模(6.58亿个参数)时，它无法完成训练，如图3所示。

#### 跨窗口分辨率传输模型时性能下降。

如表1的第一行所示，当我们通过双三次插值方法在更大的图像分辨率和窗口尺寸下直接测试预训练ImageNet-1K模型(256 × 256个图像，窗口尺寸为8 × 8)的精度时，精度显著下降。可能有必要重新检查原始Swin变压器中的相对位置偏置方法。

在下面的小节中，我们将介绍解决这些问题的技术，包括残差后归一化和缩放余弦注意力以解决不稳定性问题，以及对数间隔连续位置偏差方法以解决跨窗口分辨率转移的问题。

## 3.2 Scaling Up Model Capacity

如3.1节所述，最初的Swin变形器(和大多数视觉变形器)在每个块的开始处采用了层规范层，继承了vanilla ViT。当我们放大模型容量时，在更深的层观察到活化值的显著增加。事实上，在预归一化配置中，每个残差块的输出激活值被直接合并回主分支，并且主分支的幅度在更深的层越来越大。不同层中的大幅度差异导致训练不稳定。

### Post normalization

为了缓解这个问题，我们建议使用残差后归一化方法，如图1所示。在这种方法中，每个残差块的输出在合并回主分支之前被归一化，并且当层变得更深时，主分支的幅度不会累积。如图2所示，这种方法的激活幅度比原始归一化前的配置要温和得多

在我们最大的模型训练中，我们在每6个变压器块的主分支上引入额外的层标准化层，以进一步稳定训练。

### Scaled cosine attention

在最初的自我关注计算中，像素对的相似性项被计算为查询和关键向量的点积。我们发现，当这种方法用于大型视觉模型时，一些块和头部的学习注意图经常由几个像素对支配，特别是在res-post-norm配置中。为了缓解这个问题，我们提出了一种缩放余弦注意力方法，该方法通过缩放余弦函数来计算像素对I和j的注意力logit:
$$
\operatorname{Sim}\left(\mathbf{q}_{i}, \mathbf{k}_{j}\right)=\cos \left(\mathbf{q}_{i}, \mathbf{k}_{j}\right) / \tau+B_{i j}
$$
其中Bij是像素I和j之间的相对位置偏差；τ是可学习的标量，不跨磁头和层共享

τ设置为大于0.01。余弦函数是自然归一化的，因此可以具有较温和的注意值。

## 3.3. Scaling Up Window Resolution

在这一小节中，我们介绍了一种对数间隔连续位置偏差方法，以便相对位置偏差可以跨窗口分辨率平滑转移。

#### Continuous relative position bias

连续位置偏差方法不是直接优化参数化偏差，而是在相对坐标上采用小的元网络:
$$
B(\Delta x, \Delta y)=\mathcal{G}(\Delta x, \Delta y)
$$
其中G是小型网络，例如，默认情况下，两层MLP之间具有ReLU激活。

元网络G为任意相对坐标生成偏差值，因此可以自然地转移到具有任意变化的窗口大小的微调任务

在推断中，每个相对位置处的偏差值可以预先计算并存储为模型参数，使得推断与原始参数化偏差值方法相同。

#### Log-spaced coordinates

当跨越变化很大的窗口大小时，相对坐标范围的大部分需要外推。为了缓解这个问题，我们建议使用对数间距坐标，而不是原来的线性间距坐标:
$$
\begin{aligned}
&\widehat{\Delta x}=\operatorname{sign}(x) \cdot \log (1+|\Delta x|) \\
&\widehat{\Delta y}=\operatorname{sign}(y) \cdot \log (1+|\Delta y|)
\end{aligned}
$$
其中x，y和c，x，y分别是线性比例坐标和对数空间坐标。

通过使用对数间隔坐标，当我们跨窗口分辨率转移相对位置偏差时，所需的外推比将比使用原始线性间隔坐标小得多。例如，从预训练的8 × 8窗口大小转换到微调的16 × 16窗口大小，使用原始原始坐标，输入坐标范围为[7，7]×7，7]至[15，15]×15，15]。外推比是8 ^ 7 =原射程的1.14倍。使用对数间隔坐标时，输入范围为[2.079，2.079]×[2.079，2.079]至[2.773，2.773]×[2.773，2.773]。外推比是原始距离的0.33倍，比使用原始线间距坐标的外推比小约4倍。

表1比较了不同位置偏差计算方法的传递性能。可以看出，对数间隔CPB(连续位置偏差)方法执行得最好，特别是当转移到更大的窗口大小时。

> extrapolation 外推？

## 3.4. Self-Supervised Pre-training

更大的模型更需要数据。为了解决数据饥饿问题，以前的大型视觉模型通常利用巨大的标记数据，如JFT-3B [17，56，80]。在这项工作中，我们开发了一种自我监督的预训练方法，SimMIM [72]，以减轻对标记数据的需求

通过这种方法，我们成功地训练了一个强大的Swin Transformer模型，该模型具有30亿个参数，在4个代表性的视觉基准上达到了最先进的水平(SOTA)，仅使用了7000万张标记图像(是JFT-3B的1/40)。

## 3.5. Implementation to Save GPU Memory

另一个问题是，当容量和分辨率都很大时，常规实现的GPU内存消耗无法承受。为了解决内存问题，我们采用了以下实现方式:

#### 零冗余优化器(零)[54]。

在优化器的一般数据并行实现中，模型参数和优化状态被广播到每个GPU。这种实现对GPU内存消耗非常不友好，例如，当使用AdamW优化器和fp32权重/状态时，30亿个参数的模型将消耗48G GPU内存。使用ZeRO optimizer，模型参数和相应的优化状态将被拆分并分配给多个GPU，从而显著降低内存消耗。我们采用了DeepSpeed框架，并在实验中使用了ZeRO stage-1选项。这种优化对训练速度影响不大。

#### 激活检查点[12]。

Transformer层中的特征贴图也会消耗大量的GPU内存，这在图像和窗口分辨率较高时会产生瓶颈。激活检查点技术可以显著降低内存消耗，而训练速度最多慢30%。

#### 顺序自我注意计算。

为了在非常大的分辨率上训练大型模型，例如，具有32×32窗口大小的1，536×1，536分辨率的图像，常规的100 GPU(40GB内存)仍然是负担不起的，即使采用了上述两种优化技术。我们发现，在这种情况下，自我注意模块构成了一个瓶颈。为了缓解这个问题，我们依次实现自我注意计算，而不是使用以前的批处理计算方法。这种优化应用于前两个阶段的层，对整体训练速度几乎没有影响。

通过这些实现，我们成功训练了一个3B模型，使用Nvidia A100-40G GPU进行COCO对象检测，输入图像分辨率为1，536×1，536，Kinetics-400动作分类的输入分辨率为320 × 320 × 8。

## 3.6. Model configurations

我们为Swin Transformer V2的4种配置保留了原始Swin Transformer的阶段、模块和通道设置:

SwinV2-T: C = 96, #. block = {2, 2, 6, 2}
SwinV2-S/B/L: C=96/128/192, #.block={2, 2, 18, 2}

其中C是第一阶段的通道数。

我们进一步将Swin Transformer V2扩展到其巨大尺寸和巨型尺寸，分别具有6.58亿个参数和30亿个参数:

SwinV2-H: C = 352, #. block = {2, 2, 18, 2}
SwinV2-G: C = 512, #. block = {2, 2, 42, 4}

对于SwinV2-H和SwinV2-G，我们在主分支上每隔6层增加一层额外的层归一化层。为了节省实验时间，我们只使用SwinV2-G进行大规模实验。SwinV2-H用于另一项关于自我监督学习的平行研究[72]。

# 4. Experiments

## 4.1. Tasks and Datasets

我们在ImageNet-1K图像分类(V1和V2) [18，55]、COCO对象检测[44]和ADE20K语义分割[85]上进行了实验。对于3B模型实验，我们还报告了Kinetics-400视频动作识别的准确性[37]。

图像分类。使用ImageNet-1K V1和V2瓦尔[18，55]进行评估。具有14M图像和22K类别的ImageNet-22K [18]可选地用于预训练。对于我们最大的模型SwinV2-G的预训练，使用了私人收集的具有7000万张图像的ImageNet22K-ext数据集。对于该数据集，进行重复删除过程[51]以排除具有ImageNet1K V1和V2验证集的重叠图像。

物体检测。COCO [44]用于评价 对于我们最大的模型实验，我们使用Object 365 v2数据集[59]在图像分类预训练阶段和COCO微调阶段之间采用了额外的检测预训练阶段。

语义分割。使用ADE20K [85]。

视频动作分类。动力学-400 (K400) [37]用于评估。

预训练和微调设置将在附录中详细说明。

## 4.2. Scaling Up Experiments

我们首先通过将模型放大到30亿个参数和高图像/窗口分辨率，在各种代表性视觉基准上呈现结果。

#### Settings for SwinV2-G experiments

我们在预训练中采用更小的192 × 192图像分辨率，以节省训练成本。我们采取两步预培训方法。首先，在ImageNet-22K-ext数据集上使用自我监督方法[72]通过20个时期对模型进行预训练。第二，使用该数据集上的图像分类任务，通过30个历元进一步预训练该模型。附录中描述了详细的预训练和微调设置。

在以下段落中，我们报告了SwinV2-G在代表性视觉基准测试中的准确性。请注意，由于我们的主要目标是探索如何可行地扩大模型容量和窗口分辨率，以及视觉任务是否可以从显著更大的容量中受益，因此我们在比较中没有特别调整复杂性或预训练数据。

#### ImageNet-1K image classification results

表2比较了SwinV2-G模型与之前ImageNet-1K V1和V2分类中最大/最佳视觉模型。SwinV2-G是目前最大的密集视觉模型。在ImageNet V2基准测试中，它达到了84.0%的最高准确率，比之前的最高准确率(83.3%)高出+0.7%。我们在ImageNet-1K V1上的准确率稍低(90.17%对90.88%)。性能差异可能来自不同程度的数据集过度调整[55]。还要注意的是，与之前的努力相比，我们使用了更少的训练迭代和更低的图像分辨率，同时表现得非常好。

我们还将SwinV2-B和SwinV2-L分别与原始SwinV1-B和SwinV1-L进行了比较，观察到增益分别为+0.8%和+0.4%。与SwinV2-B相比，SwinV2-L的缩小增益可能意味着，如果超过这个大小，则需要更多的标记数据、更强的正则化或高级的自我监督学习方法。

#### COCO object detection results

表3将SwinV2-G模型与之前在COCO对象检测和实例分割方面的最佳结果进行了比较。它在COCO测试开发上实现了63.1/54.4盒/最大AP，比之前的最佳数字w (61.3/53.0 by [74])高出+1.8/1.4

这表明放大视觉模型有利于物体检测的密集视觉识别任务。我们的方法可以在测试中使用不同的窗口大小来获得额外的好处，这可能归功于有效的对数间隔CPB方法。

#### ADE20K semantic segmentation results

表4将SwinV2-G模型与之前在ADE20K语义分割基准上的最佳结果进行了比较。它在ADE20K val集上实现了5990万，比之前的最佳值(58.4乘以[4])高+1.5。这表明放大视觉模型有利于像素级视觉识别任务。在测试时使用更大的窗口大小可以额外带来+0.2的增益，这可能归功于有效的对数间隔CPB方法。

#### Kinetics-400 video action classification results

表5比较了SwinV2-G模型与Kinetics-400动作分类基准测试中之前的最佳结果。它达到了86.8%的最高准确率，比之前的最佳数字[57]高出1.4%。这表明扩大视觉模型也有利于视频识别任务。在这种情况下，在测试时使用更大的窗口大小还可以带来+0.2%的额外好处，这可能要归功于有效的对数间隔CPB方法。

## 4.3. Ablation Study

### Ablation on res-post-norm and scaled cosine attention

表6列出了将建议的res-post-norm和比例余弦注意力方法应用于Swin变压器的性能。这两种技术都提高了所有微小尺寸、小尺寸和基本尺寸的精度，总体改进分别为+0.2%、+0.4%和+0.5%，表明这些技术对于较大的模型更有益

事实证明，这也有利于ViT架构(+0.4%)。如表7所示，所提出的归一化方法的性能也优于其他一些归一化方法。

更重要的是，后规范和比例余弦注意的结合稳定了训练。如图2所示，虽然原始Swin变压器的深层激活值在大(L)尺寸时几乎爆炸，但新版本的激活值行为要温和得多

在大型模型上，自监督预训练[72]使用原始Swin变换器发散，而它通过Swin变换器V2模型训练良好。

### Scaling up window resolution by different approaches

表1和表8分别通过在ImageNet-1K图像分类、COCO对象检测和ADE20K语义分割的3个下游视觉任务中将窗口分辨率从预训练中的256 × 256缩放到更大的尺寸，烧蚀了3种方法的性能。可以看出:1)不同的方法在预训练中的准确率相近(81.7%-81.8%)；2)当转移到下游任务时，两种连续位置偏差(CPB)方法的表现始终优于Swin Transformer V1中使用的参数化位置偏差方法。与线性间隔方法相比，对数间隔方法稍好一些；3)预训练和微调之间的分辨率变化越大，所提出的对数间隔CPB方法的益处越大。

在表1和表8中，我们还报告了在没有微调的情况下使用目标窗口分辨率的精度(参见ImageNet-1K实验中每列的第一个数字)

即使当窗口大小从8扩大到24 (78.9%对81.8%)时，识别准确度仍然不错，而原始方法的前1名准确度从81.7%显著下降到68.7%。还要注意，在没有微调的情况下，使用预训练模型从未见过的12的窗口大小甚至可以比原始精度高+0.4%。这表明我们可以通过调整测试时间窗口来提高准确性，如表3、4和5所示。

# 5. Conclusion

我们已经介绍了将Swin Transformer扩展到30亿个参数并使其能够用高达1，536×1，536分辨率的图像进行训练的技术，包括res-post-norm和缩放余弦注意力，以使模型在容量上更容易扩展，以及对数间隔连续相对位置偏差方法，使模型更有效地跨窗口分辨率转移。改造后的架构被命名为Swin Transformer V2，通过扩大容量和分辨率，它在4个代表性的视觉基准上创造了新的记录。通过这些强有力的结果，我们希望在这个方向上刺激更多的研究，以便我们最终能够弥合视觉和语言模型之间的能力差距，并促进这两个领域的联合建模。

# A

## A3. Learnt Relative Position Bias by Different

图4使用SwinV2-T模型，显示了通过不同偏置计算方法获得的相对位置偏置矩阵(B∈R(2M 1)×(2M 1))。第一块中的3个头的偏置矩阵被可视化。左图显示了使用256×256的输入图像尺寸和8 × 8的窗口尺寸获得的偏差矩阵。右图显示了在512×512的较大输入图像分辨率和16×16的较大窗口尺寸上微调后的偏置矩阵。结果表明，通过两种CPB(连续位置偏差)方法学习的偏差矩阵比通过P-RPE(参数化相对位置偏差)方法学习的偏差矩阵更平滑。图5展示了更多使用这个模型的最后一个模块的例子。

