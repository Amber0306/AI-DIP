# Abstract

最近提出了  DETR，以消除在对象检测中对许多手工设计组件的需求，同时展示了良好的性能。然而，由于 Transformer  注意力模块在处理图像特征图方面的局限性，它存在收敛速度慢和特征空间分辨率有限的问题。为了缓解这些问题，我们提出了可变形  DETR，其注意力模块只关注参考周围的一小组关键采样点。可变形 DETR 可以比 DETR（尤其是在小物体上）获得更好的性能，训练时间少 10 倍。在 COCO  基准上的大量实验证明了我们方法的有效性。

# 1. Introduction

现代目标检测器采用许多手工制作的组件（Liu  等人，2020），例如锚点生成、基于规则的训练目标分配、非极大值抑制 (NMS) 后处理。它们不是完全端到端的。最近，Carion 等人。 （2020 年）提出  DETR 以消除对此类手工组件的需求，并构建了第一个完全端到端的物体检测器，实现了极具竞争力的性能。 DETR 通过结合卷积神经网络 (CNN) 和  Transformer (Vaswani et al., 2017) 编码器-解码器，利用简单的架构。在适当设计的训练信号下，他们利用 Transformer  的多功能和强大的关系建模能力来替换手工制作的规则。

尽管  DETR 具有有趣的设计和良好的性能，但它也有其自身的问题：（1）与现有的目标检测器相比，它需要更长的训练 epoch 才能收敛。例如，在 COCO (Lin  et al., 2014) 基准测试中，DETR 需要 500 个 epoch 才能收敛，这比 Faster R-CNN (Ren et al., 2015)  慢了大约 10 到 20 倍。 (2) DETR  在检测小物体时性能相对较低。现代目标检测器通常利用多尺度特征，从高分辨率特征图中检测小目标。同时，高分辨率特征图导致 DETR  的复杂性令人无法接受。上述问题主要归因于 Transformer  组件在处理图像特征图方面的不足。在初始化时，注意力模块将几乎统一的注意力权重投射到特征图中的所有像素上。要学习注意力权重以专注于稀疏的有意义的位置，需要长时间的训练周期。另一方面，Transformer  编码器中的注意力权重计算是 w.r.t 的二次计算。像素数。因此，处理高分辨率特征图具有非常高的计算和存储复杂性。

在图像域中，可变形卷积  (Dai et al., 2017) 是处理稀疏空间位置的强大而有效的机制。它自然避免了上述问题。而它缺乏元素关系建模机制，这是DETR成功的关键。

在本文中，我们提出了可变形  DETR，它缓解了 DETR 的收敛速度慢和复杂度高的问题。它结合了可变形卷积的稀疏空间采样和 Transformer  的关系建模能力的优点。我们提出了可变形注意力模块，它关注一小组采样位置，作为所有特征图像素中突出关键元素的预过滤器。该模块可以自然地扩展到聚合多尺度特征，而无需  FPN 的帮助（Lin 等人，2017a）。在 Deformable DETR 中，我们利用（多尺度）可变形注意模块来代替处理特征图的 Transformer  注意模块，如图 1 所示。

可变形 DETR  为我们提供了利用端到端目标检测器变体的可能性，这要归功于其快速收敛、计算和内存效率。我们探索了一种简单有效的迭代边界框细化机制来提高检测性能。我们还尝试了一个两阶段的可变形  DETR，其中区域建议也由可变形 DETR 的变体生成，进一步输入解码器以进行迭代边界框细化。

COCO  (Lin et al., 2014) 基准上的大量实验证明了我们方法的有效性。与 DETR 相比，Deformable DETR 可以在训练 epoch 少  10 倍的情况下获得更好的性能（尤其是在小物体上）。所提出的两阶段可变形 DETR 变体可以进一步提高性能。

# 2. Related work

## 高效的注意力机制。

Transformers  (V aswani et al., 2017) 涉及自我注意和交叉注意机制。 Transformers  最知名的问题之一是大量关键元素数量的高时间和内存复杂性，这在许多情况下阻碍了模型的可扩展性。最近，针对这个问题做了很多努力（Tay et al.,  2020b），在实践中大致可以分为三类。

第一类是在键上使用预定义的稀疏注意力模式。最直接的范例是将注意力模式限制为固定的本地窗口。大多数作品（Liu 等人，2018a；Parmar  等人，2018；Child 等人，2019；Huang 等人，2019；Ho 等人，2019；Wang 等人，2020a；Hu 等人，  2019；Ramachandran 等人，2019；Qiu 等人，2019；Beltagy 等人，2020；Ainslie 等人，2020；Zaheer  等人，2020）遵循这一范式。尽管将注意力模式限制在局部邻域可以降低复杂性，但它会丢失全局信息。为了补偿，Child 等人。 （2019）；黄等人。  （2019）；何等人。 （2019）；王等人。 (2020a) 以固定的时间间隔关注关键元素，以显着增加键的感受野。贝尔塔吉等人。 （2020）；安斯利等人。  （2020）；扎希尔等人。 (2020) 允许少量特殊令牌访问所有关键元素。扎希尔等人。 （2020）；邱等人。  （2019）还添加了一些预先固定的稀疏注意力模式来直接关注遥远的关键元素。

第二类是学习数据依赖的稀疏注意力。基塔耶夫等人。 （2020）提出了一种基于局部敏感散列（LSH）的注意力，它将查询和关键元素散列到不同的箱中。 Roy  等人提出了类似的想法。 (2020)，其中 k-means 找出最相关的键。泰等人。 （2020a）学习块置换以实现块稀疏注意力。

第三类是探索self-attention中的low-rank属性。王等人。  （2020b）通过大小维度上的线性投影而不是通道维度来减少关键元素的数量。卡塔罗普洛斯等人。 （2020）；乔罗曼斯基等人。  （2020）通过核化近似重写了selfattention的计算。

在图像领域，高效注意力机制的设计（例如，Parmar  等人（2018）；Child 等人（2019）；Huang 等人（2019）；Ho 等人（2019）；Wang 等人。 (2020a); Hu et al.  (2019); Ramachandran et al. (2019)) 仍然限于第一类。尽管理论上降低了复杂性，Ramachandran 等人。  （2019）；胡等人。 (2019) 承认，由于内存访问模式的内在限制，这种方法的实现速度比具有相同 FLOP 的传统卷积要慢得多（至少慢 3 倍）。

我们提出的可变形注意模块受到可变形卷积的启发，属于第二类。它只关注从查询元素的特征预测的一小部分固定采样点。与  Ramachandran 等人不同。 （2019）；胡等人。 (2019)，在相同的 FLOPs 下，可变形注意力仅比传统卷积稍慢。

## 用于目标检测的多尺度特征表示。

目标检测的主要困难之一是有效地表示不同尺度的目标。现代物体检测器通常利用多尺度特征来适应这一点。作为开创性的工作之一，FPN  (Lin et al., 2017a) 提出了一种自上而下的路径来组合多尺度特征。 PANet (Liu et al., 2018b) 在 FPN  的顶部进一步增加了一个自下而上的路径。孔等人。 （2018）通过全局注意力操作结合了所有尺度的特征。赵等人。 (2019) 提出了一个 U  形模块来融合多尺度特征。最近，NAS-FPN (Ghiasi et al., 2019) 和 Auto-FPN (Xu et al., 2019)  被提出通过神经架构搜索自动设计跨尺度连接。谭等人。 (2020) 提出了 BiFPN，它是 PANet  的重复简化版本。我们提出的多尺度可变形注意模块可以通过注意机制自然地聚合多尺度特征图，而无需这些特征金字塔网络的帮助。

# 3. 回顾transformer和detr

## Transformer中的多头注意力。

Transformers (V aswani et al., 2017)  是一种基于机器翻译注意机制的网络架构。给定一个查询元素（例如，输出句子中的目标词）和一组关键元素（例如，输入句子中的源词），多头注意力模块根据衡量的注意力权重自适应地聚合关键内容查询密钥对的兼容性。为了让模型关注来自不同表示子空间和不同位置的内容，不同注意力头的输出与可学习的权重线性聚合。令 $q \in \Omega_{q}$ 用表示特征 $\boldsymbol{z}_{q} \in \mathbb{R}^{C}$ 索引一个查询元素，$k \in \Omega_{k}$ 用表示特征$\boldsymbol{x}_{k} \in \mathbb{R}^{C}$ 索引一个关键元素，其中 $C$ 是特征维度，$\Omega_{q}$ 和 $\Omega_{k}$分别指定查询和关键元素的集合.然后计算多头注意力特征
$$
\operatorname{MultiHeadAttn}\left(\boldsymbol{z}_{q}, \boldsymbol{x}\right)=\sum_{m=1}^{M} \boldsymbol{W}_{m}\left[\sum_{k \in \Omega_{k}} A_{m q k} \cdot \boldsymbol{W}_{m}^{\prime} \boldsymbol{x}_{k}\right]
$$
其中$m$表示注意头，$\boldsymbol{W}_{m}^{\prime} \in \mathbb{R}^{C_{v} \times C}$和$\boldsymbol{W}_{m} \in \mathbb{R}^{C \times C_{v}}$具有可学习的权重(默认情况下，$C_{v}=C / M$)。attention权重$A_{m q k} \propto \exp \left\{\frac{\boldsymbol{z}_{q}^{T} \boldsymbol{U}_{m}^{T} \boldsymbol{V}_{m} \boldsymbol{x}_{k}}{\sqrt{C_{v}}}\right\}$被归一化为$\sum_{k \in \Omega_{k}} A_{m q k}=1$，其中$\boldsymbol{U}_{m}, \boldsymbol{V}_{m} \in \mathbb{R}^{C_{v} \times C}$也是可学习权重。为了消除不同的空间位置歧义，表示特征$\boldsymbol{z}_{q}$和$\boldsymbol{x}_{k}$通常是元素内容和位置嵌入的拼接/求和。

Transformer有两个已知的问题。一是在融合之前，需要很长的训练计划。假设查询数为$N_{q}$，关键元素数为$N_{k}$。通常，在适当的参数初始化的情况下，$\boldsymbol{U}_{m} \boldsymbol{z}_{q}$和$\boldsymbol{V}_{m} \boldsymbol{x}_{k}$服从均值为0和方差为1的分布，这使得当$N_{k}$较大时，注意权重$A_{m q k} \approx \frac{1}{N_{k}}$。这将导致输入要素的梯度不明确。因此，需要长时间的训练计划，以便注意力权重可以集中在特定的关键字上。在图像域中，关键元素通常是图像像素，$N_{k}$可能非常大，并且收敛是乏味的。

另一方面，由于有大量的查询和关键元素，多头注意的计算和存储复杂性可能非常高。方程的计算复杂性。1为$O\left(N_{q} C^{2}+N_{k} C^{2}+N_{q} N_{k} C\right)$。在图像域中，查询和关键元素都是像素，$N_{q}=N_{k} \gg C$，复杂性由第三项所支配，如$O\left(N_{q} N_{k} C\right)$。因此，多头注意模块的复杂度随着特征图的大小呈二次曲线增长。

## DETR

DETR(Carion等人，2020)构建在Transformer编解码器架构之上，结合基于集合的匈牙利损失，通过二部匹配强制对每个地面真相边界框进行唯一预测。我们简要回顾一下网络体系结构，如下所示。

给定由cnn主干(例如Resnet(他等人，2016年))提取的输入特征映射$\boldsymbol{x} \in \mathbb{R}^{C \times H \times W}$，Detr利用标准的Transformer编解码器体系结构将输入特征映射转换为一组对象查询的特征。在解码器产生的目标查询特征之上加入3层前馈神经网络(FFN)和线性投影作为检测头。FFN作为回归分支来预测边界框坐标$\boldsymbol{b} \in[0,1]^{4}$，其中$\boldsymbol{b}=\left\{b_{x}, b_{y}, b_{w}, b_{h}\right\}$编码归一化的框中心坐标、框的高度和宽度(相对于图像大小)。线性投影作为分类分支，产生分类结果。

对于DETR中的Transformer编码器，查询和关键元素都是特征地图中的像素。输入是ResNet特征地图(带有编码的位置嵌入)。让$H$和$W$分别表示要素地图的高度和宽度。自我注意的计算复杂度为$O\left(H^{2} W^{2} C\right)$，并随空间大小呈二次曲线增长。

对于DETR中的Transformer解码器，输入包括来自编码器的特征映射和由可学习位置嵌入表示的N个对象查询(例如，N=100)。解码器中存在两种类型的注意模块，即交叉注意模块和自我注意模块。在交叉注意模块中，对象查询从特征地图中提取特征。Q元素是对象查询，K元素是编码器输出的特征地图。其中，$N_{q}=N, N_{k}=H \times W$，交叉注意的复杂度为$O\left(H W C^{2}+N H W C\right)$。复杂度随着要素地图的空间大小而线性增长。在自我注意模块中，对象查询相互交互，以捕捉它们之间的关系。查询和关键元素都是对象查询。其中，$N_{q}=N_{k}=N$，自我注意模块的复杂度为$O\left(2 N C^{2}+N^{2} C\right)$。对于中等数量的对象查询，其复杂性是可以接受的。

DETR是一种有吸引力的目标检测设计，它消除了对许多手动设计组件的需要。然而，它也有自己的问题。这些问题主要可归因于Transformer在将图像特征图作为关键元素处理时的注意力不足：(1)DETR在检测小目标方面的性能相对较低。现代物体探测器使用高分辨率特征图来更好地检测小物体。然而，高分辨率的特征映射会给DETR的Transformer编码器中的自我注意模块带来不可接受的复杂性，其复杂度是输入特征映射空间大小的平方。(2)与现代目标检测器相比，DETR需要更多的训练周期才能收敛。这主要是因为处理图像特征的注意模块很难训练。例如，在初始化时，交叉注意模块对整个特征图的注意力几乎是平均的。然而，在训练结束时，注意图被学习到非常稀疏，只专注于物体的末端。看来，DETR需要很长的训练计划才能学习到注意图中的如此重大变化。

# 4. Method

## 4.1 Deformable Transformers for End-to-End Object Detection

### Deformable Attention Module

在图像特征地图上应用变形金刚注意力的核心问题是它将查看所有可能的空间位置。为了解决这个问题，我们提出了一个可变形的注意模块。受可变形卷积(Dai等人，2017；朱等人，2019b)的启发，可变形注意模块只关注参考点周围的一小组关键采样点，而不考虑特征地图的空间大小，如图2所示。通过仅为每个查询分配少量固定数量的关键字，收敛和特征空间分辨率的问题可以得到缓解。

给定输入特征映射$\boldsymbol{x} \in \mathbb{R}^{C \times H \times W}$，让q索引具有内容特征$\boldsymbol{z}_{q}$和2-d参考点$\boldsymbol{p}_{q}$的查询元素，通过下式计算可变形注意力特征
$$
\operatorname{DeformAttn}\left(\boldsymbol{z}_{q}, \boldsymbol{p}_{q}, \boldsymbol{x}\right)=\sum_{m=1}^{M} \boldsymbol{W}_{m}\left[\sum_{k=1}^{K} A_{m q k} \cdot \boldsymbol{W}_{m}^{\prime} \boldsymbol{x}\left(\boldsymbol{p}_{q}+\Delta \boldsymbol{p}_{m q k}\right)\right]
$$
其中，$m$索引注意力头部，$k$索引采样关键字，$K$是采样关键字总数($K \ll H W$)。$\Delta\boldsymbol{p}_{m q k}$和 $A_{m q k}$分别表示第m个注意力头部中第k个采样点的采样偏移量和关注权重。标量注意权重$A_{m q k}$位于范围[0，1]内，归一化为$\sum_{k=1}^{K} A_{m q k}=1 $。$ \Delta \boldsymbol{p}_{m q k} \in \mathbb{R}^{2}$是具有无约束范围的2维实数。由于$\boldsymbol{p}_{q}+\Delta \boldsymbol{p}_{m q k}$是分数的，所以采用了Dai等人的双线性插值法。(2017)计算$\boldsymbol{x}\left(\boldsymbol{p}_{q}+\Delta \boldsymbol{p}_{m q k}\right)$。$\Delta\boldsymbol{p}_{m q k}$和 $A_{m q k}$都是通过在查询特征$\boldsymbol{z}_{q}$上的线性投影获得的。在实现中，查询特征$\boldsymbol{z}_{q}$被馈送到$3MK$个通道的线性投影算子，其中前$2MK$个通道对采样偏移量$\Delta\boldsymbol{p}_{m q k}$进行编码，并且剩余的$MK$个通道被馈送到SoftMax算子以获得关注权重$A_{m q k}$。

可变形注意模块被设计用于将卷积特征映射作为关键元素进行处理。设$N_q$为查询元素的个数，当$M  K$较小时，可变形注意模块的复杂度为$O\left(2 N_{q} C^{2}+\min \left(H W C^{2}, N_{q} K C^{2}\right)\right)$(详见附录A.1)。当应用于DETR编码器时，当$N_q=HW$时，其复杂度为$O\left(H W C^{2}\right)$，其复杂度与空间大小呈线性关系。当将其应用于DETR解码器的交叉注意模块时，当$N_q=N$($N$为对象查询次数)时，复杂度为$O\left(N K C^{2}\right)$，与空间大小$HW$无关。

### Multi-scale Deformable Attention Module

大多数现代目标检测框架受益于多尺度特征地图(Liu等人，2020)。我们提出的可变形注意模块可以自然地扩展到多尺度特征地图。

设$\left\{\boldsymbol{x}^{l}\right\}_{l=1}^{L}$为输入的多比例尺特征地图，其中$\boldsymbol{x}^{l} \in \mathbb{R}^{C \times H_{l} \times W_{l}}$。设$\hat{\boldsymbol{p}}_{q} \in[0,1]^{2}$为每个查询元素q的参考点的归一化坐标，则多尺度可变形注意模块被应用为
$$
\operatorname{MSDeformAttn}\left(\boldsymbol{z}_{q}, \hat{\boldsymbol{p}}_{q},\left\{\boldsymbol{x}^{l}\right\}_{l=1}^{L}\right)=\sum_{m=1}^{M} \boldsymbol{W}_{m}\left[\sum_{l=1}^{L} \sum_{k=1}^{K} A_{m l q k} \cdot \boldsymbol{W}_{m}^{\prime} \boldsymbol{x}^{l}\left(\phi_{l}\left(\hat{\boldsymbol{p}}_{q}\right)+\Delta \boldsymbol{p}_{m l q k}\right)\right]
$$
其中$m$索引注意力头部，$l$索引输入特征级别，k索引采样点。$\Delta \boldsymbol{p}_{m l q k}$和$A_{m l q k}$分别表示第l个特征级别和第m个注意头中第k个采样点的采样偏移量和关注权重。标量注意权重$A_{m l q k}$被归一化为PL  l=1pk  k=1amlqk=1。这里，我们使用归一化坐标ˆpq∈[0，1]2来表示尺度公式的清晰度，其中归一化坐标(0，0)和(1，1)分别表示图像的左上角和右下角。公式3中的函数φl(ˆpq)将归一化坐标ˆpq重新缩放到l级的输入特征映射。多尺度可变形关注与之前的单尺度版本非常相似，不同之处在于它从多尺度特征地图中采样LK点，而不是从单尺度特征地图中采样K点。

当L=1，K=1，且W0m∈Rcv×C固定为单位矩阵时，所提出的注意模块将退化为可变形卷积(Dai等人，2017年)。可变形卷积是为单标度输入而设计的，只为每个注意力头部聚焦一个采样点。然而，我们的多尺度可变形注意关注来自多尺度输入的多个采样点。所提出的(多尺度)可变形注意模块也可以被视为变压器注意的有效变体，其中由可变形采样位置引入了预过滤机制。当采样点遍历所有可能的位置时，建议的注意模块相当于变压器注意。

### Deformable Transformer Encoder

### Deformable Transformer Decoder

## 4.2 Additional Improvements and Variants For Deformable DETR

### Iterative Bounding Box Refinement

### Two-Stage Deformable DETR

# 6. Conclusion

可变形DETR是一种端到端的目标检测器，具有高效、快速收敛的特点。它使我们能够探索更有趣和更实用的端到端对象探测器的变体。可变形DETR的核心是(多尺度)可变形注意模块，它是处理图像特征图的一种有效的注意机制。我们希望我们的工作在探索端到端目标检测方面开辟了新的可能性。