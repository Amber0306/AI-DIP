# 0. abstract
我们提出了稀疏R-CNN，一种用于图像中物体检测的纯稀疏方法。现有的物体检测工作严重依赖密集的物体候选者，例如在H×W大小的图像特征图的所有网格上预先定义的k个锚箱。然而，在我们的方法中，一个固定的、总长度为N的稀疏物体建议集被提供给物体识别头来进行分类和定位。通过消除HW k（多达几十万）手工设计的对象候选人到N（例如100）个可学习的建议，稀疏R-CNN完全避免了所有与对象候选人设计和多对一标签分配有关的工作。更重要的是，最终的预测是直接输出的，没有非最大限度的压制后程序。在具有挑战性的COCO数据集上，稀疏R-CNN展示了与成熟的检测器基线相同的准确性、运行时间和训练收敛性能，例如，在标准的3倍训练计划中实现了44.5AP，并使用ResNet-50 FPN模型以22帧/秒的速度运行。我们希望我们的工作能够启发人们重新思考物体检测器中密集先验的惯例。代码见：https://github.com/PeizeSun/SparseR-CNN。

图1：不同目标检测管道的比较。(a)在密集检测器中，HW k候选物体枚举所有图像网格，如RetinaNet[23]。(b)在稠密稀疏检测器中，从密集的HW k对象候选中选择一个小的N个候选集合，然后通过池化操作提取相应区域内的图像特征，如Faster RCNN[30]。(c)我们提出的稀疏R-CNN，直接提供了N个学习到的对象建议的小集合。N远小于HW k。

# 1. introduction
图2：RetinaNet、Faster RCNN、DETR和Sparse R-CNN在COCO val2017上的收敛曲线[24]。稀疏R-CNN在训练效率和检测质量方面取得了有竞争力的表现。

物体检测的目的是对一组物体进行定位并识别它们在图像中的类别。密集的先验一直是检测器成功的基石。在经典的计算机视觉中，滑动窗口范式，即在密集的图像网格上应用分类器，是几十年来领先的de1 arXiv:2011.12450v1 [cs.CV] 25 Nov 2020检测方法[6, 9, 38]。现代主流的单阶段检测器在密集的特征图网格上预先定义标记，如图1a所示的锚定盒[23, 29]或参考点[35, 44]，并预测物体的相对比例和与边界盒的偏移量，以及相应的类别。虽然两阶段管道工作在稀疏的建议框集上，但它们的建议生成算法仍然建立在密集的候选者上[11, 30]，如图1b所示。

这些成熟的方法在概念上是直观的，并提供稳健的性能[8, 24]，同时具有快速的训练和推理时间[40]。除了它们的巨大成功，重要的是要注意密集优先检测器存在一些限制。1）这种管道通常会产生冗余和近乎重复的结果，从而使非最大抑制（NMS）[1，39]后处理成为必要的组成部分。2）训练中的多对一标签分配问题[2, 42, 43]使得网络对启发式分配规则很敏感。3) 最终的性能在很大程度上受到锚箱的尺寸、长宽比和数量[23, 29]、参考点的密度[19, 35, 44]和提案生成算法[11, 30]的影响。

- Such pipelines usually pro-
duce redundant and near-duplicate results, thus making non-maximum suppression (NMS) post-processinga necessary component. 
- The many-to-one label assignment problem in training makes the network sensitive to heuristic assign rules.
- The final performance is
largely affected by sizes, aspect ratios and number of an-
chor boxes , density of reference points and proposal generation algorithm

尽管在物体检测器中密集惯例被广泛认可，但要问的一个自然问题是。是否有可能设计一个稀疏的检测器？最近，DETR提出将物体检测重新表述为一个直接和稀疏的集合预测问题[3]，其输入仅仅是100个学习过的物体查询[37]。最终的预测集是直接输出的，不需要任何手工设计的后处理。尽管DETR的框架简单而奇妙，但它要求每个对象查询都要与全局图像上下文互动。这种密集的特性不仅减慢了它的训练收敛速度[45]，而且还阻碍了它建立一个彻底的物体检测的稀疏管道。

我们认为稀疏的特性应该包括两个方面：稀疏的盒子和稀疏的特征。稀疏盒子意味着少量的起始盒子（例如100个）就足以预测图像中的所有物体。而稀疏特征表示每个盒子的特征不需要与整个图像上的所有其他特征进行互动。从这个角度来看，DETR不是一个纯粹的备用方法，因为每个物体的查询必须与全图上的密集特征进行交互。

在本文中，我们提出了稀疏R-CNN，这是一种纯粹的稀疏方法，没有在所有（密集）图像网格上列举物体的位置候选，也没有与全局（密集）图像特征交互的物体查询。如图1c所示，物体候选者被赋予了一个固定的、由4维坐标表示的小的可学习边界盒。以COCO数据集[24]为例，总共需要100个盒子和400个参数，而不是从区域建议网络（RPN）[30]中数十万个候选物中预测出来的参数。这些稀疏的候选者被用作RoIPool[10]或RoIAlign[13]的proposal boxes来提取感兴趣区域（RoI）的特征。

可学习的建议框是图像中潜在物体位置的统计数据。而4维坐标仅仅是物体的粗略表示，缺乏很多信息性的细节，如姿势和形状。在这里，我们引入了另一个概念，即提案特征，它是一个高维（如256）的潜在向量。与粗糙的边界盒相比，它有望编码丰富的实例特征。特别是，提议特征为其独有的对象识别头生成一系列定制参数。我们称这种操作为动态实例互动头，因为它与最近的动态方案[18, 34]有相似之处。与[30]中的共享2-fc层相比，我们的头更灵活，并且在准确性上占有明显的领先优势。我们在实验中表明，以独特的提议特征而不是固定参数为条件的头的表述实际上是稀疏R-CNN成功的关键。建议框和建议特征都是随机初始化的，并与整个网络的其他参数一起优化。

我们的稀疏R-CNN中最显著的特性是它在整个时间内的稀入稀出范式。最初的输入是一组稀疏的提议框和提议特征，以及一对一的动态实例交互。管道中既不存在密集的候选人[23, 30]，也不存在与全局（密集）特征[3]的交互。这种纯粹的稀疏性使得稀疏R-CNN成为R-CNN家族中的一个全新成员。

在具有挑战性的COCO数据集[24]上，稀疏R-CNN证明了其准确性、运行时间和训练收敛性能与成熟的检测器[2, 30, 35]相当，例如，使用ResNet-50 FPN模型，在标准的3×训练计划中达到44.5AP，运行速度为22帧/秒。就我们所知，所提出的稀疏R-CNN是第一个证明相当稀疏的设计是合格的工作。我们希望我们的工作能够启发人们重新思考密集先验在物体检测中的必要性，并探索下一代的物体检测器。

# 2. Related work
## Dense method.
滑动窗口范式在物体检测中已经流行了很多年。受经典特征提取技术[6, 38]的限制，几十年来其性能已经趋于稳定，应用场景也很有限。深度卷积神经网络（CNN）的发展[14, 17, 20]培养了一般的物体检测，实现了性能的显著提高[8, 24]。其中一个主流管道是单阶段检测器，它直接预测锚箱的类别和位置，以单次拍摄的方式密集地覆盖空间位置、比例和长宽比，如OverFeat[32]、YOLO[29]、SSD[25]和RetinaNet[23]。最近，无锚算法[16, 21, 35, 44]被提出来，通过用参考点代替手工制作的锚箱，使这个管道更加简单。上述所有的方法都是建立在密集的候选人上，每个候选人都被直接分类和回归。这些候选者在训练时根据预先定义的原则被分配到地面实况对象箱中，例如，锚点是否与其对应的地面实况有较高的交叉-重合（IoU）阈值，或者参考点是否落在对象箱中。此外，需要对NMS进行后处理[1, 39]，以便在推理过程中去除多余的预测值。

## Dense-to-sparse method.
两阶段检测器是另一个主流管道，多年来一直主导着现代物体检测[2, 4, 10, 11, 13, 30]。这种模式可以被看作是密集检测器的延伸。它首先从密集区域候选人中获得一组稀疏的前景建议框，然后细化每个建议的位置并预测其具体类别。在这些两阶段方法中，区域提议算法在第一阶段起着重要作用，如R-CNN中的选择性搜索[36]和Faster R-CNN中的区域提议网络（RPN）[30]。与密集管道类似，它也需要NMS的后处理和手工制作的标签分配。从几十万个候选者中只有几个前景建议，因此这些检测器可以被总结为密集-稀疏方法

最近，DETR[3]被提出来，直接输出预测结果，不需要任何手工制作的组件，取得了非常有竞争力的性能。DETR利用一组稀疏的对象查询，与全局（密集）图像特征进行交互，从这个角度来看，它可以被看作是另一种密集-稀疏的表述。

## Sparse method.
稀疏物体检测有可能消除设计密集候选物的努力，但已经落后于上述检测器的准确性。G-CNN[27]可以被看作是这组算法的前驱。它从图像上的多尺度规则网格开始，迭代更新方框以覆盖和分类物体。这种手工设计的规则先验显然是次优的，不能达到顶级性能。相反，我们的稀疏R-CNN应用了可学习的建议并取得了更好的性能。同时，引入了Deformable-DETR[45]，以限制每个物体查询只关注参考点周围的一小部分关键采样点，而不是特征图中的所有点。我们希望稀疏方法可以作为坚实的基线，并有助于促进物体检测领域的未来研究。

# 3. Sparse rcnn

稀疏R-CNN框架的中心思想是用一组小的提议框（如100个）取代区域提议网络（RPN）中的数十万个候选人。在这一节中，我们首先简要介绍了所提方法的整体架构。然后，我们详细描述每个组件。

图3： 稀疏R-CNN管线的概述。输入包括一幅图像、一组提议框和提议特征，其中后两者是可学习的参数。主干网提取特征图，每个提议框和提议特征都被输入到其专属的动态头中以生成物体特征，最后输出分类和位置。

## 3.1 pipeline

稀疏R-CNN是一个简单、统一的网络，由一个骨干网络、一个动态实例交互头和两个特定任务预测层组成（图3）。总共有三个输入，一个图像、一组建议框和建议特征。后两者是可学习的，可以和网络中的其他参数一起进行优化。

## 3.2 module
### backbone

基于ResNet架构的特征金字塔网络（FPN）[14, 22]被用作骨干网络，从输入图像中生成多尺度特征图。按照[22]，我们构建了具有P2到P5级别的金字塔，其中l表示金字塔级别，Pl的分辨率比输入低2l。所有的金字塔级别都有C=256通道。更多细节请参考[22]。实际上，Spare R-CNN有可能从更复杂的设计中获益，以进一步提高其性能，例如堆叠编码器层[3]和可变形卷积网络[5]，最近的工作DeformableDETR[45]就是在此基础上建立的。然而，我们将设置与Faster R-CNN[30]相一致，以显示我们方法的简单性和有效性。

### Learnable proposal box.
一组固定的小的可学习建议框（N×4）被用作区域建议，而不是来自区域建议网络（RPN）的预测。
这些建议箱由4维参数表示，范围从0到1，表示归一化中心坐标、高度和宽度。在训练过程中，提案箱的参数将通过反向传播算法进行更新。由于可学习的特性，我们在实验中发现初始化的影响很小，从而使该框架更加灵活。

从概念上讲，这些学习到的建议框是训练集中潜在物体位置的统计数据，可以看作是对最有可能包含图像中物体的区域的初步猜测，而不管输入是什么。而来自RPN的建议与当前图像密切相关，并提供粗略的物体位置。我们重新思考，在有后期阶段来细化盒子的位置时，第一阶段的定位是奢侈的。相反，一个合理的统计数字已经可以成为合格的候选人。在这种观点下，稀疏R-CNN可以被归类为物体检测器范式的扩展，从彻底密集[23, 25, 28, 35]到密集到稀疏[2, 4, 11, 30]到彻底稀疏，如图1。

### Learnable proposal feature.
尽管4维提议框是描述物体的一个简短而明确的表达方式，但它提供了一个粗略的物体定位，很多信息性的细节被丢失了，例如物体的姿势和形状。在这里，我们引入了另一个概念，即提议特征（N×d），它是一个高维（如256）的潜在向量，预计将编码丰富的实例特征。建议特征的数量与盒子相同，接下来我们将讨论如何使用它。

### Dynamic instance interactive head.
给定N个提案箱，稀疏R-CNN首先利用RoIAlign操作来提取每个箱子的特征。然后，每个盒子的特征将被用来使用我们的预测头生成最终的预测结果。

图4说明了预测头，被称为动态实例交互模块，由动态算法[18, 34]激发。每个RoI特征都被送入它自己的专属头，用于物体定位和分类，其中每个头都以特定的提议特征为条件。在我们的设计中，提议特征和提议框是一一对应的。对于N个提议箱，采用N个提议特征。每个RoI特征fi(S×S×C)将与相应的提议特征pi(C)相互作用，以过滤掉无效的箱体，输出最终的对象特征(C)。最终的回归预测是由具有ReLU激活函数和隐藏维度C的3层感知来计算的，分类预测是由线性投影层来进行的。

图4：我们的动态实例互动模块的概述。过滤器随不同的实例而变化，也就是说，第k个提案特征为相应的第k个RoI产生动态参数。

对于灯光设计，我们用ReLU激活函数进行连续的1×1卷积，以实现交互过程。每个提议的特征pi将与RoI特征进行卷积，以得到一个更具辨识度的特征。更多细节，请参考我们的代码。我们注意到，只要支持并行操作以提高效率，互动头的实现细节并不关键。

我们的建议特征可以被看作是注意力机制的实现，用于关注大小为S×S的RoI中的哪些bin。建议特征生成卷积的内核参数，然后RoI特征被生成的卷积处理，得到最终特征。通过这种方式，那些具有最多前景信息的仓对最终的物体位置和分类产生影响。

我们还采用迭代结构来进一步提高性能。新生成的对象框和对象特征将作为迭代过程中下一阶段的建议框和建议特征。由于其稀疏的特性和轻量级的动态头，它只引入了少量的计算开销。自我注意模块[37]被嵌入到动态头中以推理对象之间的关系。我们注意到，关系网络[15]也利用了注意力模块。然而，除了对象特征外，它还需要几何属性和复杂的等级特征。我们的模块要简单得多，只需要将物体特征作为输入。

DETR[3]中提出的对象查询与提议特征有类似的设计。然而，对象查询是学习位置编码。当与对象查询交互时，需要特征图来增加空间位置编码，否则会导致显著的下降。我们的提议特征与位置无关，我们证明了我们的框架可以在没有位置编码的情况下很好地工作。我们在实验部分提供了进一步的比较。

### Set prediction loss.
稀疏R-CNN在分类和箱体坐标的固定大小的预测集合上应用集合预测损失[3, 33, 41]。基于集合的损失产生了预测和地面真实对象之间的最佳双点匹配。匹配成本定义如下。

L = λcls · Lcls + λL1 · LL1 + λgiou · Lgiou

这里Lcls是预测分类和地面真实类别标签的焦点损失[23]，LL1和Lgiou分别是归一化中心坐标和预测框与地面真实框的高度和宽度之间的L1损失和广义IoU损失[31]。 λcls, λL1和λgiou是每个组件的系数。训练损失与匹配成本相同，只是只对匹配对进行训练。最后的损失是所有配对的总和，并以训练批次中的对象数量为标准。

R-CNN家族[2, 43]一直被标签分配问题所困扰，因为多对一的匹配仍然存在。在这里，我们提供了新的可能性，即直接绕过多对一的匹配，引入基于集合的损失的一对一匹配。这是探索端到端物体检测的一个尝试。

# 4. Experiments
我们的实验是在具有挑战性的MS COCO基准[24]上进行的，使用对象检测的标准度量。所有模型在COCO train2017分割训练(约118k图像)，并使用val2017 (5k图像)进行评估。
### Training details.
除非另有说明，否则使用ResNet-50[14]作为骨干网。优化器为AdamW[26]，权重衰减0.0001。小批处理为16张图像，所有模型均使用8个gpu进行训练。默认的训练时间表是36 epoch，初始学习率设置为2.5 × 10−5，分别除以27 epoch和33 epoch的10。骨干用ImageNet[7]上预训练的权值初始化，其他新添加的层用Xavier[12]初始化。数据增强包括随机水平、缩放抖动，调整输入图像的大小，使最短边至少480像素，最长边最多为800像素，最长边最多为1333像素。[3,45]， λcls = 2， λL1 = 5， λgiou = 2。提案框、提案特性和迭代的默认数量分别为100、100和6。

### Inference details.
稀疏R-CNN的推理过程非常简单。给定一个输入图像，稀疏R-CNN直接预测出与它们的分数相关的100个边界框。分数表示盒子包含一个对象的概率。为了评估，我们直接使用这100盒，没有任何后期处理。

## 4.1. Main Result
我们提供了两个版本的Sparse R-CNN，以便在不同的检测器下进行公平的比较。第一种采用了100个可学习的建议框，没有随机的作物数据增强，用于与主流的目标检测器进行比较，如Faster R-CNN和RetinaNet[40]。第二种方法利用了300个可学习的提议框和随机作物数据增强，并用于与detr系列模型进行比较[3,45]。

如表1所示，Sparse R-CNN的性能优于现有的主流检测器，如RetinaNet和Faster R-CNN，遥遥领先。令人惊讶的是，基于ResNet-50的稀疏RCNN实现了42.3 AP，这已经在准确性上与Faster R-CNN在ResNet-101上竞争。

我们注意到，DETR和Deformable DETR通常采用更强的特征提取方法，如堆叠编码器层和可变形卷积。稀疏R-CNN的较强实现被用来与这些检测器进行更公平的比较。即使使用简单的FPN作为特征提取方法，稀疏R-CNN也表现出更高的准确性。此外，与DETR相比，稀疏R-CNN在小物体上的检测性能要好得多（26.9 AP vs. 22.5 AP）。

如图2所示，稀疏R-CNN的训练收敛速度比DETR快10倍。自提出以来，DETR一直存在收敛速度慢的问题，这促使我们提出了可变形DETR。与可变形DETR相比，稀疏R-CNN在准确性方面表现得更好（44.5 AP vs. 43.8 AP），运行时间更短（22 FPS vs. 19 FPS），训练计划更短（36 epochs vs. 50 epochs）。

Sparse R-CNN的推理时间与其他检测器相当。我们注意到，有100个提议的模型以23FPS的速度运行，而300个提议只下降到22FPS，这要归功于动态实例交互头的轻型设计。
## 4.2. Module Analysis
在这一节中，我们分析了稀疏RCNN中的每个组件。除非另有说明，所有模型都是基于ResNet50-FPN主干，100个提议，3次训练计划。

### Learnable proposal box.
从Faster R-CNN开始，我们天真地用一组稀疏的可学习的提议框代替RPN。性能从40.2AP（表1第3行）下降到18.5（表2）。我们发现，即使堆叠更多的全连接层，也没有明显的改善。

### Iterative architecture.
迭代更新盒子是一个直观的想法，可以提高其性能。然而，我们发现一个简单的级联结构并没有带来很大的变化，如表3所示。我们分析其原因是，与[2]中主要定位在物体周围的精炼建议框相比，我们的案例中的候选者要粗糙得多，使得它难以被优化。

我们观察到，在整个迭代过程中，一个提议框的目标对象通常是一致的。因此，前一阶段的对象特征可以重复使用，为下一阶段的工作提供强有力的线索。物体特征编码了丰富的信息，如物体的姿势和位置。在原始级联结构的基础上，这种特征重用的微小变化导致了11.7AP的巨大收益。最后，迭代结构带来了13.7个AP的改进，如表2第二行所示。

### Dynamic head.
动态头以一种与上面讨论的迭代结构不同的方式使用前一阶段的对象特征。前一阶段的对象特征不是简单的串联，而是先由自我关注模块处理，然后作为建议特征来实现当前阶段的实例交互。自我关注模块被应用于对象特征集，以推理对象之间的关系。表4显示了自我注意和动态实例交互的好处。最后，稀疏R-CNN实现了42.3AP的准确率表现。

### Initialization of proposal boxes.
密集检测器总是严重依赖对象候选者的设计，而稀疏R-CNN中的对象候选者是可学习的，因此，与设计手工制作的锚有关的所有努力都被避免了。然而，人们可能会担心，提案箱的初始化在稀疏RCN中起着关键作用。在这里，我们研究了不同方法对初始化建议框的影响。

- "中心 "是指所有建议框在开始时都位于图像的中心，高度和宽度被设置为图像大小的0.1。

- "图像 "是指所有建议框都被初始化为整个图像的大小。

- 网格 "表示提案框被初始化为图像中的规则网格，这正是GCNN中的初始框[27]

- 随机 "表示提案框的中心、高度和宽度都是以高斯分布随机初始化的。

从表5中我们可以看出，稀疏R-CNN的最终性能对提案箱的初始化是相对稳健的。

### Number of proposals.
提案的数量在很大程度上影响了密集型和稀疏型检测器。原始的Faster R-CNN使用300个提议[30]。后来它增加到2000个[40]，获得了更好的性能。我们还在表6中研究了提案数量对稀疏R-CNN的影响。从100到500的提案数量的增加导致了持续的改进，表明我们的框架很容易在各种情况下使用。而500个提议需要更多的训练时间，所以我们选择100和300作为主要配置。

### Number of stages in iterative architecture.
迭代结构是一种广泛使用的技术，可以提高物体检测性能[2, 3, 38]，特别是对于稀疏RCNN。表7显示了迭代结构中阶段数的影响。没有迭代结构，性能仅为21.7AP。考虑到第一阶段的输入建议是对可能的物体位置的猜测，这个结果并不令人惊讶。增加到2个阶段会带来14.5个AP的收益，最高可达到36.2个AP的竞争力。逐渐增加级数，性能在6级时趋于饱和。我们选择6级作为默认配置。

### Dynamic head vs. Multi-head Attention.
如第3节所述，动态头使用提议特征来过滤RoI特征，最后输出对象特征。我们发现，多头注意力模块[37]为实例交互提供了另一种可能的实现。我们在表8中进行了对比实验，其性能落后于6.6AP。与线性多头关注相比，我们的动态头更加灵活，其参数以其特定的提议特征为条件，更多的非线性能力可以很容易地引入。

### Proposal feature vs. Object query.
这里我们对DETR中提出的对象查询[3]和我们的建议特征进行了比较。正如在[3]中所讨论的，物体查询是学习位置编码，引导解码器与图像特征图和空间位置编码的总和进行交互。只使用图像特征图会导致显著的下降。然而，我们的建议特征可以被看作是一个特征过滤器，它与位置无关。比较结果如表9所示，如果去除空间位置编码，DETR下降了7.8AP。相反，位置编码在稀疏R-CNN中没有任何增益。

## 4.3. The Proposal Boxes Behavior
图5显示了学习到的融合模型的建议框。这些盒子是随机分布在图像上的，以覆盖整个图像区域。这保证了在候选人稀少的情况下的召回性能。

此外，每个阶段的级联头都会逐渐细化盒子的位置，并删除重复的盒子。这导致了高精确度的表现。图5还显示，稀疏RCNN在稀少和拥挤的情况下都表现出强劲的性能。对于稀有场景中的物体，其重复的盒子在几个阶段内就被移除。人群场景需要更多的阶段来完善，但最终每个物体都被精确和独特地检测到。

# 5. conclusion
我们提出了稀疏R-CNN，一种用于图像中物体检测的纯稀疏方法。提供了一个固定的稀疏的学习对象建议集，通过动态头进行分类和定位。最终的预测结果直接输出，没有非最大限度的抑制后程序。稀疏R-CNN证明了其准确性、运行时间和训练收敛性能与成熟的检测器相当。我们希望我们的工作能够启发人们重新思考密集先验的惯例，并探索下一代的物体检测器。
