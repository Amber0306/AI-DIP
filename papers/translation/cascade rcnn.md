cascade rcnn

# 摘要

在物体检测中，需要一个交集大于联合（IoU）的阈值来定义阳性和阴性。阈值来定义阳性和阴性。一个物体检测器，用低的IoU阈值训练，例如0.5。通常会产生嘈杂的检测结果。然而，随着IoU阈值的增加，检测性能趋于下降。造成这种情况的主要因素有两个。1）训练期间的过度训练期间的过度拟合，由于指数级消失的正它的样本，和2）推理时间的不匹配。探测器是最佳的IoU，而那些在假设。一个多阶段的物体检测结构级联R-CNN，是为了解决这些问题而提出的。它由一连串的检测器组成，这些检测器的训练是以不断提高的IoU阈值进行训练，以依次提高对接近假阳性的选择性。检测器是一个阶段一个阶段地训练的，利用的是这样的观察，即检测器的输出是一个很好的分布，用于训练下一个更高质量的检测器。逐步改进的假设的再抽样保证了所有的检测器都有一个同等大小的正向例子集，减少了缩小过拟合问题。同样的级联程序在推理中应用，使每个阶段的假设和检测器质量之间的匹配更加紧密。假设和每个阶段的检测器质量之间更加匹配。A级联R-CNN的简单实现被证明是在具有挑战性的COCO数据集上超过了所有的单模型物体检测器。实验还表明，Cas-cade R-CNN广泛适用于各种检测器结构，实现了与基线探测器强度无关的一致收益。该代码将在以下网站提供
https://github.com/zhaoweicai/cascade-rcnn。



# 简介

目标检测是一个复杂的问题，需要完成两个主要任务.首先，检测器必须解决识别问题，区分前景对象和背景，并为它们分配合适的对象类别标签。第二，检测器必须解决定位问题，为不同的对象分配精确的边界框(b-box)。这两种方法都是特别困难的，因为检测器面临许多“相似的”错误，对应于“相似但不正确”的边界框。检测器必须在消除这些相似假阳性的同时找到真阳性。

最近提出的许多目标检测器是基于两阶段R-cnn框架[12，11，27，21]，其中检测是一个结合分类和边界框回归的多任务学习问题。与目标识别不同的是，需要一个交并比(IOU)阈值来定义正/负。然而，通常使用的阈值u(通常u=0.5)对正项的要求相当宽松。产生的检测器经常产生噪声边界框（FP），如图1(A)所示。假设大多数人会经常考虑相似假阳性，通过IOU≥0.5测试。虽然在u=0.5准则下汇集的例子丰富多样，但它们使训练能够有效地拒绝相似假阳性的检测器变得困难。

在本工作中，我们将假设的质量定义为其与真值框的IOU，并将探测器的质量定义为用于训练它的IOU阈值。我们的目标是研究到目前为止学习高质量对象检测器的研究不足的问题，它的输出很少包含相似的假阳性，如图1(B)所示。其基本思想是，单个检测器只能是单个质量级别的最优检测器。这是众所周知的成本敏感的学习文献[7，24]，其中最优的不同点的接收操作特性(ROC)需要不同的损失函数。主要区别在于我们考虑的是给定IOU阈值的优化，而不是假阳性率。

图1©和(D)说明了这一想法，它们分别介绍了以IOU阈值u=0.5、0.6、0.7训练的三个探测器的定位和检测性能。定位性能被评估为输入提案的IOU的函数，检测性能是IOU阈值的函数，如COCO[20]。请注意，在图1©中，每个边界框回归器对于这些IOU接近于检测器训练阈值IOU的示例的性能最好。这也适用于检测性能，直到过度拟合。图1(D)显示，对于低IOU示例，u=0.5的检测器优于u=0.6的检测器，但在较高的IOU级别上表现不佳。一般来说，在单一IOU水平上优化的检测器并不一定是其他级别的最佳检测器。这些观察表明，高质量的检测要求检测器与其所处理的假设之间进行更密切的质量匹配。一般来说，只有在给出高质量的建议时，检测器才能具有高质量。

然而，为了制造出高质量的探测器，仅仅在训练中增加u是不够的。事实上，如图1(D)中u=0.7的检测器所见，这会降低检测性能。问题是，假设在提案检测器之外的分布通常是严重不平衡的低质量。一般来说，强迫较大的IOU阈值会导致阳性训练样本以指数方式减小。对于神经网络来说，这是一个特别大的问题，因为神经网络的例子非常密集，这使得“高u”训练策略很容易被过度适用。另一个困难是检测器的质量与推断的测试假设的质量不匹配。如图1所示，高质量的检测器必然是高质量假设的最佳选择。当要求他们研究其他质量水平的假设时，检测可能是次优的。

在本文中，我们提出了一种新的探测器结构，Cascade  R-CNN，以解决这些问题.这是R-cnn的多阶段扩展，检测器的级联阶段越深，对相似假阳性就有更多的选择性。R-CNN级的级联是按顺序训练的，使用一个阶段的输出来训练下一个阶段。这是因为观察到回归器的输出IOU几乎总是优于输入IOU。这个观察可以在图1©中进行，其中所有的线都在灰色线之上。结果表明，用一定的IOU阈值训练的检测器的输出是训练下一次较高IOU阈值检测器的良好分布。这类似于在对象分析文献[31，8]中常用的用于组装数据集的引导方法[31，8]。主要的区别在于，Cascade  R-CNN的重采样程序并不是为了挖掘硬负面。相反，通过调整边界框，每一阶段的目标是找到一组好的相似假阳性来训练下一阶段。当以这种方式操作时，适应于越来越高的IoU的一系列检测器可以克服过度拟合的问题，从而得到有效的训练。在推理时，采用相同的级联过程。逐步改进的假设在每个阶段都能更好地与不断提高的探测器质量相匹配。如图1©和(D)所示，这使检测精度更高。

级联R-CNN的实施和进行端到端训练是相当简单的。我们的结果表明，在具有挑战性的COCO检测任务[20]上，一个没有任何花哨的普通实现在很大程度上超过了所有以前的最先进的单模探测器，特别是在更高质量的评估指标下。此外，基于R-CNN框架的任何两级目标检测器都可以建立Cascade  R-CNN。我们观察到了一致的增益(2∼4点)，在计算上略有增加。这种增益与基线目标检测器的强度无关。因此，我们相信，这种简单而有效的检测架构对于许多对象检测研究工作都是有意义的。

# 相关工作

由于R-cnn[12]体系结构的成功，通过将提议检测器和区域分类器结合起来的检测问题的两阶段公式在最近已经成为主流。为了减少R-CNN中多余的cnn计算量，SPP-net[15]和Fast-RCNN[11]引入了区域特征提取的思想，大大加快了整个检测器的速度。后来，Faster  RCNN[27]通过引入区域提案网络(RPN)实现了进一步加速。该体系结构已成为一个领先的对象检测框架。最近的一些工作已经将它扩展到解决各种细节问题。例如，R-FCN[4]提出了有效的没有准确度损失的区域方向的完全卷积，以避免对Faster  RCNN进行繁重的区域CNN计算；而MS-CNN[1]和FPN[21]则在多输出层检测提案，以缓解RPN接收字段与实际对象大小之间的规模不匹配，从而实现高召回建议检测。

或者，单阶段目标检测架构也变得流行起来，主要是因为它们的计算效率。这些体系结构接近经典的滑动窗口策略[31，8]。YOLO[26]通过转发输入图像来输出非常稀疏的检测结果，当使用高效的骨干网络实现时，它使实时目标检测具有公平的性能。SSD[23]以类似于RPN[27]的方式检测对象，但使用不同分辨率的多个特征映射在不同的尺度上覆盖对象。这些结构的主要限制是它们的精度通常低于两级探测器的精度。最近，RetinaNet[22]被提出来解决密集目标检测中的极端前景-背景类不平衡问题，取得了比最先进的两级目标检测器更好的结果。

在多阶段目标检测中的一些探索也已经被提出了。多区域检测器[9]引入了迭代边界框回归，其中多次应用R-CNN来产生更好的边界框，CRAFT[33]和AttractioNet[10]使用多级程序生成精确的建议，并将它们传递给FAST-RCNN。[19，25]在对象检测网络中嵌入了经典的[31]级联结构。[3]交替地迭代检测和分割任务，例如实例分割。

# 检测

在本文中，我们扩展了Faster  RCNN[27，21]的两阶段体系结构，如图3(A)所示。第一阶段是一个提案子网络(H0)，应用于整个图像，产生初步的检测假设，被称为目标提案。在第二阶段，这些假设然后被一个感兴趣的区域检测子网络(H1)处理，表示为检测头。每个假设都有一个最终的分类分数(“C”)和一个边框(“B”)。我们专注于多阶段检测子网络的建模，并采用但不限于RPN[27]来进行提案检测。


