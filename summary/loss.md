# 0.目标检测loss简述

# 1.分类loss
## 1.1交叉熵损失CE

## 1.2 focal loss
### motivation
在one-stage检测算法中，会出现正负样本数量不平衡以及难易样本数量不平衡的情况，为了解决则以问题提出了focal loss。

hit的检测框就是正样本。容易的正样本是指置信度高且hit的检测框，困难的负样本就是置信度低但hit的检测框，容易的负样本是指未hit且置信度低的检测框，困难的负样本指未hit但置信度高的检测框。
### solution
目的是解决样本数量不平衡的情况 - 正样本loss增加，负样本loss减小 - 难样本loss增加，简单样本loss减小

# 2.回归loss

## 2.1 L1 loss
$$
loss(x,y) = \frac{1}{n} \sum_{i=1}^{n} \rvert y_i - f(x_i)\rvert
$$
- l1 loss在零点不平滑，用的较少。一般来说，l1正则会制造稀疏的特征，大部分无用的特征的权重会被置为0。

- （适合回归任务，简单的模型，由于神经网络通常解决复杂问题，很少使用。）

- L1 loss的鲁棒性(抗干扰性)比L2 loss强。概括起来就是L1对异常点不太敏感，而L2则会对异常点存在放大效果。因为L2将误差平方化，当误差大于1时，误会会放大很多，所以使用L2 loss的模型的误差会比使用L1 loss的模型对异常点更敏感。如果这个样本是一个异常值，模型就需要调整以适应单个的异常值，这会牺牲许多其它正常的样本，因为这些正常样本的误差比这单个的异常值的误差小。如果异常值对研究很重要，最小均方误差则是更好的选择。

- L1 loss 对 x(损失值)的导数为常数，在训练后期，x较小时，若学习率不变，损失函数会在稳定值附近波动，很难收敛到更高的精度。

- L2 loss的稳定性比L1 loss好。概括起来就是对于新数据的调整，L1的变动很大，而L2的则整体变动不大。

### L1不可导？
当损失函数不可导,梯度下降不再有效,可以使用坐标轴下降法。
梯度下降是沿着当前点的负梯度方向进行参数更新；
而坐标轴下降法是沿着坐标轴的方向；
假设有m个特征个数,坐标轴下降法进行参数更新的时候,先固定m-1个值,然后再求另外一个的局部最优解,从而避免损失函数不可导问题。

使用Proximal Algorithm对L1进行求解,此方法是去优化损失函数上界结果。

## 2.2 L2 loss
$$
loss(x,y) = \frac{1}{n}\sum_{i=1}^{n}(y_i-f(x_i))^2
$$
- 对离群点比较敏感，如果feature是unbounded的话，需要好好调整学习率，防止出现梯度爆炸的情况。l2正则会让特征的权重不过大，使得特征的权重比较平均。

- （适合回归任务，数值特征不大，问题维度不高）

- 从L2 loss的图像可以看到，图像(上图左边红线)的每一点的导数都不一样的，离最低点越远，梯度越大，使用梯度下降法求解的时候梯度很大，可能导致梯度爆炸。

- L1 loss一般用于简单的模型，但由于神经网络一般是解决复杂的问题，所以很少用L1 loss，例如对于CNN网络，一般使用的是L2-loss，因为L2-loss的收敛速度比L1-loss要快。如下图：

## 2.3 Smooth L1 loss
$$
loss(x,y)=\frac{1}{n}\sum_{i=1}^n 
\begin{cases}
0.5 * (y_i-f(x_i))^2, & if\rvert y_i-f(x_i)\rvert <1 \\
\lvert y_i-f(x_i) \rvert - 0.5, & otherwise
\end{cases}
$$
- 修改零点不平滑问题，L1-smooth比l2 loss对异常值的鲁棒性更强。具有l1和l2的优点，当绝对差值小于1，梯度不至于太大，损失函数较平滑，当差别大的时候，梯度值足够小，较稳定，不容易梯度爆炸。

- （回归，当特征中有较大的数值，适合大多数问题）

- 当预测值和ground truth差别较小的时候（绝对值差小于1），其实使用的是L2 Loss，当绝对值差小于1时，由于L2会对误差进行平方，因此会得到更小的损失，有利于模型收敛。而当差别大的时候，是L1 Loss的平移，因此相比于L2损失函数，其对离群点（指的是距离中心较远的点）、异常值（outlier）不敏感，可控制梯度的量级使训练时不容易跑飞。。SooothL1Loss其实是L2Loss和L1Loss的结合，它同时拥有L2 Loss和L1 Loss的部分优点。

![smmoth l1](images/loss/1.png)

**Smooth L1**

当预测框与 ground truth 差别过大时，梯度值不至于过大；
当预测框与 ground truth 差别很小时，梯度值足够小。

应用在目标检测中，位置损失为：
$$
L_{loc}(t^u,v)=\sum_{i\in\lbrace x,y,w,h\rbrace} smooth_{L_1}(t_i^u-v_i)
 $$
其中$v=(v_x,v_y,v_w,v_h)$表示bbox的真实值，$t^u=(t_x^u,t_y^u,t_w^u,t_h^u)$表示bbox的预测值

### 缺点
L1/L2/smooth l1 在计算目标检测的 bbox loss时，都是独立的求出4个点的 loss，然后相加得到最终的 bbox loss。这种做法的默认4个点是相互独立的，与实际不符。举个例子，当(x, y)为右下角时，w h其实只能取0;目标检测的评价 bbox 的指标是 IoU，IoU 与smooth L1 loss 的变化不匹配
## 2.4 IOU loss

### 简述

Yu, J., Jiang, Y., Wang, Z., Cao, Z., Huang, T., 2016. UnitBox: An Advanced Object Detection Network, in: Proceedings of the 24th ACM International Conference on Multimedia. pp. 516–520. https://doi.org/10.1145/2964284.2967274


IoU 计算让 x, y, w, h 相互关联，同时具备了尺度不变性，克服了 smooth l1 Loss 的缺点。

尺度不变性，也就是对尺度不敏感（scale invariant）， 在regression任务中，判断predict box和gt的距离最直接的指标就是IoU。(满足非负性；同一性；对称性；三角不等性)
### 流程
![](images/loss/3.png)
计算过程如下：
对每个像素点而言，定义一个四维向量
$$
\widetilde{\boldsymbol{x}}_{i, j}=\left(\widetilde{x}_{t_{i, j}}, \widetilde{x}_{b_{i, j}}, \widetilde{x}_{l_{i, j}}, \widetilde{x}_{r_{i, j}}\right)
$$
到gt和prediction上下左右的距离。

具体的计算过程如下。
![](images/loss/4.png)
### 缺点

当预测框和目标框不相交，即 IoU(bbox1, bbox2)=0 时，不能反映两个框距离的远近，此时损失函数不可导，IoU Loss 无法优化两个框不相交的情况。假设预测框和目标框的大小都确定，只要两个框的相交值是确定的，其 IoU 值是相同时，IoU 值不能反映两个框是如何相交的，如图所示：

![](images/loss/2.jpg)

灰色框为真实框，虚线框为预测框。这两者情况的IoU相同，但是这两个框的匹配状态不一样。我们认为右边框匹配的好一点，因为它匹配的角度更好。故下文定义了GIoU。 


## 2.5 GIOU loss

![](images/loss/5.png)
Two sets of examples (a) and (b) with the bounding boxes represented by (a) two corners (x1, y1, x2, y2) and (b) center and size (xc, yc, w, h).



### 通用流程
![](images/loss/6.png)

该计算过程可以是:交并比+并集的倒数，因为希望并集最小。

GIoU 的实现方式如上式，其中 C 为 A 和 B 的外接矩形。用 C 减去 A 和 B 的并集除以 C 得到一个数值，然后再用 A 和 B 的 IoU 减去这个数值即可得到 GIoU 的值。可以看出：

GIoU 取值范围为 [-1, 1]，在两框重合时取最大值1，在两框无限远的时候取最小值-1；与 IoU 只关注重叠区域不同，GIoU不仅关注重叠区域，还关注其他的非重合区域，能更好的反映两者的重合度。

### 包围框回归的流程
![](images/loss/8.png)
### IOU和GIOU的对比
![](images/loss/7.png)

### 缺点
![](images/loss/9.jpg)

当目标框完全包裹预测框的时候，IoU 和 GIoU 的值都一样，此时 GIoU 退化为 IoU, 无法区分其相对位置关系。

## 2.6 DIOU loss
### GIOU和DIOU回归过程的比较
![](images/loss/10.png)

GIOU退化为IOU的例子
![](images/loss/11.png)

IOU loss可以定义为下式：

$$
\mathcal{L}=1-I o U+\mathcal{R}\left(B, B^{g t}\right)
$$
DIOU的惩罚项定义为：
$$
\mathcal{R}_{D I o U}=\frac{\rho^{2}\left(\mathbf{b}, \mathbf{b}^{g t}\right)}{c^{2}}
$$
DIOU的损失函数定义为：
$$
\mathcal{L}_{D I o U}=1-I o U+\frac{\rho^{2}\left(\mathbf{b}, \mathbf{b}^{g t}\right)}{c^{2}}
$$
其中c、d的定义如图所示。
![](images/loss/12.png)
**c放在分母上？**
### 进而提出的DIOU NMS
Distance IOU
$$
s_{i}=\left\{\begin{array}{l}
s_{i}, I o U-\mathcal{R}_{D I o U}\left(\mathcal{M}, B_{i}\right)<\varepsilon \\
0, \quad \operatorname{IoU}-\mathcal{R}_{D I o U}\left(\mathcal{M}, B_{i}\right) \geq \varepsilon
\end{array}\right.
$$
### 缺点
边框回归的三个重要几何因素：重叠面积、中心点距离和长宽比，DIoU 没有包含长宽比因素。
## 2.7 CIOU loss
Complete IOU
### CIOU惩罚项：
$$
\mathcal{R}_{C I o U}=\frac{\rho^{2}\left(\mathbf{b}, \mathbf{b}^{g t}\right)}{c^{2}}+\alpha v
$$

$\alpha$是正权衡参数,$v$是纵横比的一致性。
$$
v=\frac{4}{\pi^{2}}\left(\arctan \frac{w^{g t}}{h^{g t}}-\arctan \frac{w}{h}\right)^{2}
$$
$$
\alpha=\frac{v}{(1-I o U)+v}
$$

arctan图像：

![](images/loss/13.jpg)

### v求导：
$$
\begin{aligned}
\frac{\partial v}{\partial w} &=\frac{8}{\pi^{2}}\left(\arctan \frac{w^{g t}}{h^{g t}}-\arctan \frac{w}{h}\right) \times \frac{h}{w^{2}+h^{2}} \\
\frac{\partial v}{\partial h} &=-\frac{8}{\pi^{2}}\left(\arctan \frac{w^{g t}}{h^{g t}}-\arctan \frac{w}{h}\right) \times \frac{w}{w^{2}+h^{2}}
\end{aligned}
$$

### CIOU loss定义：
$$
\mathcal{L}_{C I o U}=1-I o U+\frac{\rho^{2}\left(\mathbf{b}, \mathbf{b}^{g t}\right)}{c^{2}}+\alpha v
$$

### 缺点：
在CIoU的定义中，衡量长宽比的$v$过于复杂，从两个方面减缓了收敛速度：
长宽比不能取代单独的长宽，比如$w=k w^{g t}, h=k h^{g t}$都会导致v=0；
从v的导数可以得到$\frac{\partial v}{\partial w}=-\frac{h}{w} \frac{\partial v}{\partial h}$，这说明$\frac{\partial v}{\partial w}$和$\frac{\partial v}{\partial h}$在优化中意义相反。
- 只是反映了长宽比的差异，而不是W和WGT或H和HGT之间的真实关系。也就是说，所有具有属性的框{（W = kwgt，h = khgt）|k∈R+}的v = 0，这与现实不一致。
- 我们有∂v∂w=  -  hw∂v∂h。 ∂v∂w和∂v∂h具有相反的符号。因此，在任何时候，如果这两个变量（w或h）中有一个，另一个变量将减少。这是不合理的，尤其是当w <wgt和h < hgt或w> wgt和hgt时。
- 由于V仅反映了长宽比的差异，因此CIOU损失可能以不合理的方式优化相似性。如图1所示，目标框的比例设置为WGT = 1和HGT = 1。 50次迭代后，锚盒的尺度回归为W = 1.64，H = 2.84。在这里，CIOU的损失确实增加了纵横比的相似性，而它阻碍了模型有效地降低（W，H）和（WGT，HGT）之间的真实差异。
## 2.8 EIOU loss
### 和GIOU CIOU的比较
![](images/loss/14.png)

### EIOU定义
$$
\begin{aligned}
&L_{E I O U}=L_{I O U}+L_{d i s}+L_{a s p} \\
&\quad=1-I O U+\frac{\rho^{2}\left(\mathbf{b}, \mathbf{b}^{\mathrm{gt}}\right)}{c^{2}}+\frac{\rho^{2}\left(w, w^{g t}\right)}{C_{w}^{2}}+\frac{\rho^{2}\left(h, h^{g t}\right)}{C_{h}^{2}}
\end{aligned}
$$

$C_w$为最小包围框的宽，高同理。

## 2.9 focal EIOU loss

### 2.9.1 focal L1 loss

### 2.9.2 focal EIOU loss


## 2.10 seasaw loss