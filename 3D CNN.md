#### 一、3D CNN

3D网络的优越性：在不考虑计算和显存性能的情况下，**3d网络因为可以结合图像层间信息，能够保证隔层图像Mask之间的一个变化连续性**，效果会比2d的好。即使是层间距大的图像，我们预处理中是会有插值的，层间信息虽然薄弱但至少会比没有强。

3D网络存在明显的硬伤：3d网络的硬伤在于显存占用问题，这导致不可能将整个3d体素作为输入，**必须做crop，裁成一系列3d patch作为输入**。裁块会限制网络所能达到的最大感受野，导致丢失一定的全局信息，如果待分割目标本身比裁的块大很多的情况下，网络难以学习目标的整体结构信息。如果要做多目标的分割（多个目标仅位置不同，局部特征相同），分块学习的网络很难去区分这些目标。

3D网络也有很好的适用场景：3d网络比较适用于coarse to fine的两级分割结构，用于小目标的分割(脑肿瘤分割)。第一级用检测网络，或者下采样过的图像做粗定位选bounding box，第二级裁剪出bounding box作为3d网络的输入。

**总结的来说，3d网络因为显存的限制，导致其感受野有限，通常只能专注于细节和局部特征，适合作为第二级网络用于对细节做精优化。**3d相比于2d最大的好处是能结合层间信息，但是相比于2d网络性能要求太大。因此出现了很多能**结合层间信息（上下文信息）的2D网络，号称“2.5D”网络。**

以MICCAI 2018分割十项全能挑战赛冠军方案[nnU-Net](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1809.10486.pdf)在7个医学图像分割数据集上的实验结果为例，不同的任务上，2D、3D或者级联、集成等方案各有优势。整体来说，3D更有优势一些。

![img](https://pic2.zhimg.com/80/v2-9c77dab69c0ae5285eb2834f36a11383_1440w.jpg?source=1940ef5c)

结合两种网络的优点的两种思路： 

在2D网络中引入时序(LSTM,RNN)等。

- [Combining Fully Convolutional and Recurrent Neural Networks for 3D Biomedical Image Segmentation](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/6448-combining-fully-convolutional-and-recurrent-neural-networks-for-3d-biomedical-image-segmentation.pdf)
- [Recurrent Neural Networks for Aortic Image Sequence Segmentation with Sparse Annotations](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1808.00273.pdf)

2D卷积和3D卷积相结合。 

- [More Knowledge is Better: Cross-Modality Volume Completion and 3D+2D Segmentation for Intracardiac Echocardiography Contouring](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1812.03507.pdf)
- [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1711.11248.pdf)

在医疗影像分割，可研究2D和3D如何做融合。

##### 二、3D U-Net

论文链接：https://arxiv.org/abs/1606.06650 
代码链接：https://github.com/zhengyang-wang/3D-Unet--Tensorflow

加权损失函数和特殊数据增强功能使我们能够使用很少的手动注释切片（即来自稀疏注释的训练数据）来训练网络。本文的重点是它可以从零开始在稀疏注释的输入上进行训练，并且由于其无缝的切片策略而可以在任意大的模型上工作。

3D Unet网络的结构和2D Unet网络十分相似，只不过是把所有的2D操作全部替换成了3D操作。**除此以外的区别在于通道数翻倍的时机和反卷积操作。**在2D Unet中，通道数翻倍的时机在下采样后的第一次卷积时；而在3D Unet中，通道数翻倍发生在下采样或上采样前的卷积中。对于反卷积操作，区别在于通道数是否减半，2D Unet中通道数减半，而3D Unet中通道数不变

- 输入为132x132x116的三通道voxel，输出为44x44x28的voxels。

- 编码部分每个层包含两个3×3×3卷积，卷积层后使用BN+ReLU激活函数，然后加上2×2×2 max pooling，stride为2。

- 解码部分，每一层都有一个2x2x2的上卷积操作，stride为2，紧接着是2个3x3x3的卷积和BN+ReLU激活函数。**BN加快收敛和避免网络结构的瓶颈**

- 和2D-UNet相似的shortcut连接，为解码层提供高分辨率的特征。

- 在最后一层中，1×1×1卷积将输出通道数减少到标签数，并使用Softmax作为损失函数。

- 3D-UNet结构共有19069955个参数。

  3D Unet使用了旋转、缩放和灰度增强等**数据增强**方法，此外在训练数据和正确标注数据上运用平滑的密集变形场，即从一个标准差为4的正态分布中抽取随机向量，每个方向的间距为32个体素，然后应用b样条插值。

  3D Unet使用**带加权交叉熵损失**的softmax函数对网络输出和正确标注数据进行比较，对经常出现的背景减少权重，对标注到的图像数据部分增加权重，以平衡微管和背景体素对损失的影响。未标记的像素不参与损失计算，即权重为0。可以让网络可以更多地仅仅学习标注到的像素点，从而达到普适性地特点。

![1.png](https://aijishu.com/img/bVxtJ)

Github上有一些比较优秀的3D-UNet的开源实现：
1、https://github.com/shiba24/3d-unet（基于Pytorch实现）
2、[https://github.com/zhengyang-wang/3D-Unet--Tensorflow](https://github.com/zhengyang-wang/3D-Unet)（婴儿大脑图像分割）
3、https://github.com/shreyaspadhy/UNet-Zoo（各类U-Net汇总，包括3D U-Net）
4、https://github.com/tkuanlun350/3DUnet-Tensorflow-Brats18（3D Unet生物医学分割模型）



##### 三、V-Net

原文地址：[http://arxiv.org/abs/1606.04797](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1606.04797)

原文代码：[https://github.com/faustomilletari/VNet](https://link.zhihu.com/?target=https%3A//github.com/faustomilletari/VNet)

1. 基本上网络架构就是3D conv+residual Block版的U-Net，池化用卷积代替，转置卷积上采样

2. 提出了一个新的指标函数，类似IoU、Pa，叫做Dice coefficient。

采用端到端的训练方式，包含一个新式的目标函数用于训练时进行优化使用。同时能很好的处理背景和非背景之间的强烈不平衡问题。为了解决数据量有限的问题，使用了非线性变换和直方图匹配的方式来进行数据增强。

使用卷积操作来提取数据的特征，于此同时在每个“阶段”的末尾通过合适的步长来降低数据的分辨率。整个结构的左边是一个逐渐压缩的路径，而右边是一个逐渐解压缩的路径。最终输出的大小是和图像原始尺寸一样大的。所有的卷积操作都使用了合适的padding操作。

左边的压缩路径被分为了多个阶段，每个阶段都具有相同的分辨率。每个阶段都包含1到3个卷积层。为了使每个阶段学习一个参数函数：将每阶段的输入和输出进行相加以获得残差函数的学习。结合试验观察得知：这种结构为了确保在短时间内收敛需要一个未曾学过残差函数的相似性网络。

每个阶段的卷积操作使用的卷积核大小为5x5x5。在压缩路径一端，数据经过每个阶段处理之后会通过大小为2x2x2且步长为2的卷积核进行分辨率压缩。因此，每个阶段结束之后，特征图大小减半，这与池化层起着类似的作用。因为图像分辨率降低和残差网络的形式，从而将特征图的通道数进行的翻倍。整个网络结构中，均使用PReLu非线性激活函数。

使用卷积操作替代池化操作，在一些特殊的实现方式下可以在训练过程中减小内存的使用。这是因为在方向传播过程中并不需要像池化操作一样去切换输入和输出之间的映射，同时也更易于理解和分析。

下采样有利于在接下来的网络层中减小输入信号的尺寸同时扩大特征的感受野范围，下一层感受到的特征数量是上一层的两倍。

网络右边部分的功能主要是提取特征和扩展低分辨率的空间支持以组合必要的信息，从而输出一个两通道的体数据分割。这最后一个卷积层使用的卷积核大小是1x1x1，输出的大小与原输入大小一致。两个特征图通过这个卷积层来利用soft-max来生成前景和背景的分割概率图。在右边解压缩路径中每个阶段的最后，通过一个解卷积操作来恢复输入数据的大小。

于此同时，在收缩路径中每阶段的结果都会作为输入的一部分加入到右边解压缩对应的阶段中。这样就能够保留一部分由于压缩而丢失的信息，从而提高最终边界分割的准确性。同时这样有利于提高模型的收敛速度。

和U-Net类似，用skip connection传递细节信息，从而提高了最终预测的质量,这些skip connection改善了模型的收敛时间。

损失函数：网络预测由两个具有与原始输入数据相同分辨率的体素组成，并通过softmax层进行处理，该层输出每个体素属于前景和背景的概率。在我们的研究中，感兴趣的解剖一般仅占据很小的扫描区域。这通常会使学习过程陷入损失函数的局部最小值，从而产生一个网络，其预测强烈地倾向背景。结果，前景区域经常丢失或仅被部分检测。先前的几种方法都采用基于样本重加权的损失函数，在学习过程中，前景区域比背景区域具有更高的重要性。 在这项工作中，我们提出了一个基于dice coefficient的目标函数，该函数的值在0到1之间，我们的目标是最大化。

Dice coefficient ![[公式]](https://www.zhihu.com/equation?tex=D%3D%5Cfrac%7B2%5Csum_%7Bi%7D%5ENp_ig_i%7D%7B%5Csum_%7Bi%7D%5ENp_i%5E2%2B%5Csum_%7Bi%7D%5ENg_i%5E2%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=N) 为像素的总数， ![[公式]](https://www.zhihu.com/equation?tex=p%E3%80%81g) 可以是3D体素，也可以是2维像素。总体概括，就是两个矩阵按位相乘后整体求和再乘以2，除以2个矩阵的按位平方后的总和。

![å¾2-æä»¬çç½ç»æ¶æçç¤ºæå¾ã æä»¬èªå®ä¹çcaffe [8]å®ç°éè¿æ§è¡ä½ç§¯å·ç§¯æ¥å¤ç3Dæ°æ®ã æå¥½ä»¥çµå­æ ¼å¼æ¥çã](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/7784491/7785060/7785132/7785132-fig-2-source-small.gif)

##### 四、文献调研

3D CNN medical segmentation  26800

[V-net：用于体积医学图像分割的全卷积神经网络](https://ieeexplore.ieee.org/abstract/document/7785132/)    （2014年，2425被引）

 [具有完全连接的CRF的](https://www.sciencedirect.com/science/article/pii/S1361841516301839)[高效多尺度3D CNN，可进行准确的脑部病变分割](https://www.sciencedirect.com/science/article/pii/S1361841516301839)（2017年1545被引）

[3D深监督网络进行自动分割的体积的医用图像](https://www.sciencedirect.com/science/article/pii/S1361841517300725)（2017年，262被引）

[级联3D全卷积网络在医学图像分割中的应用](https://www.sciencedirect.com/science/article/pii/S0895611118301472)（2015年，93被引）



![image-20201019221351340](C:\Users\16018\AppData\Roaming\Typora\typora-user-images\image-20201019221351340.png)