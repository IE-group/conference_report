# **U-Net**

 论文地址：https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ 

代码地址：https://github.com/wolny/pytorch-3dunet

自2015年以来，在生物医学图像分割领域，U-Net得到了广泛的应用，该方法在2015年MICCAI会议上提出，目前已达到四千多次引用。至今，U-Net已经有了很多变体，目前已有许多新的卷积神经网络设计方式，但很多仍延续了U-Net的核心思想，加入了新的模块或者融入其他设计理念。U-Net架构在不同的生物医学分割应用中实现了非常好的性能。由于借助**具有弹性形变的数据增强功能**，它**只需要少量的的带标注的图像**，并且**在NVidia Titan GPU（6 GB）上仅需要10个小时的训练时间。**

 U-Net如下图所示，是一个**encoder-decoder结构**，左边一半的encoder包括若干卷积，池化，把图像进行下采样，右边的decoder进行上采样，恢复到原图的形状，给出每个像素的预测。

 具体来说，左侧可视为一个编码器，右侧可视为一个解码器。**编码器有四个子模块，**每个子模块包含两个卷积层，**每个子模块之后有一个通过max pool实现的下采样层**。输入图像的分辨率是572x572, 第1-5个模块的分辨率分别是572x572, 284x284, 140x140, 68x68和32x32。由于卷积使用的是valid模式，故这里**后一个子模块的分辨率等于（前一个子模块的分辨率-4）/2**。**解码器包含四个子模块**，分辨率通过上采样操作依次上升，直到与输入图像的分辨率一致（由于卷积使用的是valid模式，实际输出比输入图像小一些）。该网络还使用了跳跃连接，将上采样结果与编码器中具有相同分辨率的子模块的输出进行连接，作为解码器中下一个子模块的输入。

![img](https://img2020.cnblogs.com/blog/1137273/202005/1137273-20200518141120092-2115542025.png)

架构中的一个重要修改部分是**在上采样中还有大量的特征通道**，这些通道允许网络将上下文信息传播到具有更高分辨率的层。因此，拓展路径或多或少地与收缩路径对称，并产生一个**U形结构**。

在该网络中没有任何完全连接的层，并且仅使用每个卷积的有效部分，即**分割映射仅包含在输入图像中可获得完整上下文的像素**。该策略允许通过重叠平铺策略对任意大小的图像进行无缝分割，如图所示。**为了预测图像边界区域中的像素，通过镜像输入图像来推断缺失的上下文。**这种平铺策略对于将网络应用于大型的图像非常重要，否则分辨率将受到GPU内存的限制。

![img](https://img2020.cnblogs.com/blog/1137273/202005/1137273-20200518141156058-1497747740.png)

对于可用训练数据非常少的情况，可以通过**对可用的训练图像应用弹性变形来进行数据增强**。这使得网络学习这种变形的不变性，而不需要在标注图像语料库中看到这些变形。这在生物医学分割中尤其重要，因为变形曾是组织中最常见的变化，并且可以有效地模拟真实的变形。

许多细胞分割任务中的另一个挑战是分离同一类的接触目标。为此建议**使用加权损失，其中在接触单元之间分开的背景标签在损失函数中获得较大的权重。**

相比于FCN和Deeplab等，UNet共进行了4次上采样，并在同一个stage使用了skip connection，而不是直接在高级语义特征上进行监督和loss反传，这样就保证了最后恢复出来的特征图融合了更多的low-level的feature，也使得不同scale的feature得到了的融合，从而可以进行多尺度预测和DeepSupervision。4次上采样也使得分割图恢复边缘等信息更加精细。



# **3D U-Net**

 论文地址：https://arxiv.org/pdf/1606.06650.pdf

代码地址：https://github.com/wolny/pytorch-3dunet

 3D U-Net是U-Net的一个简单扩展，应用于**三维图像分割**，结构如下图所示。相比于U-Net，该网络**仅用了三次下采样操作，在每个卷积层后使用了batch normalization**，但3D U-Net和U-Net均没有使用dropout。![img](https://img2020.cnblogs.com/blog/1137273/202005/1137273-20200518141305347-1306622737.png)

为了避免瓶颈，**在上采样和下采样之前都将通道数增加为原来的二倍。**左侧的在进行 maxpooling 之前将通道数从 64 变为了 128；右侧的红色虚线内，在进行转置卷积之前，将通道数从 256 变为了 512。这种思想来源于 inception V3。使用加权的 softmax 损失函数，将没有标签的部分置为 0，让模型只从有有标签的部分学习。

# **Res-UNet 和Dense U-Net**

Res-UNet和Dense-UNet分别**受到残差连接和密集连接的启发，将UNet的每一个子模块分别替换为具有残差连接和密集连接的形式。**

将Res-UNet用于视网膜图像的分割，其结构如下图所示，其中灰色实线表示各个模块中添加的残差连接。

 ![](https://img2020.cnblogs.com/blog/1137273/202005/1137273-20200518141337083-1560618419.png)

 密集连接即**将子模块中某一层的输出分别作为后续若干层的输入的一部分，某一层的输入则来自前面若干层的输出的组合**。

下图是密集连接的一个例子。该文章中将U-Net的各个子模块替换为这样的密集连接模块，提出Fully Dense UNet 用于去除图像中的伪影。

 ![img](https://img2020.cnblogs.com/blog/1137273/202005/1137273-20200518141402662-658984445.png)

# Attention UNet

 

论文地址：https://arxiv.org/pdf/1804.03999.pdf

代码地址：https://github.com/ozan-oktay/Attention-Gated-Networks

 Attention UNet**在UNet中引入注意力机制，在对编码器每个分辨率上的特征与解码器中对应特征进行拼接之前，使用了一个注意力模块，重新调整了编码器的输出特征。**该模块生成一个门控信号，用来控制不同空间位置处特征的重要性，如下图中红色圆圈所示。

![img](https://img2020.cnblogs.com/blog/1137273/202005/1137273-20200518141510349-1083328229.png)

 该方法的注意力模块内部如下图所示，该模块通过1x1x1的卷积分别与ReLU和Sigmoid结合，生成一个权重图α, 通过与编码器中的特征相乘来对其进行校正。

 ![img](https://img2020.cnblogs.com/blog/1137273/202005/1137273-20200518141552368-382079176.png)

 下图展示了注意力权重图的可视化效果。从左至右分别是一幅图像和随着训练次数的增加该图像中得到的注意力权重。可见得到的注意力权重倾向于在目标器官区域取得大的值，在背景区域取得较小的值，有助于提高图像分割的精度。

![img](https://img2020.cnblogs.com/blog/1137273/202005/1137273-20200518141612452-887464440.png)