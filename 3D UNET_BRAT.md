https://github.com/tkuanlun350/3DUnet-Tensorflow-Brats18

在此基础上提炼出Keras代码

地址：http://47.101.153.124:8081/tree/GuoChuang/3D-unet-keras-Brats2019-master

##### 一、网络结构

四层的3DUNet，增加了深监督

http://47.101.153.124:8081/notebooks/GuoChuang/3D-unet-keras-Brats2019-master/train/model.ipynb

##### 二、2D/[3D U-Net](https://github.com/wolny/pytorch-3dunet)

![image-20201027115304461](C:\Users\16018\AppData\Roaming\Typora\typora-user-images\image-20201027115304461.png)

![1.png](https://aijishu.com/img/bVxtJ)

##### 三、深监督

https://zhuanlan.zhihu.com/p/49492520

![image-20201026233733059](C:\Users\16018\AppData\Roaming\Typora\typora-user-images\image-20201026233733059.png)

##### 四、dice

generalised dice_loss：https://arxiv.org/abs/1707.03237

    prod = np.multiply(s, g)
    # s: segmentation volume；g: ground truth volume
    
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
    #后面的常数项保证loss的稳定性，避免被0除 
    
    if(type_idx ==0): # whole tumor（WT）
                temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
                dice_one_volume = [temp_dice]
    elif(type_idx == 1): # tumor core（TC）
                s_volume[s_volume == 2] = 0
                g_volume[g_volume == 2] = 0
                temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
                dice_one_volume = [temp_dice]
    else:
               # dice of each class（ET）
                temp_dice = binary_dice3d(s_volume == 4, g_volume == 4)
                dice_one_volume = [temp_dice]
            dice_all_data.append(dice_one_volume)
##### 五、批归一化

https://medium.com/techspace-usict/normalization-techniques-in-deep-neural-networks-9121bf100d8

https://blog.csdn.net/weixin_37737254/article/details/102920430

##### 六、回调函数

提前终止：https://blog.csdn.net/u012587076/article/details/78702526

```
   '''
    提前终止，当被监测的值不再提升，则停止训练
    patience: 没有进步的训练轮数，在这之后训练就会被停止
    '''
    early = EarlyStopping(monitor = monitor,mode = mode,patience = early_p)
```

学习率衰减：https://cloud.tencent.com/developer/article/1488834

    '''
    当评估标准停止提升时，降低学习速率
    factor: 0.5 学习速率被降低的因数。新的学习速率 = 学习速率 * 因数（0.5）
    patience: 没有进步的训练轮数，在这之后训练速率会被降低
    cooldown: 2 在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量。
    min_lr: 1e-6 学习速率的下边界。
    '''       
    reduceLROnPlat = ReduceLROnPlateau(monitor = monitor,factor = factor, 
                     patience = reduce_lr_p, verbose = 1, mode = mode, 
                     epsilon = min_delta,  cooldown = cooldown, min_lr = min_lr)


七、使用tensorpack的数据流来提高io速度

参见[train.py](http://47.101.153.124:8081/edit/GuoChuang/3D-unet-keras-Brats2019-master/train/train.py)

`tensorpack`的`dataflow`模块使用多进程和多线程加载数据，使用`CPU`和`GPU`队列，隐藏了数据转移到内存和显存的延迟，十分高效。运行的速度比同等的keras代码要快1.2~5倍。同时提供了很多公开数据集的读取接口，方便实验。

https://tensorpack.readthedocs.io/modules/tfutils.html#module-tensorpack.tfutils.scope_utils

##### 八、图片展示

https://github.com/NifTK/NiftyNet/tree/dev/demos/BRATS17/example_outputs

<img src="C:\Users\16018\AppData\Roaming\Typora\typora-user-images\image-20201027130947235.png" alt="image-20201027130947235" style="zoom:50%;" />

<img src="C:\Users\16018\AppData\Roaming\Typora\typora-user-images\image-20201027131021364.png" alt="image-20201027131021364" style="zoom:50%;" />

<img src="C:\Users\16018\AppData\Roaming\Typora\typora-user-images\image-20201027131041051.png" alt="image-20201027131041051" style="zoom:50%;" />

https://github.com/taigw/brats17/

An example of brain tumor segmentation result.

![image-20201027131240740](C:\Users\16018\AppData\Roaming\Typora\typora-user-images\image-20201027131240740.png)



https://arxiv.org/pdf/1709.03485.pdf

<img src="C:\Users\16018\AppData\Roaming\Typora\typora-user-images\image-20201027121636217.png" alt="image-20201027121636217" style="zoom:67%;" />