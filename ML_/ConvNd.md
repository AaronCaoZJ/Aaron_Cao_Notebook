# Conv1d vs Conv2d

## 应用场景

1. 图像数据，一般是三维的 $(W,H,C)$

2. 序列数据，一般是二维的 $(L,D)$

$C$ 代表图像的通道数，$D$ 代表词向量的维度。这么一看，图像中每个像素点向量和文本中词向量是对应的。

## Conv1d 的直观效果

一维卷积不代表卷积核只有一维，也不代表被卷积的feature也是一维。一维的意思是说卷积的方向是一维的。

```python
class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# in_channels(int) – 输入信号的通道。在文本分类中，即为词向量的维度
# out_channels(int) – 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
# kernel_size(int or tuple) - 卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
# stride(int or tuple, optional) - 卷积步长
# padding (int or tuple, optional)- 输入的每一条边补充0的层数
# dilation(int or tuple, `optional``) – 卷积核元素之间的间距
# groups(int, optional) – 从输入通道到输出通道的阻塞连接数
# bias(bool, optional) - 如果bias=True，添加偏置
```

![5faf64f4eb86dd37121774c720877b1d44f7f617](assets/5faf64f4eb86dd37121774c720877b1d44f7f617-1712564665224-2.gif)
