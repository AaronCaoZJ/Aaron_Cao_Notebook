> Referenc:
>
> https://www.armcvai.cn/2024-11-28/llava-structure.html
>
> https://zhuanlan.zhihu.com/p/28971454220



---



# Background

## Vison Transformer - ViT

![image-20250805105336001](assets/image-20250805105336001.png)

一个`ViT Block`包括五个部分：

1. `Patch Embeddings`：图像划分成大小固定的patch，patch展平为向量，经过线性投影
2. `Position Embeddings`：位置编码，提供Transformer每个patch在图像中的位置
3. `Transformer Encoder`：与NLP类似，包含Multi-Head Self Attention & Feed-Forward Network（FFN，激活函数为GLUE）
4. `Classification Head`：通常将Transformer输出的第一个token传入全连接层MLP
5. `Layer Normalization and Skip Connections`：在每个子层之后使用层归一化和残差链接

## Diffusion

在连续场景下，前向加噪过程表示为：
$$
q(\mathbf{x}_t|\mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_t|\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})
$$
对于离散为K个类别的场景则用随机转移矩阵表示：
$$
q(\mathbf{x}_t|\mathbf{x}_{t-1})=Cat(\mathbf{x}_t|\mathbf{x}_{t-1}\mathbf{Q}_t)
$$

$$
q(\mathbf{x}_t|\mathbf{x}_0)=Cat(\mathbf{x}_t|\mathbf{x}_0\overline{\mathbf{Q}}_t)
$$

该过程是马尔科夫的，由贝叶斯公式计算得：
$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)=q(\mathbf{x}_{t-1}|\mathbf{x}_t)=Cat(\mathbf{x}_t|\frac{\mathbf{x}_tQ_t^\mathrm{T}⊙\mathbf{x}_0\overline{\mathbf{Q}}_{t-1}}{\mathbf{x}_0\overline{\mathbf{Q}}_t\mathbf{x}_t^\mathrm{T}})
$$

> 哈达玛积
>
> $$
> \left(\begin{array}{lll}a_{11} & a_{12} & a_{13} \\a_{21} & a_{22} & a_{23} \\a_{31} & a_{32} & a_{33}\end{array}\right) \odot\left(\begin{array}{lll}b_{11} & b_{12} & b_{13} \\b_{21} & b_{22} & b_{23} \\b_{31} & b_{32} & b_{33}\end{array}\right)=\left(\begin{array}{llll}a_{11} b_{11} & a_{12} b_{12} & a_{13} b_{13} \\a_{21} b_{21} & a_{22} b_{22} & a_{23} b_{23} \\a_{31} b_{31} & a_{32} b_{32} & a_{33} b_{33}\end{array}\right)
> $$



---



常见模型架构是`视觉编码器`+`映射层`+`LLM`，如VILA-1.5，LLaVA，QWen-VL等

# LLaVA系列

## LLaVA

![image-20250805145302352](assets/image-20250805145302352.png)
$$
H_V=W \cdot Z_V,\ where\ Z_V=g(X_V)
$$

### Architecture

预训练CLIP模型的视觉编码器`ViT-L/14`+`线性映射层`+`LLama or Vicuna(基于LLama2微调)`

> `ViT-B`（Base）：通常有 12 层 Transformer
>
> `ViT-L`（Large）：通常有 24 层 Transformer
>
> `ViT-H`（Huge）：通常有 32 层 Transformer
>
> `ViT-L/14`表示ViT会将图像分割成14x14的patch

### Training

1. 图像和语言特征对齐：冻结Vision Encoder和LLM，预训练Projection权重
2. 指令微调：使用ChatGPT生成的具有上下文的数据ScienceQA数据集，微调投影矩阵和LLM

## LLaVA-1.5-HD

### Improvements vs LLaVA

1. 使用2层的线性投影层
2. 输入图像的分辨率从224x224提高到336x336（CLIP支持的分辨率上限）
3. 使用分辨率更大的图像，先将其分为若干与视觉编码器一致的块，依此编码后合并为大特征图输入LLM，同时将下采样（resize）的原始图像合并到特征图中![image-20250805154833834](assets/image-20250805154833834.png)

4. 对于期望回答的长短进行明确的指令提示
5. 使用更多样化数据集训练

### Experiment

* LLaVA-1.5预训练和训练数据量小，但效果更好

* 提升分辨率有助于模型减少幻觉

## LLaVA-NeXT/1.6

支持更大更多种分辨率

## LLaVA-OneVision

### Improvements

1. 更多样化数据（>1M）~9M
2. 训练中第一、二步之间使用高质量的新数据微调模型，并在第二步中依此使用单个图片、视频、多图片上指令微调
3. 使用插值方式减少高清图像、多帧视频的token数量



# Show-o

![image-20250805175558063](assets/image-20250805175558063.png)

> **Q: can such one single transformer involve both autoregressive and diffusion modeling?**
>
> 文本天然适合自回归模型（AR），其嵌入空间是连续的，但处理的token是离散；图像生成的常用的扩散模型，图像表征方式可以是连续的像素值or离散的token。两种表征范式需要统一。
>
> 相比于使用额外的Diffusion模块，使用一个Transformer融合自回归建模和扩散模型。

## Tokenization

* Text tokenizer：直接使用pre-trained LLM

* Image tokenizer：借鉴`MAGVIT-v2`，使用3500w张图像训练一个大小为8192的码本，可将分辨率256x256的图像编码为16x16的离散token

  ![image-20250805220754732](assets/image-20250805220754732.png)

## Architecture

继承常规的LLM结构，关键调整包括：

1. 在每个注意力层前增加`QK-Norm`操作
2. 在原有文本token的嵌入层基础上扩展了8192个可学习嵌入，用于离散的图像token

## Unified Prompting

![image-20250806003901550](assets/image-20250806003901550.png)

## Omni-Attention Mechanism

对于文本使用因果注意力，对于图像使用全注意力

![image-20250806010304316](assets/image-20250806010304316.png)

## Training Objectives

为同时实现自回归建模和离散扩散建模，需要两个训练目标：

1. Next Token Prediction
   $$
   \mathcal{L}_{NTP}=\sum_i\mathrm{log}p_\theta(v_i|v_1,...,v_{i-1},u_1,...,u_{i-1})
   $$

2. Mask Token Prediction

   使用`MaskGIT`实现视觉和文本token的统一训练，每一步随机将部分图像token转换为MASK，标记为$u_*$

   $$
   \mathcal{L}_{MTP}=\sum_i\mathrm{log}p_\theta(u_j|u_*,u_x,...,u_*,u_M,v_1,...,v_N)
   $$

最终的损失函数是两者的加权和

## Training Pipeline

![image-20250808014532242](assets/image-20250808014532242.png)
