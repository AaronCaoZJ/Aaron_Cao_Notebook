# Object Hallucination in VLMs

物体幻觉指模型生成语义连贯但与图像中真实物体不一致的文本内容，其源于：

* 训练数据中固有的统计偏差
* 单峰先验（语言先验）的过度依赖，VLM地解码过程由LLM主导，导致模型决策时语义偏见的权重更大

且视觉输入的uncertainty+，以上导致的幻觉+

traditional方法包括：

* 细粒度对比学习
* Region of Interest特征融合
* 数据增强手段

recent研究方向包括：

* 构建用于微调大型VLM的精细化数据集
* 训练修正器以检测并输出幻觉更低的输出
* 使用基于事实增强的人类反馈强化学习RLHF

# Visual Contrastive Decoding

- 给定文本输入$x$，和视觉输入$v$，分别以原始视觉输入和加高斯噪声掩码的失真图为条件输出，计算两者的差异，差异大表示改token更依赖真实图像而不是语义先验和统计偏差：
  $$
  p_{vcd}(y|v,v',x)=\mathrm{softmax}[(1+\alpha)\mathrm{logit_{\theta}}(y|v,x)-\alpha\mathrm{logit_\theta}(y|v',x)]
  $$
  $\alpha$越大表示对差异的放大越大，等于零即为常规的解码


* 简单地对失真输出惩罚可能影响正确部分的性能，引入自适应合理性约束：

$$
\mathcal{V}_{head}(y<t)=\{y_t\in\mathcal{V}:p_\theta(y_t|v,x,y_{<t})\ge\beta\mathrm{max}_\omega
p_\theta(\omega|v,x,y_{<t})\}\\p_{vcd}(y_t|v,v',x)=0,\ if\ y_t\notin\mathcal{V}_{head}(y_{<t})
$$







