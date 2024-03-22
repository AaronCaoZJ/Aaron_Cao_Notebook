| *Robomimic v0.3* |
| ---------------- |

> `Robomimic` 是一个用于机器人从演示学习的框架，该项目提供了一系列机器人操作的演示数据和离线学习算法，可以让人们对任务和算法进行标准化测试。

> *ref:*
>
> *马浩飞博客，《【仿真实验】robomimic项目复现》,`https://www.mahaofei.com/post/8cdbc8de`* 
>
> *Robomimic 官方文档，`https://robomimic.github.io/docs/introduction/overview.html`*
>
> *Robosuite 官方文档，`https://robosuite.ai/docs/overview.html`*



# 安装

## Create and activate conda environment

```bash
conda create -n robomimic_venv python=3.8.0 -y
conda activate robomimic_venv
```

## Install PyTorch

> *注意安装 GPU 版和 CPU 版的区别*

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

> *`torchversion==0.15.0`可能找不到，改为`torchversion==0.15.1`成功*

```bash
conda install pytorch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Install robomimic from source

```bash
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
# clone from https
git clone https://github.com/AaronCaoZJ/robomimic.git
# clone with ssh
git clone git@github.com:AaronCaoZJ/robomimic.git
cd robomimic
pip install -e .
```

## Install robosuite from source

```bash
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
# clone from https
git clone https://github.com/AaronCaoZJ/robosuite.git
# clone with ssh
git clone git@github.com:AaronCaoZJ/robosuite.git
cd robosuite
pip install -r requirements.txt
```



# Mujoco

## 阿里云`.mujoco`备份

> 安装及报错处理参考博客`https://blog.csdn.net/weixin_51844581/article/details/128454472`



# 测试运行

```bash
cd <PATH_TO_robomimic_INSTALL_DIRECTORY>
python examples/train_bc_rnn.py --debug
bash test.sh
```

> - [x] *运行过程可能持续几分钟，命令行会出现很多 passed，说明没有问题*
> - [ ] *EGL 是用于没有显示器的 server 版服务器进行渲染的时候使用的*



# 实验复现

## 下载数据集

```bash
cd <PATH_TO_robomimic_INSTALL_DIRECTORY>/robomimic/scripts
python download_datasets.py --tasks lift
```

## 可视化

```bash
python playback_dataset.py --dataset <path/to/.hdf5> --video_path <path/to/.mp4> --n 5
```

## 生成配置文件

```bash
cd <PATH_TO_robomimic_INSTALL_DIRECTORY>/robomimic
python scripts/generate_paper_configs.py --output_dir tmp/experiment_results
```

生成的配置文件默认在`robomimic/exps/paper`路径下

## 执行训练

``` bash
cd <PATH_TO_robomimic_INSTALL_DIRECTORY>/robomimic/scripts
python train.py --config <path/to/.json>
```

## check the training outputs

After the script finishes, you can check the training outputs in the `<train.output_dir>/<experiment.name>/<date>` experiment directory:

```
config.json               # config used for this experiment
logs/                     # experiment log files
  log.txt                 # terminal output
  tb/                     # tensorboard logs
  wandb/                  # wandb logs
videos/                   # videos of robot rollouts during training
models/                   # saved model checkpoints
```

- 使用 Tensorboard 查看训练过程

```bash
tensorboard --logdir <experiment-log-dir> --bind_all
```
