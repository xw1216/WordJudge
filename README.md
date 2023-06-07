# 使用说明

## 代码功能

- 本代码模型改进自 BrainGNN 与 BrainNetCNN，主要用于分析 fMRI 图像分类。
- 使用 `Hydra` 读取本地 `./conf/*.yaml` 文件进行项目配置。
- 使用 `Wandb` 在线平台记录训练结果。
- 默认使用自建数据集，兼容Conn组水平结果。
- 如需自定义请调整 `./data` 中的数据与代码。

## 运行要求
- 需要在 `Wandb` 平台注册账号并在本地登录。
- OS软件要求
    - Python 3.10.10
    - Nvidia Cuda Toolkit 11.7.1
    - Nvidia Cudnn Library 8.5.0
- 其他 Python 包要求位于 `requirements.txt` 中，可使用 `pip` 安装。

## 运行方法
- 自定义 `./conf` 中的配置。
- 将输入数据复制入 `./data/` 中新建的 `data_type` 文件夹中。
- 训练标签使用 `./data/labels.csv` 文件。
- 运行 `_main_.py` 文件。
- 前往 `./log/sys` 或者 `Wandb` 在线平台查看运行结果。
