# Physics-MachineLearning-Course-Project
物理中的机器学习课程作业

- [2026.6.2 23:20]
    - 新增archive文件夹：存储每个版本的模型
    - 新增outputs文件夹：存档ipynb的输出，省去重复跑模型的费时费力

---
# [2026.6.3 13:00]添加tensorflow面板
### 如何打开面板？
运行了训练代码之后，代码会在本地项目目录下创建一个 logs/ 文件夹。

本脚本使用的是Jupyter Notebook / VS Code Notebook，以下这个单元格能让精美的**可交互**面板在脚本里面弹出来。可以直接在CNN-BiLSTM-addTesorflowBoard.ipynb文件中找到以下代码块，它的输出就是tensorflow交互面板：

```Bash
%load_ext tensorboard
%tensorboard --logdir logs/fit
```
### 运行完脚本后再次打开TensorflowBoard/纯 Python 脚本（.py）/终端里运行，可以在系统的命令行/终端里输入：

```Bash
tensorboard --logdir=logs/fit
```
例如：
```bash
tensorboard --logdir=archive\CNN-BiLSTM-addTesorflowBoard\logs\fit
```
运行后终端会给你一个网址（通常是 http://localhost:6006/ ），用浏览器打开它，就能看到实时的 Loss、MAE 变化曲线和双向 LSTM 的网络拓扑图了！


---

# [2026.6.2 23:20]提交CNN-BiLSTM模型

### 核心改进：从 CNN-LSTM 升级为 CNN-BiLSTM
为了进一步捕捉太阳黑子时间序列数据中的前向与后向长短期依赖关系（11年/132个月的周期特征），本次提交对网络架构进行了核心升级，将原有的单向 LSTM 层替换为**双向循环神经网络 (Bidirectional LSTM)**。

### 🚨 遇到的挑战：显存溢出 (OOM) 与“瘦身”优化
在首次引入双向结构时，由于网络参数量成倍增加，直接导致了**显存爆掉 (Out of Memory)** 的问题。

为了在有限的硬件资源下顺利运行模型，本次提交对模型参数及数据批次进行了 **1/2 的剪裁**：
1. **Batch Size 减半**：由原本的 `145` 缩减为 `64`。
2. **CNN 卷积核瘦身**：`Conv1D` 的 `filters` 从 `132` 缩减为 `64`。
3. **BiLSTM 隐藏单元缩减**：
   - 第一层 LSTM 神经元由 `256` 瘦身至 `128`。
   - 第二层 LSTM 神经元由 `132` 瘦身至 `64`。

### 📊 输出结果与对比分析 (Results & Evaluation)

在相同的测试时间区间（**2024-01 至 2026-04**）内，对 CNN-LSTM 模型与改进后的 CNN-BiLSTM 模型在测试集上的标量预测指标（MAE 与 MSE）进行了量化对比：

| 模型架构 | 参数规模 (主要层) | Batch Size | 训练轮数 | MAE (测试集) | MSE (测试集) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN-LSTM** | Conv(132) + LSTM(256+132) | 145 | 200 | ~18.2787 | ~595.4198 |
| **CNN-BiLSTM** | Conv(64) + BiLSTM(128+64) | 64 | 200 | ~18.7629 | ~659.7210 |

### 🔍 结果与架构差异分析

1. **时序双向依赖特征的引入**：
   BiLSTM 结构通过正向和反向两个时序流同时提取特征，有助于模型对时序趋势的拐点（如 2024-2026 年期间的波峰起伏）进行更平衡的拟合。这是测试集指标产生差异的主要结构原因。

2. **网络轻量化的设计尝试**：
   由于硬件显存限制，CNN-BiLSTM 方案大幅压缩了通道数和隐藏单元（整体参数体量接近减半），并将 Batch Size 调整为 64。实验表明，在降低参数体量的前提下，通过变换时序层的网络拓扑结构（单向LSTM调整为双向BiLSTM），模型依然能够保持较好的收敛效率，并在当前测试区间内取得了相对稳定的拟合输出。

---