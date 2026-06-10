# Physics-MachineLearning-Course-Project
物理中的机器学习课程作业

- [2026.6.2 23:20]
    - 新增archive文件夹：存储每个版本的模型
    - 新增outputs文件夹：存档ipynb的输出，省去重复跑模型的费时费力

---
# [2026.6.10 14:40]模型的数据导入及处理部分说明
整个数据流水线（数据导入、清洗、切片、滑动窗口处理、数据集构建以及划分训练集/验证集）主要集中在前几个代码块（Cells）以及辅助函数 `windowed_dataset` 中。

数据处理链路可以拆解为三个核心阶段：


### 第一阶段：原始数据导入与可视化（Data Loading & Exploration）

这部分代码负责从比利时 SILSO 官方网站拉取太阳黑子 CSV 数据，并转换为内存中的 Pandas `DataFrame`。

```python
# %% 1. 从 URL 异步下载并导入数据
url = "https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv"
data = pd.read_csv(url, sep=';', header=None,
                   names=['year', 'month', 'decimal_year', 'sunspot_number', 
                          'std_dev', 'observations', 'indicator'])
data.head()

# %% 2. 提取目标单变量时间序列
# 第4列（索引为3）是具体的太阳黑子月均数值（sunspot_number），提取为纯 numpy 数组
series = data['sunspot_number'].values
time = data.index.values 

```

* **功能**：通过 `pd.read_csv` 把原始无表头的数据强行赋上对应的物理含义（年、月、浮点年、太阳黑子数、标准差等）。
* **后续的数据对齐**：在最下方的结果展示区中，还补充了对日期的显式转换：`data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))`，这也是导入后处理非常核心的一步，用于精准定位 2024–2026 年的物理区间。

### 第二阶段：训练集与验证集物理划分（Train-Test Split）

在时间序列中，我们不能破坏时间的前后因果律，所以必须采用**断后划分（按时间先后截断）**，而不是随机打乱（Random Shuffle）。

```python
# %% 划分逻辑
split_time = int(len(series)*0.9)          # 计算 90% 数据处的边界点索引
time_train = time[:split_time]             # 前 90% 的时间索引
x_train = series[:split_time]              # 前 90% 的太阳黑子历史数据（用于训练）

time_valid = time[split_time:]             # 后 10% 的时间索引
x_valid = series[split_time:]              # 后 10% 的太阳黑子历史数据（用于验证测试）

```

### 第三阶段：构建高效的滑动窗口数据流水线（High-Performance Input Pipeline）

这是整个深度学习项目中**技术含金量最高**的数据处理部分。单纯靠上面的 Numpy 数组是没法直接喂给神经网络训练的，因为循环网络（LSTM）和一维卷积（Conv1D）需要特殊的特征形态。

这一步由定义的 `windowed_dataset` 辅助函数和调用行彻底完成：

```python
# %% 调用核心函数，将 numpy 转化为 tf.data.Dataset 迭代器
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

```

我们进入 `windowed_dataset` 的内部，看看它如何利用 **`tf.data` 框架** 榨干 CPU 并将数据升维：

```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # 1. 维度扩张：将一维数组 (N,) 转换为二维矩阵 (N, 1)，为了满足 Conv1D 要求的通道维度 (in_channels=1)
    series = series[:, np.newaxis]                                
    
    # 2. 转换为张量切片：把 NumPy 转换为 TensorFlow 原生能读取的 Dataset 数据流
    ds = tf.data.Dataset.from_tensor_slices(series)               
    
    # 3. 滑动窗口切片：创建一个包含 (window_size + 1) 个元素（即 60+1=61 个月）的滑动窗口。
    # shift=1 代表每次只向右移动 1 个月，drop_remainder=True 确保最后凑不满 61 个月的数据被利落扔掉
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True) 
    
    # 4. 扁平化合并：把滑窗划分出来的独立“小窗口对象”扁平化，打包转换为真正可以参与矩阵运算的常规 Tensor Batch
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))          
    
    # 5. 时间去相关（洗牌）：用一个容量为 900 的 Shuffle Buffer 动态打乱这些滑窗的先后顺序
    # 核心目的：消除长时间序列宏观上的趋势依赖，让小样本更独立分布（I.I.D.），让模型学得更泛化
    ds = ds.shuffle(shuffle_buffer)                               
    
    # 6. 特征与标签分离（最关键映射）：
    # 把这 61 个月的数据切割。w[:-1] 是前 60 个月的历史数据（X，输入特征），w[-1] 是第 61 个月的真实黑子数（y，监督标签）
    ds = ds.map(lambda w: (w[:-1], w[-1]))                        
    
    # 7. 批处理与异步预加载：
    # 把数据打包成修改后的 `batch_size=64`，并开启 .prefetch(1) 触发 CPU-GPU 异步硬件加速
    return ds.batch(batch_size).prefetch(1)                       

```

### 💡 总结

* **`Pandas`** 负责从网络把数据“抓进来”。
* **`Numpy`** 负责把数据按 9:1 **“切两半”**。
* **`tf.data.Dataset`** 负责在训练前把数据通过滑窗“重组、洗牌、打包装车（Batching）”，直接变成发动机（CNN-BiLSTM）需要的格式。



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