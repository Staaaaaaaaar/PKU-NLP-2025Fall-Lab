# PKU-NLP-2025Fall-Lab

自然语言处理 2025 学年秋季学期期末作业

## 作业要求

作业主题为「基于 Seq2Seq 的机器翻译模型实现」，你需要在 baseline（[seq2seq-rnn.py](seq2seq-rnn.py)）基础上完成模型改进与实验报告撰写。

### 1. 核心代码实现

1. **门控循环单元（LSTM / GRU 二选一）**

- 将 baseline 中的普通 RNN 单元替换为 LSTM 或 GRU。
- **严禁直接调用** `nn.LSTMCell`, `nn.LSTM`, `nn.GRUCell`, `nn.GRU` 等现成模块。
- 必须使用 `nn.Linear()` 或 `torch.mm()` 配合激活函数（如 `torch.sigmoid`, `torch.tanh`, `torch.softmax`）从零构建门控单元内部计算逻辑。
- 代码量参考：手写 LSTM Cell 约 30 行左右（不含空行/注释）。

2. **Attention 机制（Decoder 端对 Encoder 端）**

- 在 Decoder 端实现对 Encoder 隐状态序列的 Attention。
- 需要在每个解码步基于 attention 权重计算上下文向量（context），并参与 Decoder 的状态更新与输出预测。
- 代码量参考：Attention 约 20 行左右（不含空行/注释）。

### 2. 理论分析

基于你手写的 LSTM/GRU 代码，解释其门控机制如何缓解普通 RNN 的梯度消失/爆炸问题，从而更好地捕捉长距离依赖。

### 3. 报告撰写

提交一份《研究报告》，应至少包含：

- **任务描述**：简述作业目标与数据集/评测指标
- **模型原理与实现思路**：
  - 手写 LSTM/GRU 的公式推导与实现对应关系
  - Attention 的打分、归一化（softmax）与 context 计算过程
- **思考题分析**：对应第 2 部分
- **实验结果与分析**

### 4. 扩展任务（可选）

在完成以上要求后，可尝试基于 `nn.Linear` 实现 Transformer Seq2Seq（同样不调用现成 Transformer 模块），并给出实现思路或核心代码。

## 数据集

数据选自[Tatoeba](https://www.manythings.org/anki/)数据集中的中文-英文翻译数据。

数据已经下载并处理好，位于`data`文件夹中。其中包含 26,187 条训练集数据`zh_en_train.txt`，1,000 条验证集数据`zh_en_val.txt`，以及 1,000 条测试集数据`zh_en_test.txt`。每一行是一组数据，形式为“中文句子\t 英文句子”。

## 评测指标

作业采用 BLEU Score 为评测指标。评测代码提供在示例代码 `seq2seq-rnn.py` 中，需要下载 `sacrebleu` 包。

## Baseline 模型

作业提供的 baseline 模型是基于普通 RNN 的 Seq2Seq 模型，在`seq2seq-rnn.py`中。

## 文件结构

- `seq2seq-rnn.py`：Baseline（普通 RNNCell）
- `seq2seq-lstm.py`：手写 LSTM（无 Attention）
- `seq2seq-lstm-attn.py`：手写 LSTM + Decoder 端 Attention
- `seq2seq-transformer.py`：手写 Transformer Seq2Seq
  - 不调用 `nn.Transformer` / `nn.MultiheadAttention` 等现成模块
  - 主要由 `nn.Linear()` + 基础算子手写多头注意力、LayerNorm、FFN、mask 等

## 运行方式

安装依赖：

```bash
pip install sacrebleu tqdm
```

训练/验证/测试（与示例代码一致，默认会保存 best checkpoint）：

```bash
# Baseline
python seq2seq-rnn.py

# 手写 LSTM（无 Attention）
python seq2seq-lstm.py

# 手写 LSTM + Attention
python seq2seq-lstm-attn.py

# 手写 Transformer
python seq2seq-transformer.py
```

常用参数（各脚本均支持与示例代码相同的训练参数）：

```bash
python xxx.py --num_train -1 --max_len 10 --batch_size 128 --optim adam --num_epoch 10 --lr 0.0005
```

Transformer 额外结构参数（本仓库默认值已对齐下表配置）：

```bash
python seq2seq-transformer.py --num_layers 3 --num_heads 8 --d_model 256 --d_ff 512 --dropout 0.1
```

## 实验结果

以下模型都可以在有 16GB 内存 CPU 的电脑上训练。其中 RNN 为提供 baseline 示例模型的效果。其他模型需要自己实现，结果仅供参考。

其中 Transformer 模型的设置为：`num_layer=3`, `num_head=8`, `hidden_size=256`, `ffn_hidden_size=512`, `dropout=0.1`。训练相关超参数和示例代码相同。

| Model       | #train data | BLEU  | Train Time (GPU) | Train Time (CPU) | GPU Mem  |
| ----------- | :---------: | :---: | :--------------: | :--------------: | :------: |
| RNN         |   26,187    | 1.41  |     1.5 min      |     ~ 50min      | 1,249 MB |
| RNN+Att     |   26,187    | 13.15 |     2.4 min      |    ~ 1h 10min    | 1,431 MB |
| LSTM+Att    |   26,187    | 13.52 |     3.1 min      |    ~ 1h 10min    | 1,449 MB |
| Transformer |   26,187    | 23.41 |     5.5 min      |    ~ 1h 10min    | 1,501 MB |
