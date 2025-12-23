# seq2seq_translation_example
自然语言处理23学年秋季学期作业2（基于Seq2Seq模型的机器翻译）示例代码

## 数据集
本次作业数据选自[Tatoeba](https://www.manythings.org/anki/)数据集中的中文-英文翻译数据。

数据已经下载并处理好，位于`data`文件夹中。其中包含26,187条训练集数据`zh_en_train.txt`，1,000条验证集数据`zh_en_val.txt`，以及1,000条测试集数据`zh_en_test.txt`。每一行是一组数据，形式为“中文句子\t英文句子”。

## 评测指标
本次作业采用BLEU Score为评测指标。评测代码提供在示例代码中，需要下载`sacrebleu`包。

## Baseline模型
本次作业提供的baseline模型是基于普通RNN的Seq2Seq模型，在`seq2seq-rnn.py`中。

## 实验结果

以下模型都可以在有16GB内存CPU的电脑上训练。其中RNN为提供baseline示例模型的效果。其他模型需要同学们自己实现，结果是提供参考的。
其中Transformer模型的设置为：`num_layer=3`, `num_head=8`, `hidden_size=256`, `ffn_hidden_size=512`, `dropout=0.1`。训练相关超参数和示例代码相同。
| Model |#train data |  BLEU  |  Train Time (GPU) | Train Time (CPU)| GPU Mem |
|----------|:-------------:|:-----:|:-----:|:-----:|:-----:|
| RNN         |  26,187   | 1.41 | 1.5 min | ~  50min    | 1,249 MB |
| RNN+Att     |  26,187   | 13.15 | 2.4 min | ~ 1h 10min    | 1,431 MB |
| LSTM+Att    |  26,187   | 13.52 | 3.1 min | ~ 1h 10min    | 1,449 MB |
| Transformer |  26,187   | 23.41  | 5.5 min | ~ 1h 10min | 1,501 MB  |
