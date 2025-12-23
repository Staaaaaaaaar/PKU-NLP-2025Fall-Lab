"""Seq2Seq Machine Translation (Handwritten LSTM + Attention)

严格遵守作业约束：
- 不调用 nn.LSTM / nn.GRU / nn.LSTMCell / nn.GRUCell 等现成循环模块
- LSTM Cell 内部用 nn.Linear + 激活函数手写门控逻辑
- Decoder 端实现对 Encoder hidden states 的 Attention

运行方式与参考代码保持一致：
  python seq2seq-lstm-attn.py --num_epoch 10 --max_len 10 --batch_size 128

注意：本文件会尝试导入 sacrebleu 与 tqdm；如果环境缺包，请先安装：
  pip install sacrebleu tqdm
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    from sacrebleu.metrics import BLEU
except Exception as e:  # pragma: no cover
    raise ImportError("缺少依赖 sacrebleu：请先 pip install sacrebleu") from e

try:
    from tqdm import tqdm
except Exception as e:  # pragma: no cover
    raise ImportError("缺少依赖 tqdm：请先 pip install tqdm") from e


# =========================
# Data
# =========================

def load_data(num_train):
    zh_sents = {}
    en_sents = {}
    for split in ["train", "val", "test"]:
        zh_sents[split] = []
        en_sents[split] = []
        with open(f"data/zh_en_{split}.txt", encoding="utf-8") as f:
            for line in f.readlines():
                zh, en = line.strip().split("\t")
                zh = zh.split()
                en = en.split()
                zh_sents[split].append(zh)
                en_sents[split].append(en)
    num_train = len(zh_sents["train"]) if num_train == -1 else num_train
    zh_sents["train"] = zh_sents["train"][:num_train]
    en_sents["train"] = en_sents["train"][:num_train]
    print("训练集 验证集 测试集大小分别为", len(zh_sents["train"]), len(zh_sents["val"]), len(zh_sents["test"]))
    return zh_sents, en_sents


class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = []
        self.add_word("[BOS]")
        self.add_word("[EOS]")
        self.add_word("[UNK]")
        self.add_word("[PAD]")

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2cnt[word] = 0
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        self.word2cnt[word] += 1

    def add_sent(self, sent):
        for word in sent:
            self.add_word(word)

    def index(self, word):
        return self.word2idx.get(word, self.word2idx["[UNK]"])

    def encode(self, sent, max_len):
        encoded = [self.word2idx["[BOS]"]] + [self.index(word) for word in sent][:max_len] + [self.word2idx["[EOS]"]]
        return encoded

    def decode(self, encoded, strip_bos_eos_pad=False):
        return [
            self.idx2word[_]
            for _ in encoded
            if not strip_bos_eos_pad or self.idx2word[_] not in ["[BOS]", "[EOS]", "[PAD]"]
        ]

    def __len__(self):
        return len(self.idx2word)


def collate(data_list):
    src = torch.stack([torch.LongTensor(_[0]) for _ in data_list])
    tgt = torch.stack([torch.LongTensor(_[1]) for _ in data_list])
    return src, tgt


def padding(inp_ids, max_len, pad_id):
    max_len += 2  # include [BOS] and [EOS]
    ids_ = np.ones(max_len, dtype=np.int32) * pad_id
    max_len = min(len(inp_ids), max_len)
    ids_[:max_len] = inp_ids
    return ids_


def create_dataloader(zh_sents, en_sents, max_len, batch_size, pad_id):
    dataloaders = {}
    for split in ["train", "val", "test"]:
        shuffle = True if split == "train" else False
        datas = [
            (padding(zh_vocab.encode(zh, max_len), max_len, pad_id), padding(en_vocab.encode(en, max_len), max_len, pad_id))
            for zh, en in zip(zh_sents[split], en_sents[split])
        ]
        dataloaders[split] = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloaders["train"], dataloaders["val"], dataloaders["test"]


# =========================
# Core: Handwritten LSTM Cell
# =========================


class ManualLSTMCell(nn.Module):
    """手写 LSTM Cell（单层、单时间步）。

    数学定义（批量形式）：
    - x_t \in R^{N\times D}
    - h_{t-1}, c_{t-1} \in R^{N\times H}

    将四个门拼在一起：
    - z_t = W_x x_t + W_h h_{t-1} + b,  z_t \in R^{N \times 4H}

    切分后：
    - [z_i, z_f, z_g, z_o] = split(z_t)
    - i_t = \sigma(z_i)                      (输入门 input gate)
    - f_t = \sigma(z_f)                      (遗忘门 forget gate)
    - g_t = tanh(z_g)                        (候选记忆 candidate)
    - o_t = \sigma(z_o)                      (输出门 output gate)

    状态更新：
    - c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    - h_t = o_t ⊙ tanh(c_t)

    关键点：c_t 的“线性通道”让梯度可以沿时间更稳定地传播。
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # W_x: R^D -> R^{4H}（包含 bias = b）
        self.x2gates = nn.Linear(input_size, 4 * hidden_size, bias=True)
        # W_h: R^H -> R^{4H}
        self.h2gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x_t: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        h_prev, c_prev = state

        # z_t = W_x x_t + W_h h_{t-1} + b
        z_t = self.x2gates(x_t) + self.h2gates(h_prev)

        # [z_i, z_f, z_g, z_o]，每个 shape = (N, H)
        z_i, z_f, z_g, z_o = z_t.chunk(4, dim=-1)

        # i_t = sigmoid(z_i)
        i_t = torch.sigmoid(z_i)
        # f_t = sigmoid(z_f)
        f_t = torch.sigmoid(z_f)
        # g_t = tanh(z_g)
        g_t = torch.tanh(z_g)
        # o_t = sigmoid(z_o)
        o_t = torch.sigmoid(z_o)

        # c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        c_t = f_t * c_prev + i_t * g_t
        # h_t = o_t ⊙ tanh(c_t)
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


# =========================
# Core: Decoder Attention (Additive/Bahdanau)
# =========================


class AdditiveAttention(nn.Module):
    """Bahdanau(Additive) Attention（Decoder 对 Encoder）。

    给定：
    - query q_t = h^{dec}_{t-1} \in R^{N\times H}
    - keys  k_s = h^{enc}_s     \in R^{N\times L\times H}
    - values v_s = h^{enc}_s    \in R^{N\times L\times H}

    打分函数：
    - e_{t,s} = v^T tanh(W_q q_t + W_k k_s)   (标量)

    归一化：
    - α_t = softmax(e_t)                      (对 s 维度)

    上下文向量：
    - context_t = Σ_s α_{t,s} v_s             \in R^{N\times H}

    mask：对 PAD 位置令 e_{t,s} = -inf，避免被 softmax 分配概率。
    """

    def __init__(self, hidden_size: int, attn_size: Optional[int] = None):
        super().__init__()
        attn_size = hidden_size if attn_size is None else attn_size

        # W_q: R^H -> R^A
        self.W_q = nn.Linear(hidden_size, attn_size, bias=False)
        # W_k: R^H -> R^A
        self.W_k = nn.Linear(hidden_size, attn_size, bias=False)
        # v^T: R^A -> R^1
        self.v = nn.Linear(attn_size, 1, bias=False)

    def forward(self, query: Tensor, keys: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # query: (N, H)
        # keys : (N, L, H)
        # mask : (N, L)  True 表示有效（非 PAD）

        # W_q q_t: (N, A)
        q = self.W_q(query)
        # W_k k_s: (N, L, A)
        k = self.W_k(keys)

        # energy_{t,s} = tanh(W_q q_t + W_k k_s): (N, L, A)
        energy = torch.tanh(k + q.unsqueeze(1))
        # e_{t,s} = v^T energy: (N, L)
        scores = self.v(energy).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # α_t = softmax(scores): (N, L)
        attn_weights = torch.softmax(scores, dim=-1)
        # context_t = Σ_s α_{t,s} v_s: (N, H)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)

        return context, attn_weights


# =========================
# Seq2Seq (LSTM + Attention)
# =========================


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.cell = ManualLSTMCell(embedding_dim, hidden_size)

    def forward(self, x_t_ids: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # x_t = Emb(x_t_ids)
        x_t = self.embed(x_t_ids)
        # (h_t, c_t) = LSTMCell(x_t, (h_{t-1}, c_{t-1}))
        return self.cell(x_t, state)


class DecoderLSTMWithAttn(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.attn = AdditiveAttention(hidden_size)
        # LSTM 输入拼接 [embedding; context]
        self.cell = ManualLSTMCell(embedding_dim + hidden_size, hidden_size)
        # 输出层用 [h_t; context_t]
        self.h2o = nn.Linear(2 * hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        y_t_ids: Tensor,
        state: Tuple[Tensor, Tensor],
        encoder_hiddens: Tensor,
        src_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h_prev, c_prev = state

        # y_t = Emb(y_t_ids)
        y_t = self.embed(y_t_ids)

        # context_t = Attention(h_{t-1}^{dec}, H^{enc})
        context, _ = self.attn(h_prev, encoder_hiddens, mask=src_mask)

        # x_t' = [y_t; context_t]
        x_t_prime = torch.cat([y_t, context], dim=-1)

        # (h_t, c_t) = LSTMCell(x_t', (h_{t-1}, c_{t-1}))
        h_t, c_t = self.cell(x_t_prime, (h_prev, c_prev))

        # logits = W_o [h_t; context_t]
        logits = self.h2o(torch.cat([h_t, context], dim=-1))
        log_probs = self.log_softmax(logits)

        return log_probs, (h_t, c_t)


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab: Vocab, tgt_vocab: Vocab, embedding_dim: int, hidden_size: int, max_len: int):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.hidden_size = hidden_size
        self.encoder = EncoderLSTM(len(src_vocab), embedding_dim, hidden_size)
        self.decoder = DecoderLSTMWithAttn(len(tgt_vocab), embedding_dim, hidden_size)
        self.max_len = max_len

    def init_state(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.hidden_size, device=device)
        return h0, c0

    def init_tgt_bos(self, batch_size: int) -> Tensor:
        device = next(self.parameters()).device
        return (torch.ones(batch_size, device=device) * self.tgt_vocab.index("[BOS]")).long()

    def forward_encoder(self, src: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tensor]:
        # src: (N, Ls)
        Bs, Ls = src.size()
        state = self.init_state(Bs)

        pad_id = self.src_vocab.word2idx["[PAD]"]
        src_mask = src != pad_id  # True 为有效

        encoder_hiddens = []
        for i in range(Ls):
            x_t_ids = src[:, i]
            h_t, c_t = self.encoder(x_t_ids, state)
            state = (h_t, c_t)
            encoder_hiddens.append(h_t)

        # (N, Ls, H)
        encoder_hiddens = torch.stack(encoder_hiddens, dim=1)
        return state, encoder_hiddens, src_mask

    def forward_decoder(self, tgt: Tensor, state: Tuple[Tensor, Tensor], encoder_hiddens: Tensor, src_mask: Tensor) -> Tensor:
        # tgt: (N, Lt)
        Bs, Lt = tgt.size()
        outputs = []
        for i in range(Lt):
            y_t_ids = tgt[:, i]  # teacher forcing
            logp, state = self.decoder(y_t_ids, state, encoder_hiddens, src_mask)
            outputs.append(logp)
        return torch.stack(outputs, dim=1)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        state, encoder_hiddens, src_mask = self.forward_encoder(src)
        outputs = self.forward_decoder(tgt, state, encoder_hiddens, src_mask)
        return outputs

    def predict(self, src: Tensor) -> Tensor:
        state, encoder_hiddens, src_mask = self.forward_encoder(src)
        y = self.init_tgt_bos(batch_size=src.shape[0])

        preds = [y]
        eos_id = self.tgt_vocab.index("[EOS]")

        while len(preds) < self.max_len:
            logp, state = self.decoder(y, state, encoder_hiddens, src_mask)
            y = logp.argmax(-1)
            preds.append(y)
            if torch.all(y == eos_id):
                break

        return torch.stack(preds, dim=-1)


# =========================
# Train / Eval
# =========================


def train_loop(model, optimizer, criterion, loader, device):
    model.train()
    epoch_loss = 0.0
    for src, tgt in tqdm(loader):
        src = src.to(device)
        tgt = tgt.to(device)
        outputs = model(src, tgt)
        loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.shape[-1]), tgt[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    return epoch_loss


def test_loop(model, loader, tgt_vocab, device):
    model.eval()
    bleu = BLEU(force=True)
    hypotheses, references = [], []
    for src, tgt in tqdm(loader):
        B = len(src)
        for _ in range(B):
            _src = src[_].unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model.predict(_src)

            ref = " ".join(tgt_vocab.decode(tgt[_].tolist(), strip_bos_eos_pad=True))
            hypo = " ".join(tgt_vocab.decode(outputs[0].cpu().tolist(), strip_bos_eos_pad=True))
            references.append(ref)
            hypotheses.append(hypo)

    score = bleu.corpus_score(hypotheses, [references]).score
    return hypotheses, references, score


# =========================
# Main
# =========================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", default=-1, help="训练集大小，等于-1时将包含全部训练数据")
    parser.add_argument("--max_len", default=10, help="句子最大长度")
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--optim", default="adam")
    parser.add_argument("--num_epoch", default=10)
    parser.add_argument("--lr", default=0.0005)
    args = parser.parse_args()

    zh_sents, en_sents = load_data(args.num_train)

    zh_vocab = Vocab()
    en_vocab = Vocab()
    for zh, en in zip(zh_sents["train"], en_sents["train"]):
        zh_vocab.add_sent(zh)
        en_vocab.add_sent(en)
    print("中文词表大小为", len(zh_vocab))
    print("英语词表大小为", len(en_vocab))

    trainloader, validloader, testloader = create_dataloader(
        zh_sents,
        en_sents,
        args.max_len,
        args.batch_size,
        pad_id=zh_vocab.word2idx["[PAD]"],
    )

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2Seq(zh_vocab, en_vocab, embedding_dim=256, hidden_size=256, max_len=int(args.max_len))
    model.to(device)

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr))
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    else:
        raise ValueError("optim must be sgd or adam")

    weights = torch.ones(len(en_vocab)).to(device)
    weights[en_vocab.word2idx["[PAD]"]] = 0
    criterion = nn.NLLLoss(weight=weights)

    start_time = time.time()
    best_score = 0.0

    for epoch in range(int(args.num_epoch)):
        loss = train_loop(model, optimizer, criterion, trainloader, device)
        hypotheses, references, bleu_score = test_loop(model, validloader, en_vocab, device)

        if bleu_score > best_score:
            torch.save(model.state_dict(), "model_best_lstm_attn.pt")
            best_score = bleu_score

        print(f"Epoch {epoch}: loss = {loss}, valid bleu = {bleu_score}")
        print(references[0])
        print(hypotheses[0])

    end_time = time.time()

    model.load_state_dict(torch.load("model_best_lstm_attn.pt", map_location=device))
    hypotheses, references, bleu_score = test_loop(model, testloader, en_vocab, device)
    print(f"Test bleu = {bleu_score}")
    print(references[0])
    print(hypotheses[0])
    print(f"Training time: {round((end_time - start_time) / 60, 2)}min")
