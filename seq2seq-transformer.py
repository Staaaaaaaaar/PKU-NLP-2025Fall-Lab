from __future__ import annotations

import argparse
import math
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
# Utilities: masks
# =========================


def make_pad_mask(ids: Tensor, pad_id: int) -> Tensor:
    """ids: (N, L) -> mask: (N, L) with True for non-pad."""
    return ids != pad_id


def make_causal_mask(L: int, device) -> Tensor:
    """causal mask for self-attn (L, L): True means allowed."""
    # allow attending to <= current index
    return torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))


# =========================
# Core: Handwritten LayerNorm
# =========================


class ManualLayerNorm(nn.Module):
    """手写 LayerNorm。

    对最后一维做归一化：
    - μ = mean(x, dim=-1)
    - σ^2 = var(x, dim=-1)
    - y = (x - μ) / sqrt(σ^2 + eps)
    - out = γ ⊙ y + β
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return y * self.gamma + self.beta


# =========================
# Core: Positional Encoding (sinusoidal)
# =========================


class SinusoidalPositionalEncoding(nn.Module):
    """PE(pos,2i)=sin(pos/10000^{2i/d}); PE(pos,2i+1)=cos(...)

    x: (N, L, d_model) -> x + pe[:, :L]
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        L = x.size(1)
        return x + self.pe[:, :L, :]


# =========================
# Core: Multi-Head Attention (handwritten)
# =========================


class MultiHeadAttention(nn.Module):
    """手写 Multi-Head Attention（不使用 nn.MultiheadAttention）。

    输入：
    - Q: (N, Lq, d)
    - K: (N, Lk, d)
    - V: (N, Lv, d)  (通常 Lv==Lk)

    线性投影：
    - Q' = Q W_Q, K' = K W_K, V' = V W_V

    分头：
    - reshape -> (N, h, L, d_k), 其中 d_k = d/h

    Scaled Dot-Product Attention：
    - scores = (Q' K'^T) / sqrt(d_k)  (对每个 head)
    - scores += mask(非法位置置为 -inf)
    - A = softmax(scores)
    - head = A V'

    合并：
    - concat(heads) W_O
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        # (N, L, d) -> (N, h, L, d_k)
        N, L, _ = x.shape
        x = x.view(N, L, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # (N, h, L, d_k) -> (N, L, d)
        N, h, L, d_k = x.shape
        x = x.transpose(1, 2).contiguous().view(N, L, h * d_k)
        return x

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Q,K,V: (N, L, d)
        q = self._split_heads(self.W_q(Q))
        k = self._split_heads(self.W_k(K))
        v = self._split_heads(self.W_v(V))

        # scores: (N, h, Lq, Lk)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # attn_mask: (Lq, Lk) with True allowed
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # key_padding_mask: (N, Lk) with True for valid
        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        A = torch.softmax(scores, dim=-1)
        out = torch.matmul(A, v)  # (N, h, Lq, d_k)

        out = self._merge_heads(out)
        out = self.W_o(out)
        return out


# =========================
# Core: Position-wise FFN
# =========================


class PositionwiseFFN(nn.Module):
    """FFN(x)=max(0, xW1+b1)W2+b2"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


# =========================
# Encoder/Decoder blocks
# =========================


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.ln1 = ManualLayerNorm(d_model)
        self.ln2 = ManualLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        # Pre-LN Transformer (更稳)：
        # x = x + MHA(LN(x))
        x = x + self.dropout(
            self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=src_key_padding_mask)
        )
        # x = x + FFN(LN(x))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.ln1 = ManualLayerNorm(d_model)
        self.ln2 = ManualLayerNorm(d_model)
        self.ln3 = ManualLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_key_padding_mask: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        # masked self-attn
        x = x + self.dropout(
            self.self_attn(
                self.ln1(x),
                self.ln1(x),
                self.ln1(x),
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )
        )
        # encoder-decoder attention
        x = x + self.dropout(self.cross_attn(self.ln2(x), memory, memory, key_padding_mask=src_key_padding_mask))
        # ffn
        x = x + self.dropout(self.ffn(self.ln3(x)))
        return x


# =========================
# Transformer Seq2Seq
# =========================


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

        self.src_embed = nn.Embedding(len(src_vocab), d_model)
        self.tgt_embed = nn.Embedding(len(tgt_vocab), d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len + 5)
        self.dropout = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout=dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout=dropout) for _ in range(num_layers)])

        self.out_proj = nn.Linear(d_model, len(tgt_vocab))
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def encode(self, src_ids: Tensor) -> Tuple[Tensor, Tensor]:
        # src_ids: (N, Ls)
        pad_id = self.src_vocab.word2idx["[PAD]"]
        src_key_padding_mask = make_pad_mask(src_ids, pad_id)  # (N, Ls) True valid

        x = self.src_embed(src_ids)  # (N, Ls, d)
        x = self.dropout(self.pos(x))

        for layer in self.enc_layers:
            x = layer(x, src_key_padding_mask)

        return x, src_key_padding_mask

    def decode(self, tgt_ids: Tensor, memory: Tensor, tgt_key_padding_mask: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        # tgt_ids: (N, Lt)
        N, Lt = tgt_ids.shape
        device = tgt_ids.device
        tgt_mask = make_causal_mask(Lt, device=device)  # (Lt, Lt)

        x = self.tgt_embed(tgt_ids)
        x = self.dropout(self.pos(x))

        for layer in self.dec_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, src_key_padding_mask=src_key_padding_mask)

        logits = self.out_proj(x)  # (N, Lt, V)
        return self.log_softmax(logits)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        # tgt includes BOS...EOS/PAD. We predict next token distribution at each position.
        memory, src_kpm = self.encode(src)

        pad_id = self.tgt_vocab.word2idx["[PAD]"]
        tgt_kpm = make_pad_mask(tgt, pad_id)

        return self.decode(tgt, memory, tgt_key_padding_mask=tgt_kpm, src_key_padding_mask=src_kpm)

    def init_tgt_bos(self, batch_size: int, device) -> Tensor:
        return (torch.ones(batch_size, device=device) * self.tgt_vocab.index("[BOS]")).long()

    def predict(self, src: Tensor) -> Tensor:
        # greedy decoding
        device = src.device
        memory, src_kpm = self.encode(src)

        B = src.size(0)
        y = self.init_tgt_bos(B, device=device)  # (B,)
        preds = [y]

        eos_id = self.tgt_vocab.index("[EOS]")
        pad_id = self.tgt_vocab.word2idx["[PAD]"]

        # keep a growing tgt sequence (B, t)
        while len(preds) < self.max_len:
            tgt_ids = torch.stack(preds, dim=1)
            tgt_kpm = make_pad_mask(tgt_ids, pad_id)

            logp = self.decode(tgt_ids, memory, tgt_key_padding_mask=tgt_kpm, src_key_padding_mask=src_kpm)
            next_token = logp[:, -1, :].argmax(-1)
            preds.append(next_token)

            if torch.all(next_token == eos_id):
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
        for i in range(B):
            _src = src[i].unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model.predict(_src)

            ref = " ".join(tgt_vocab.decode(tgt[i].tolist(), strip_bos_eos_pad=True))
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
    parser.add_argument("--d_model", default=256)
    parser.add_argument("--num_heads", default=8)
    parser.add_argument("--num_layers", default=3)
    parser.add_argument("--d_ff", default=512)
    parser.add_argument("--dropout", default=0.1)
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
        int(args.max_len),
        int(args.batch_size),
        pad_id=zh_vocab.word2idx["[PAD]"],
    )

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerSeq2Seq(
        zh_vocab,
        en_vocab,
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        d_ff=int(args.d_ff),
        dropout=float(args.dropout),
        max_len=int(args.max_len),
    ).to(device)

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
            torch.save(model.state_dict(), "model_best_transformer.pt")
            best_score = bleu_score

        print(f"Epoch {epoch}: loss = {loss}, valid bleu = {bleu_score}")
        print(references[0])
        print(hypotheses[0])

    end_time = time.time()

    model.load_state_dict(torch.load("model_best_transformer.pt", map_location=device))
    hypotheses, references, bleu_score = test_loop(model, testloader, en_vocab, device)
    print(f"Test bleu = {bleu_score}")
    print(references[0])
    print(hypotheses[0])
    print(f"Training time: {round((end_time - start_time) / 60, 2)}min")
