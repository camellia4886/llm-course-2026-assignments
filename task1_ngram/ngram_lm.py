# -*- coding: utf-8 -*-
"""自研 n-gram 语言模型：训练、困惑度评测与文本生成。

实现两种平滑：
  - Laplace（加一平滑）：最简单基线，方差大
  - 插值修正 Kneser-Ney（Chen & Goodman 1998 经典形式）：
      P_kn(w|u,v) = max(c(u,v,w)-D,0)/c(u,v) + gamma(u,v) * P_kn(w|v)
      P_kn(w|v)   = max(N1+(.,v,w)-D,0)/N1+(.,v,.) + gamma(v) * P_kn(w)
      P_kn(w)     = N1+(.,w) / N1+(.,.)
    其中 N1+ 是"续接多样性"计数（不同上下文的种类数），
    中阶用 trigram 的续接计数，最底层 P_kn(w) 用 bigram 的续接计数；
    D 取修正 KN 折扣 D = n1/(n1+2n2)。

计数用 numpy 排序 + np.unique 实现（类 SRILM 的 sort-count 思路），
n-gram 键打包成 int64，避免 Python 字典 tuple 键的内存开销。
"""
import time
from collections import Counter
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


# ---------------------------------------------------------------- 计数工具
def load_tokens(files, min_freq=1):
    """读取已清理语料，返回 int 化词序列与词表。0/1 号保留给 <s>/</s>。"""
    sentences = []
    for f in files:
        with open(f, encoding="utf-8") as fin:
            for line in fin:
                words = line.split()
                if words:
                    sentences.append(words)
    vocab = Counter(w for s in sentences for w in s)
    if min_freq > 1:
        vocab = {w: c for w, c in vocab.items() if c >= min_freq}
    word2id = {BOS: 0, EOS: 1}
    for i, w in enumerate(sorted(vocab), start=2):
        word2id[w] = i
    unk_id = word2id.get(UNK)
    if unk_id is None:
        unk_id = len(word2id)
        word2id[UNK] = unk_id
    seqs = [[word2id.get(w, unk_id) for w in s] for s in sentences]
    return seqs, word2id


def ngram_arrays(seqs, n):
    """把所有句子的 n-gram 拼成 int32 数组 (M, n)。句首句尾加 <s>/</s>。"""
    parts = []
    for s in seqs:
        if n == 1:
            parts.append(np.asarray(s, dtype=np.int32)[:, None])
        else:
            seq = [0] * (n - 1) + s + [1]
            if len(seq) < n:
                continue
            parts.append(np.lib.stride_tricks.sliding_window_view(
                np.asarray(seq, dtype=np.int32), n))
    return np.concatenate(parts, axis=0)


def count_ngrams(seqs, n, V, min_freq=1):
    """返回 {packed_key: count}。min_freq>1 时丢弃低频（内存控制/剪枝）。"""
    arr = ngram_arrays(seqs, n)
    if n == 1:
        uniq, cnt = np.unique(arr[:, 0], return_counts=True)
        return {int(u): int(c) for u, c in zip(uniq, cnt) if c >= min_freq}
    packed = np.zeros(len(arr), dtype=np.int64)
    mult = 1
    for col in range(n - 1, -1, -1):
        packed += arr[:, col].astype(np.int64) * mult
        mult *= V
    uniq, cnt = np.unique(packed, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, cnt) if c >= min_freq}


def pack_ids(ids, V):
    k = 0
    for i in ids:
        k = k * V + int(i)
    return k


# ---------------------------------------------------------------- 模型
class BaseLM:
    def __init__(self, n, V, word2id):
        self.n = n
        self.V = V
        self.word2id = word2id
        self.id2word = {i: w for w, i in word2id.items()}

    def prob(self, history, w):
        raise NotImplementedError

    def next_probs(self, history):
        raise NotImplementedError

    def candidate_words(self, history):
        """生成时只在这些候选词上算概率，避免对全词表循环。"""
        raise NotImplementedError

    def next_probs_fast(self, history):
        cands = list(dict.fromkeys(self.candidate_words(history)))
        if not cands:
            cands = list(range(self.V))
        p = np.zeros(self.V)
        vals = np.array([self.prob(history, w) for w in cands], dtype=np.float64)
        s = vals.sum()
        if s > 0:
            vals /= s
        p[cands] = vals
        return p


class LaplaceLM(BaseLM):
    """加一平滑：P(w|h) = (c(h,w)+1) / (c(h)+V)。"""

    def __init__(self, n, V, word2id, counts, **kw):
        super().__init__(n, V, word2id)
        self.counts = counts
        self.total = sum(counts.values())
        self.ctx_counts = Counter()
        self.suffix = {}
        if n > 1:
            uni = Counter()
            for key, c in counts.items():
                ctx = key // V
                self.ctx_counts[ctx] += c
                self.suffix.setdefault(ctx, []).append(key % V)
                uni[key % V] += c
        else:
            uni = counts
        self.top_words = sorted(uni, key=uni.get, reverse=True)[:5000]

    def candidate_words(self, history):
        if self.n == 1:
            return self.top_words
        ctx = pack_ids(history[-(self.n - 1):], self.V)
        return self.suffix.get(ctx, []) + self.top_words

    def prob(self, history, w):
        if self.n == 1:
            return (self.counts.get(w, 0) + 1.0) / (self.total + self.V)
        ctx = pack_ids(history[-(self.n - 1):], self.V)
        key = ctx * self.V + w
        return (self.counts.get(key, 0) + 1.0) / (self.ctx_counts.get(ctx, 0) + self.V)

    def next_probs(self, history):
        return self.next_probs_fast(history)


class KneserNeyLM(BaseLM):
    """插值修正 Kneser-Ney。

    n=2：最高阶用 bigram 计数，底层 P_kn(w) 用 bigram 续接计数。
    n=3：最高阶用 trigram 计数，中阶 P_kn(w|v) 用 trigram 续接计数，
         底层 P_kn(w) 用 bigram 续接计数（需要传入 bigram_counts）。
    """

    def __init__(self, n, V, word2id, counts, bigram_counts=None, **kw):
        super().__init__(n, V, word2id)
        t0 = time.time()
        self.counts = counts
        freqs = Counter(counts.values())
        n1, n2 = freqs.get(1, 0), freqs.get(2, 0)
        self.D = n1 / (n1 + 2.0 * n2) if n1 else 0.5
        self.D = min(max(self.D, 0.1), 0.9)

        if n == 1:
            tot = sum(counts.values())
            self.uni = {w: c / tot for w, c in counts.items()}
        elif n == 2:
            self.ctx_counts = Counter()          # c(v)
            n1_before = Counter()                # N1+(.,w)
            self.n1_after = Counter()            # N1+(v,.)
            self.suffix = {}
            for key, c in counts.items():
                v, w = key // V, key % V
                self.ctx_counts[v] += c
                n1_before[w] += 1
                self.n1_after[v] += 1
                self.suffix.setdefault(v, []).append(w)
            tot = sum(n1_before.values())
            self.uni_cont = {w: n1_before[w] / tot for w in n1_before}
        else:  # n == 3
            self.ctx_counts = Counter()          # c(u,v)
            self.n1_ctx_after = Counter()        # N1+(u,v,.)
            self.n1_mid = Counter()              # N1+(.,v,w)，键 (v,w) 打包
            self.suffix = {}
            for key, c in counts.items():
                ctx = key // V
                self.ctx_counts[ctx] += c
                self.n1_ctx_after[ctx] += 1
                self.n1_mid[key % (V * V)] += 1
                self.suffix.setdefault(ctx, []).append(key % V)
            self.n1_mid_sum = Counter()          # N1+(.,v,.) = Σ_w N1+(.,v,w)
            self.n1_mid_after = Counter()        # N1+(v,.) = |{w: N1+(.,v,w)>0}|
            for pw, c in self.n1_mid.items():
                v = pw // V
                self.n1_mid_sum[v] += c
                self.n1_mid_after[v] += 1
            # 底层 P_kn(w) 用 bigram 续接计数；缺省时用 trigram 中阶近似
            if bigram_counts is not None:
                n1_before = Counter()
                for key in bigram_counts:
                    n1_before[key % V] += 1
            else:
                n1_before = Counter()
                for pw in self.n1_mid:
                    n1_before[pw % V] += 1
            tot = sum(n1_before.values())
            self.uni_cont = {w: n1_before[w] / tot for w in n1_before}
        self.top_words = sorted(self._uni_ref(), key=self._uni_ref().get,
                                reverse=True)[:2000]
        self.train_time = time.time() - t0

    def _uni_ref(self):
        if self.n == 1:
            return self.uni
        return self.uni_cont

    def candidate_words(self, history):
        if self.n == 1:
            return self.top_words
        ctx = pack_ids(history[-(self.n - 1):], self.V)
        if self.n == 2:
            return self.suffix.get(ctx, []) + self.top_words
        return self.suffix.get(ctx, []) + self.top_words

    def _p_uni(self, w):
        if self.n == 1:
            return self.uni.get(w, 0.0)
        return self.uni_cont.get(w, 0.0)

    def _p_bi(self, v, w):
        """中阶/二阶分布。n=2 用 bigram 计数；n=3 用 trigram 续接计数。"""
        if self.n == 2:
            c_b = self.counts.get(v * self.V + w, 0)
            c_v = self.ctx_counts.get(v, 0)
            if c_v == 0:
                return self._p_uni(w)
            disc = max(c_b - self.D, 0.0) / c_v
            gamma = self.D * self.n1_after.get(v, 0) / c_v
            return disc + gamma * self._p_uni(w)
        # n == 3 中阶
        n_mid = self.n1_mid.get(v * self.V + w, 0)
        denom = self.n1_mid_sum.get(v, 0)
        if denom == 0:
            return self._p_uni(w)
        disc = max(n_mid - self.D, 0.0) / denom
        gamma = self.D * self.n1_mid_after.get(v, 0) / denom
        return disc + gamma * self._p_uni(w)

    def prob(self, history, w):
        if self.n == 1:
            return self._p_uni(w)
        if self.n == 2:
            return self._p_bi(history[-1], w)
        u, v = history[-2], history[-1]
        key = pack_ids((u, v, w), self.V)
        ctx = pack_ids((u, v), self.V)
        c_tri = self.counts.get(key, 0)
        c_ctx = self.ctx_counts.get(ctx, 0)
        if c_ctx == 0:
            return self._p_bi(v, w)
        disc = max(c_tri - self.D, 0.0) / c_ctx
        gamma = self.D * self.n1_ctx_after.get(ctx, 0) / c_ctx
        return disc + gamma * self._p_bi(v, w)

    def next_probs(self, history):
        return self.next_probs_fast(history)


# ---------------------------------------------------------------- 训练入口
def train_model(model_cls, seqs, n, V, word2id, min_freq=1, **kw):
    t0 = time.time()
    counts = count_ngrams(seqs, n, V, min_freq=min_freq)
    t_count = time.time() - t0
    bigram_counts = None
    if n == 3 and model_cls is KneserNeyLM:
        bigram_counts = count_ngrams(seqs, 2, V)
    t0 = time.time()
    lm = model_cls(n, V, word2id, counts, bigram_counts=bigram_counts, **kw)
    t_est = time.time() - t0
    return lm, t_count, t_est


# ---------------------------------------------------------------- 生成
def sample_from(p, temperature=1.0, top_k=None):
    if temperature <= 0:
        return int(np.argmax(p))
    positive = p > 0
    if not np.any(positive):
        raise ValueError("Cannot sample from a distribution without positive mass")
    logits = np.full_like(p, -np.inf, dtype=float)
    logits[positive] = np.log(p[positive]) / temperature
    if top_k:
        k = min(top_k, len(logits))
        idx = np.argpartition(logits, -k)[-k:]
        mask = np.ones_like(logits, dtype=bool)
        mask[idx] = False
        logits[mask] = -np.inf
    e = np.exp(logits - logits.max())
    e /= e.sum()
    return int(np.random.choice(len(e), p=e))


def generate(lm, seed_words, max_len=100, temperature=1.0, top_k=50, seed=42):
    """给定开头（已分词词表），续写 max_len 个词。OOV 词拆成单字。"""
    np.random.seed(seed)
    history = []
    for w in seed_words:
        if w in lm.word2id:
            history.append(lm.word2id[w])
        else:
            history.extend(lm.word2id.get(c, lm.word2id[UNK]) for c in w)
    out = list(history)
    full_hist = [0] * (lm.n - 1) + history
    for _ in range(max_len):
        ctx = full_hist[-(lm.n - 1):] if lm.n > 1 else []
        p = lm.next_probs(ctx)
        w = sample_from(p, temperature, top_k)
        if w == 1:  # </s>
            break
        out.append(w)
        full_hist.append(w)
    return [lm.id2word[i] for i in out]


def perplexity(lm, seqs, n):
    logp = 0.0
    total = 0
    for s in seqs:
        seq = [0] * (n - 1) + s + [1]
        for i in range(n - 1, len(seq)):
            w = seq[i]
            h = seq[i - n + 1:i]
            p = lm.prob(h, w)
            logp += np.log(max(p, 1e-12))
            total += 1
    return float(np.exp(-logp / total))
