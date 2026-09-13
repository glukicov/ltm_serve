"""Exact context caching for TabFM: encode the training rows once, then serve each request in O(n_query x n_train).

Why this is exact (read off tabfm/src/pytorch/model.py):

* Every attention that mixes *rows* masks its keys to the training rows. The column set-transformer's
  inducing points attend to training rows only (`ColEmbedding` mask), and the 24-layer in-context encoder
  lets every row attend to training rows only (`ICLearning` mask).
* Everything else (cell embedding, the row transformer across columns, feed-forwards, norms) is per row.

So the training rows' hidden states never depend on the query rows, and query rows never see each other.
Stock `predict_proba` still re-runs the full (n_train + n_query)-row sequence through all 24 layers for
every call. Here we run the training rows once and keep, per ensemble member:

* the inducing-point states `hidden` of each column set-transformer block (2 stages x 3 blocks), and
* the key/value projections of the training rows in each of the 24 in-context layers,

after which a query row needs only its own per-row stages plus cross-attention into those caches. The
same argument makes dynamic batching of unrelated requests exact: a row's prediction cannot depend on
which other rows share its batch.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tabfm import TabFMClassifier
from tabfm.src.classifier_and_regressor import _pad_features
from torch import Tensor

from ltm_serve.model import synchronize

# Constant from tabfm's MultiheadAttention (matches JAX PerDimScale).
_PER_DIM_SCALE = 1.442695041


@dataclass
class ContextCache:
    """Per-ensemble-member states of the training rows. Leading axis E = ensemble members."""

    col_hidden: list[list[Tensor]]  # [stage 0/1][block] -> [E, columns, inducing points, dim]
    icl_kv: list[tuple[Tensor, Tensor]]  # [layer] -> (K, V), each [E, heads, n_train, head_dim]
    cat_mask: Tensor  # [E, H]
    d: Tensor  # [E] active feature count per member
    n_train: int

    def nbytes(self) -> int:
        tensors = [t for stage in self.col_hidden for t in stage] + [t for kv in self.icl_kv for t in kv]
        return sum(t.numel() * t.element_size() for t in tensors)


def _project_kv(attn: Any, key_in: Tensor) -> tuple[Tensor, Tensor]:
    b, t, _ = key_in.shape
    k = attn.key_ln(attn.k_proj(key_in).view(b, t, attn.nhead, attn.hd)).transpose(1, 2)
    v = attn.v_proj(key_in).view(b, t, attn.nhead, attn.hd).transpose(1, 2)
    return k, v


def _attend(attn: Any, query_in: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """tabfm MultiheadAttention.forward with precomputed K/V (the in-context encoder has no RoPE)."""
    b, tq, d = query_in.shape
    q = attn.query_ln(attn.q_proj(query_in).view(b, tq, attn.nhead, attn.hd))
    scale = _PER_DIM_SCALE / math.sqrt(attn.hd) * F.softplus(attn.per_dim_scale.float())
    q = (q * scale.to(q.dtype)).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, scale=1.0)
    out: Tensor = attn.out_proj(o.transpose(1, 2).reshape(b, tq, d))
    return out


def _block_with_kv(blk: Any, x: Tensor, k: Tensor, v: Tensor) -> Tensor:
    x = x + blk.post_attn_ln(_attend(blk.attn, blk.pre_attn_ln(x), k, v))
    out: Tensor = x + blk._ff(x)
    return out


def _chunks(n: int, size: int | None) -> list[slice]:
    step = n if size is None else size
    return [slice(s, min(s + step, n)) for s in range(0, n, step)]


def _col_stage_context(col: Any, x: Tensor) -> tuple[Tensor, list[Tensor]]:
    """ColEmbedding over training rows only, returning inducing-point states per block."""
    b, t, hc, e = x.shape
    src = x.permute(0, 2, 1, 3).reshape(b * hc, t, e)
    outs: list[Tensor] = []
    hiddens: list[list[Tensor]] = [[] for _ in col.tf_col.blocks]
    for sl in _chunks(src.shape[0], col.col_chunk_size):
        s = src[sl]
        for i, blk in enumerate(col.tf_col.blocks):
            ind = blk.ind_vectors.unsqueeze(0).expand(s.shape[0], -1, -1)
            hidden = blk.mab1(ind, s, s)  # all keys are training rows, so the mask is all-true
            s = blk.mab2(s, hidden, hidden)
            hiddens[i].append(hidden)
        outs.append(col.ln_w(col.out_w(s)))
    out = torch.cat(outs).reshape(b, hc, t, e).permute(0, 2, 1, 3)
    stacked = [torch.cat(h).reshape(b, hc, *h[0].shape[1:]) for h in hiddens]
    return out, stacked


def _col_stage_query(col: Any, x: Tensor, hiddens: list[Tensor]) -> Tensor:
    b, t, hc, e = x.shape
    src = x.permute(0, 2, 1, 3).reshape(b * hc, t, e)
    flat = [h.reshape(b * hc, *h.shape[2:]) for h in hiddens]
    outs = []
    for sl in _chunks(src.shape[0], col.col_chunk_size):
        s = src[sl]
        for blk, hidden in zip(col.tf_col.blocks, flat, strict=True):
            s = blk.mab2(s, hidden[sl], hidden[sl])
        outs.append(col.ln_w(col.out_w(s)))
    return torch.cat(outs).reshape(b, hc, t, e).permute(0, 2, 1, 3)


def _y_encode(icl: Any, y: Tensor, dtype: torch.dtype) -> Tensor:
    out: Tensor = icl.y_encoder(y) if icl.is_classifier else icl.y_encoder(y[..., None].to(dtype))
    return out


@torch.inference_mode()
def encode_context(model: Any, X: Tensor, y: Tensor, cat_mask: Tensor, d: Tensor) -> ContextCache:
    """Run the training rows through TabFM once. X: [E, n_train, H], y: [E, n_train]."""
    dtype = model.cls_tokens.dtype
    b, t, _ = X.shape
    x = torch.nan_to_num(X, nan=-100.0).to(dtype)
    train_size = torch.full((b,), t, dtype=torch.long, device=X.device)
    emb = model.cell_embedder(x, y, train_size, cat_mask, d=d)
    emb, hidden_1 = _col_stage_context(model.col_embedder, emb)
    emb = torch.cat([model.cls_tokens.expand(b, t, -1, -1), emb], dim=2)
    emb = model.row_interactor(emb, d=d)
    emb, hidden_2 = _col_stage_context(model.col_embedder_2, emb)
    r = model.row_interactor_2(emb, d=d)

    icl = model.icl_predictor
    r = r + _y_encode(icl, y, dtype)
    kv: list[tuple[Tensor, Tensor]] = []
    for blk in icl.tf_icl.blocks:
        xn = blk.pre_attn_ln(r)
        k, v = _project_kv(blk.attn, xn)
        kv.append((k, v))
        r = r + blk.post_attn_ln(_attend(blk.attn, xn, k, v))
        r = r + blk._ff(r)
    return ContextCache(col_hidden=[hidden_1, hidden_2], icl_kv=kv, cat_mask=cat_mask, d=d, n_train=t)


@torch.inference_mode()
def forward_with_cache(model: Any, cache: ContextCache, X: Tensor, members: slice | None = None) -> Tensor:
    """Logits for query rows X: [E', n_query, H] against the cached context -> [E', n_query, classes]."""
    return _forward_impl(model, cache, X, members or slice(0, X.shape[0]))


def _forward_impl(model: Any, cache: ContextCache, X: Tensor, sl: slice) -> Tensor:
    """Undecorated body of `forward_with_cache`, so `torch.compile` can trace it without an inference-mode break."""
    dtype = model.cls_tokens.dtype
    b, t, _ = X.shape
    x = torch.nan_to_num(X, nan=-100.0).to(dtype)
    d, cat_mask = cache.d[sl], cache.cat_mask[sl]
    no_train = torch.zeros((b,), dtype=torch.long, device=X.device)
    y_dummy = torch.zeros((b, t), dtype=torch.long if model.is_classifier else dtype, device=X.device)

    emb = model.cell_embedder(x, y_dummy, no_train, cat_mask, d=d)
    emb = _col_stage_query(model.col_embedder, emb, [h[sl] for h in cache.col_hidden[0]])
    emb = torch.cat([model.cls_tokens.expand(b, t, -1, -1), emb], dim=2)
    emb = model.row_interactor(emb, d=d)
    emb = _col_stage_query(model.col_embedder_2, emb, [h[sl] for h in cache.col_hidden[1]])
    r = model.row_interactor_2(emb, d=d)

    icl = model.icl_predictor
    for blk, (k, v) in zip(icl.tf_icl.blocks, cache.icl_kv, strict=True):
        r = _block_with_kv(blk, r, k[sl], v[sl])
    out: Tensor = icl.decoder(icl.ln(r))
    return out


class CachedTabFMClassifier:
    """A fitted `TabFMClassifier` whose context is encoded once; `predict_proba` then touches query rows only.

    Preprocessing, ensembling, class-shift correction and logit averaging are all delegated to the wrapped
    classifier, so outputs match stock `predict_proba` up to floating-point kernel differences.
    """

    def __init__(
        self,
        clf: TabFMClassifier,
        member_batch_size: int | None = None,
        compile_mode: str | None = None,
        bucket_rows: bool = False,
    ) -> None:
        """
        Args:
          member_batch_size: ensemble members per forward pass (None = all at once).
          compile_mode: `torch.compile` mode for the query forward ("default", "reduce-overhead", "max-autotune").
          bucket_rows: pad query rows to the next power of two. Exact (rows are independent) and bounds the number
            of distinct shapes, which is what lets compiled graphs and CUDA graphs be reused across requests.
        """
        if clf.permute_categorical or clf.n_feature_crosses or clf.n_svd_features:
            raise ValueError("categorical permutation, feature crosses and SVD features are not supported")
        self.clf = clf
        self.member_batch_size = member_batch_size
        self.bucket_rows = bucket_rows
        self.cache: ContextCache | None = None
        self.encode_seconds = 0.0
        self.last_timings: dict[str, float] = {}
        self._max_features = 0
        self._example_row = pd.DataFrame()
        self._forward: Any = (
            torch.compile(_forward_impl, mode=compile_mode, dynamic=None) if compile_mode else _forward_impl
        )

    @property
    def device(self) -> torch.device:
        device: torch.device = next(self.clf.model.parameters()).device
        return device

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> CachedTabFMClassifier:
        self.clf.fit(X, y)
        eg = self.clf.ensemble_generator_
        # Reuse the library's own tensor assembly for the context: one dummy query row, sliced off below.
        data = eg.transform(self.clf.X_encoder_.transform(X.iloc[:1]))
        Xs, ys, cat_masks, ds, _ = eg.prepare_ensemble_tensors(data)
        n_train = ys.shape[1]
        dev = self.device
        t0 = time.perf_counter()
        parts = []
        for sl in _chunks(Xs.shape[0], self.member_batch_size):
            parts.append(
                encode_context(
                    self.clf.model,
                    torch.from_numpy(Xs[sl, :n_train]).to(dev, torch.float32),
                    torch.from_numpy(ys[sl]).to(dev),
                    torch.from_numpy(cat_masks[sl]).to(dev),
                    torch.from_numpy(ds[sl]).to(dev),
                )
            )
        self.cache = _concat_caches(parts)
        synchronize(dev)
        self.encode_seconds = time.perf_counter() - t0
        self._max_features = Xs.shape[-1]
        self._example_row = X.iloc[:1]
        return self

    def warm_up(self, max_rows: int) -> float:
        """Run every power-of-two row bucket up to `max_rows` once, so compilation (and CUDA-graph capture) happens
        before live traffic instead of inside the first unlucky requests. Returns the seconds it took."""
        t0 = time.perf_counter()
        n = 1
        while n <= max_rows:
            self.predict_proba(pd.concat([self._example_row] * n, ignore_index=True))
            n *= 2
        synchronize(self.device)
        return time.perf_counter() - t0

    def query_tensors(self, X: pd.DataFrame) -> np.ndarray:
        """Query rows only, preprocessed per ensemble member exactly as EnsembleGenerator._transform_features does."""
        eg = self.clf.ensemble_generator_
        Xf = eg.unique_filter_.transform(self.clf.X_encoder_.transform(X))
        members = []
        for norm_method, configs in eg.ensemble_configs_.items():
            transformed = eg.preprocessors_[norm_method].transform(Xf)
            for shuffle_pattern, _, _, _ in configs:
                members.append(_pad_features(transformed[:, shuffle_pattern], self._max_features))
        return np.stack(members).astype(np.float32)

    def logits(self, X: pd.DataFrame) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("call fit() first")
        t0 = time.perf_counter()
        members = self.query_tensors(X)
        n_rows = members.shape[1]
        if self.bucket_rows:
            padded = 1 << max(0, (n_rows - 1).bit_length())
            members = np.pad(members, ((0, 0), (0, padded - n_rows), (0, 0)))
        Xq = torch.from_numpy(members).to(self.device)
        synchronize(self.device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            outs = [
                self._forward(self.clf.model, self.cache, Xq[sl], sl)
                for sl in _chunks(Xq.shape[0], self.member_batch_size)
            ]
            raw = torch.cat(outs)[:, :n_rows, : self.clf.n_classes_].float().cpu().numpy()
        t2 = time.perf_counter()
        offsets = [c[1] for configs in self.clf.ensemble_generator_.ensemble_configs_.values() for c in configs]
        out = np.stack([np.roll(raw[i], -off, axis=-1) for i, off in enumerate(offsets)])
        self.last_timings = {"preprocess_ms": (t1 - t0) * 1e3, "forward_ms": (t2 - t1) * 1e3}
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs: np.ndarray = self.clf._process_logits(self.logits(X))
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        labels: np.ndarray = self.clf.classes_[proba.argmax(axis=1)]
        return labels


def _concat_caches(parts: list[ContextCache]) -> ContextCache:
    if len(parts) == 1:
        return parts[0]
    return ContextCache(
        col_hidden=[
            [torch.cat([p.col_hidden[s][i] for p in parts]) for i in range(len(parts[0].col_hidden[s]))]
            for s in range(2)
        ],
        icl_kv=[
            (torch.cat([p.icl_kv[li][0] for p in parts]), torch.cat([p.icl_kv[li][1] for p in parts]))
            for li in range(len(parts[0].icl_kv))
        ],
        cat_mask=torch.cat([p.cat_mask for p in parts]),
        d=torch.cat([p.d for p in parts]),
        n_train=parts[0].n_train,
    )
