"""Task 2 Part B: MQAR small decoder-only Transformer, attention mechanisms, and training/eval."""

from __future__ import annotations

import gzip
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.env import measure_cuda_latency_ms, reset_and_measure_peak_memory_bytes, set_seed

MQAR_T = 256
MQAR_VOCAB = 256
MQAR_VALUE_OFFSET = 128
MQAR_NUM_CLASSES = 64
MQAR_CKPT_STEPS = (500, 1000, 1500, 2000, 3000, 4500)


def sinusoidal_positional_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def build_sparse_head_masks(seq_len: int, mechanism: str, device: torch.device) -> Optional[torch.Tensor]:
    """Return boolean mask [1, H, L, L] (True = allowed) for static causal/sparse masks."""
    idx = torch.arange(seq_len, device=device)
    diff = idx[:, None] - idx[None, :]  # i - j
    causal = diff >= 0
    if mechanism == "dense":
        return causal[None, None, :, :]
    if mechanism == "sliding_w64":
        m = causal & (diff < 64)
        return m[None, None, :, :]
    if mechanism == "local_dilated":
        local_m = causal & (diff < 64)
        dilated_m = causal & (diff <= 252) & ((diff % 4) == 0)
        # Heads 0, 1 local; heads 2, 3 dilated
        return torch.stack([local_m, local_m, dilated_m, dilated_m], dim=0).unsqueeze(0)
    return None


def chunkwise_linear_elu1_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exact causal ELU+1 linear attention via chunkwise recurrent state (O(T) memory).

    Mathematically identical to S_t = S_{t-1} + phi(k_t) v_t^T, z_t = z_{t-1} + phi(k_t),
    a_t = (phi(q_t)^T S_t) / (phi(q_t)^T z_t + eps), with phi(x) = ELU(x) + 1.
    """
    B, H, L, d_h = q.shape
    phi_q = (F.elu(q.float()) + 1.0)
    phi_k = (F.elu(k.float()) + 1.0)
    v_f = v.float()

    S = torch.zeros((B, H, d_h, d_h), dtype=torch.float32, device=q.device)
    z = torch.zeros((B, H, d_h), dtype=torch.float32, device=q.device)

    out_chunks = []
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        c_len = end - start
        qc = phi_q[:, :, start:end, :]   # [B, H, c, d]
        kc = phi_k[:, :, start:end, :]   # [B, H, c, d]
        vc = v_f[:, :, start:end, :]     # [B, H, c, d]

        tril = torch.tril(torch.ones((c_len, c_len), dtype=torch.bool, device=q.device))
        qk_intra = torch.matmul(qc, kc.transpose(-2, -1)).masked_fill(~tril, 0.0)
        num_intra = torch.matmul(qk_intra, vc)
        den_intra = qk_intra.sum(dim=-1, keepdim=True)

        num_inter = torch.matmul(qc, S)  # [B, H, c, d]
        den_inter = torch.einsum("bhcd,bhd->bhc", qc, z).unsqueeze(-1)

        out_c = (num_intra + num_inter) / (den_intra + den_inter + eps)
        out_chunks.append(out_c.to(q.dtype))

        S = S + torch.matmul(kc.transpose(-2, -1), vc)
        z = z + kc.sum(dim=-2)

    return torch.cat(out_chunks, dim=-2)


def topk_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int = 64,
) -> torch.Tensor:
    """Content-selected causal top-k attention retaining top `top_k` largest QK^T scores per query."""
    B, H, L, d_h = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d_h))
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=q.device))
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~causal[None, None, :, :], neg_inf)
    if top_k < L:
        # Select top_k causal indices per row; note rows i < top_k have some -inf entries which stay -inf
        k_eff = min(top_k, L)
        _, top_idx = torch.topk(scores, k=k_eff, dim=-1, largest=True, sorted=False)
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep.scatter_(-1, top_idx, True)
        keep = keep & causal[None, None, :, :]
        scores = scores.masked_fill(~keep, neg_inf)
    probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


class MQARAttention(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, mechanism: str = "dense", top_k: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.mechanism = mechanism
        self.top_k = top_k

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cached_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        if self.mechanism == "linear_elu1":
            attn_out = chunkwise_linear_elu1_attention(q, k, v, chunk_size=64)
        elif self.mechanism == "topk_64":
            attn_out = topk_causal_attention(q, k, v, top_k=self.top_k)
        elif self.mechanism in ("dense", "sliding_w64", "local_dilated"):
            if cached_mask is None:
                cached_mask = build_sparse_head_masks(L, self.mechanism, x.device)
            scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            scores = scores.masked_fill(~cached_mask, torch.finfo(scores.dtype).min)
            probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
            attn_out = torch.matmul(probs, v)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(attn_out)


class MQARBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, mechanism: str = "dense"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MQARAttention(d_model=d_model, n_heads=n_heads, mechanism=mechanism)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, cached_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cached_mask=cached_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MQARTransformer(nn.Module):
    """Two-layer pre-normalized decoder-only Transformer for MQAR query-position classification."""

    def __init__(
        self,
        vocab_size: int = MQAR_VOCAB,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        num_classes: int = MQAR_NUM_CLASSES,
        mechanism: str = "dense",
    ):
        super().__init__()
        self.d_model = d_model
        self.mechanism = mechanism
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [MQARBlock(d_model=d_model, n_heads=n_heads, mechanism=mechanism) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)
        self._mask_cache: dict[tuple[str, int, str], Optional[torch.Tensor]] = {}

    def _get_mask(self, seq_len: int, mechanism: str, device: torch.device) -> Optional[torch.Tensor]:
        key = (mechanism, seq_len, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = build_sparse_head_masks(seq_len, mechanism, device)
        return self._mask_cache[key]

    def forward(
        self,
        tokens: torch.Tensor,
        query_positions: Optional[torch.Tensor] = None,
        override_mechanism: Optional[str] = None,
    ) -> torch.Tensor:
        B, L = tokens.shape
        mech = override_mechanism or self.mechanism
        pe = sinusoidal_positional_encoding(L, self.d_model, tokens.device).to(self.tok_emb.weight.dtype)
        x = self.tok_emb(tokens) + pe.unsqueeze(0)
        mask = self._get_mask(L, mech, tokens.device)
        for blk in self.blocks:
            # Allow test-time mask override (e.g. dense trained -> W=64 test control)
            if override_mechanism is not None and mech in ("dense", "sliding_w64", "local_dilated"):
                prev = blk.attn.mechanism
                blk.attn.mechanism = mech
                x = blk(x, cached_mask=mask)
                blk.attn.mechanism = prev
            else:
                x = blk(x, cached_mask=mask)
        x = self.ln_f(x)
        if query_positions is not None:
            b_idx = torch.arange(B, device=tokens.device).unsqueeze(1)
            x = x[b_idx, query_positions]  # [B, K, D]
        return self.head(x)


def load_mqar_jsonl_gz(path: str | Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_training_stream_tensors(
    dataset_root: Path,
    k: int,
    regime: str,
    total_examples: int,
    start_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate deterministic MQAR training tensors using the course mqar_generator."""
    import sys

    scripts_dir = str(dataset_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from mqar_generator import make_example

    tokens = np.empty((total_examples, MQAR_T), dtype=np.int64)
    q_pos = np.empty((total_examples, k), dtype=np.int64)
    answers = np.empty((total_examples, k), dtype=np.int64)

    for i in range(total_examples):
        ex = make_example("train", k, regime, start_index + i)
        tokens[i] = ex["tokens"]
        q_pos[i] = ex["query_positions"]
        answers[i] = np.asarray(ex["answer_ids"], dtype=np.int64) - MQAR_VALUE_OFFSET

    return torch.from_numpy(tokens), torch.from_numpy(q_pos), torch.from_numpy(answers)


@torch.inference_mode()
def evaluate_mqar_records(
    model: MQARTransformer,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 128,
    override_mechanism: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate MQAR records returning overall, per-(K, regime), and distance-binned accuracy."""
    model.eval()
    by_k: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        by_k.setdefault(int(r["K"]), []).append(r)

    total_correct = 0
    total_queries = 0
    total_loss_sum = 0.0
    cell_stats: dict[tuple[int, str], list[int]] = {}  # [correct, count]
    bin_bounds = [(0, 32), (32, 64), (64, 128), (128, 256)]
    bin_stats: dict[str, list[int]] = {f"[{lo},{hi})": [0, 0] for lo, hi in bin_bounds}

    for k, recs in sorted(by_k.items()):
        for start in range(0, len(recs), batch_size):
            chunk = recs[start : start + batch_size]
            toks = torch.tensor([r["tokens"] for r in chunk], dtype=torch.long, device=device)
            qpos = torch.tensor([r["query_positions"] for r in chunk], dtype=torch.long, device=device)
            ans = torch.tensor(
                [[a - MQAR_VALUE_OFFSET for a in r["answer_ids"]] for r in chunk],
                dtype=torch.long,
                device=device,
            )
            logits = model(toks, query_positions=qpos, override_mechanism=override_mechanism)
            loss = F.cross_entropy(logits.reshape(-1, MQAR_NUM_CLASSES), ans.reshape(-1), reduction="sum")
            total_loss_sum += float(loss.item())
            preds = logits.argmax(dim=-1).cpu().numpy()
            ans_np = ans.cpu().numpy()

            for row_i, r in enumerate(chunk):
                reg = r["regime"]
                cs = cell_stats.setdefault((k, reg), [0, 0])
                for q_j, dist in enumerate(r["distances"]):
                    ok = int(preds[row_i, q_j] == ans_np[row_i, q_j])
                    total_correct += ok
                    total_queries += 1
                    cs[0] += ok
                    cs[1] += 1
                    for lo, hi in bin_bounds:
                        if lo <= dist < hi:
                            bs = bin_stats[f"[{lo},{hi})"]
                            bs[0] += ok
                            bs[1] += 1
                            break

    by_cell = {
        f"K{k}_{reg}": {"acc": c / n, "correct": c, "count": n}
        for (k, reg), (c, n) in sorted(cell_stats.items())
    }
    by_bin = {
        b: {"acc": (c / n if n else None), "correct": c, "count": n}
        for b, (c, n) in bin_stats.items()
    }
    return {
        "accuracy": total_correct / max(total_queries, 1),
        "loss": total_loss_sum / max(total_queries, 1),
        "total_queries": total_queries,
        "by_cell": by_cell,
        "by_distance_bin": by_bin,
    }


def train_mqar_run(
    mechanism: str,
    train_k: int,
    train_regime: str,
    seed: int,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dev_records: list[dict[str, Any]],
    device: torch.device,
    total_updates: int = 4500,
    batch_size: int = 32,
    peak_lr: float = 8e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    min_lr_frac: float = 0.15,
) -> tuple[MQARTransformer, dict[str, Any]]:
    """Train one MQAR run following the fixed PA4 schedule and checkpoint evaluation grid."""
    set_seed(seed)
    model = MQARTransformer(mechanism=mechanism).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_updates - warmup_steps))
        return max(min_lr_frac, 1.0 - (1.0 - min_lr_frac) * progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    tok_all, qpos_all, ans_all = train_tensors
    if len(tok_all) < total_updates * batch_size:
        raise ValueError("Insufficient pre-generated training examples")

    # Filter dev records matching train_k for focused learning curves (plus full dev by regime)
    dev_k = [r for r in dev_records if int(r["K"]) == train_k]
    learning_curve: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    model.train()
    for step in range(1, total_updates + 1):
        s0 = (step - 1) * batch_size
        s1 = s0 + batch_size
        tok = tok_all[s0:s1].to(device)
        qpos = qpos_all[s0:s1].to(device)
        ans = ans_all[s0:s1].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(tok, query_positions=qpos)
        loss = F.cross_entropy(logits.reshape(-1, MQAR_NUM_CLASSES), ans.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step in MQAR_CKPT_STEPS:
            dev_eval = evaluate_mqar_records(model, dev_k, device=device, batch_size=128)
            learning_curve.append({
                "step": step,
                "train_loss": float(loss.item()),
                "dev_loss": dev_eval["loss"],
                "dev_acc": dev_eval["accuracy"],
                "by_cell": dev_eval["by_cell"],
                "by_distance_bin": dev_eval["by_distance_bin"],
                "elapsed_sec": time.perf_counter() - t_start,
            })
            model.train()

    total_time = time.perf_counter() - t_start
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, {
        "mechanism": mechanism,
        "train_k": train_k,
        "train_regime": train_regime,
        "seed": seed,
        "trainable_params": int(n_params),
        "train_time_sec": total_time,
        "learning_curve": learning_curve,
    }


def theoretical_cost_at_T256() -> dict[str, dict[str, Any]]:
    """Theoretical causal attention pairs and persistent KV/recurrent state bytes per layer at T=256."""
    T, H, d_h = 256, 4, 32
    dense_pairs = H * (T * (T + 1) // 2)
    w64_pairs = H * sum(min(i + 1, 64) for i in range(T))
    dilated_head_pairs = sum(1 + (min(i, 252) // 4) for i in range(T))
    local_head_pairs = sum(min(i + 1, 64) for i in range(T))
    local_dilated_pairs = 2 * local_head_pairs + 2 * dilated_head_pairs

    return {
        "dense": {
            "computed_pairs_per_layer": dense_pairs,
            "attended_pairs_per_layer": dense_pairs,
            "persistent_kv_bytes_per_layer_fp32": 2 * H * T * d_h * 4,
        },
        "sliding_w64": {
            "computed_pairs_per_layer": w64_pairs,
            "attended_pairs_per_layer": w64_pairs,
            "persistent_kv_bytes_per_layer_fp32": 2 * H * 64 * d_h * 4,
        },
        "local_dilated": {
            "computed_pairs_per_layer": local_dilated_pairs,
            "attended_pairs_per_layer": local_dilated_pairs,
            "persistent_kv_bytes_per_layer_fp32": 2 * (2 * 64 + 2 * 256) * d_h * 4,
        },
        "linear_elu1": {
            "computed_pairs_per_layer": 0,
            "attended_pairs_per_layer": 0,
            "persistent_state_bytes_per_layer_fp32": H * (d_h * d_h + d_h) * 4,
        },
        "topk_64": {
            "computed_pairs_per_layer": dense_pairs,
            "attended_pairs_per_layer": w64_pairs,
            "persistent_kv_bytes_per_layer_fp32": 2 * H * T * d_h * 4,
        },
    }


@torch.inference_mode()
def benchmark_part_b_inference_sweep(
    lengths: tuple[int, ...] = (256, 1024, 4096, 8192),
    B: int = 2,
    H: int = 4,
    d_h: int = 32,
    device: str | torch.device = "cuda",
    repeats: int = 10,
) -> list[dict[str, Any]]:
    """Forward-only inference sweep across sequence lengths with fixed random embeddings."""
    dev = torch.device(device)
    d_model = H * d_h
    mechs = ("dense", "sliding_w64", "local_dilated", "linear_elu1", "topk_64")
    results = []

    for mech in mechs:
        model = MQARTransformer(d_model=d_model, n_heads=H, n_layers=2, mechanism=mech).to(dev).eval()
        for T in lengths:
            gen = torch.Generator(device="cpu").manual_seed(6304 + T)
            toks = torch.randint(0, MQAR_VOCAB, (B, T), generator=gen, dtype=torch.long).to(dev)
            row: dict[str, Any] = {
                "mechanism": mech,
                "T": T,
                "B": B,
                "H": H,
                "d_head": d_h,
                "materializes_dense_mask": mech in ("dense", "sliding_w64", "local_dilated", "topk_64"),
                "status": "ok",
            }
            try:
                def run_fwd():
                    return model(toks)

                _, mem = reset_and_measure_peak_memory_bytes(run_fwd, dev)
                lat = measure_cuda_latency_ms(run_fwd, warmup=3, repeats=repeats, device=dev)
                row["fwd_latency"] = lat
                row["fwd_memory"] = mem
            except torch.cuda.OutOfMemoryError as exc:
                row["status"] = "OOM"
                row["error"] = str(exc)
            results.append(row)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return results

