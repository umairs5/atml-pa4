"""Packed storage accounting and exact bit-packing/unpacking utilities for Task 1.

All packed-byte formulas match the PA4 specification:
- Uncompressed FP16: 2 bytes per unique parameter (shared/tied weights counted once).
- Unstructured 16-bit sparse: 1 mask bit per eligible weight (byte-aligned per matrix)
  + 16 bits per nonzero eligible weight + uncompressed FP16 bytes.
- Group-wise asymmetric weight-only quantization (group size g=128 along input columns):
  - Packed b-bit weights: ceil(d_out * n_groups_per_row * g * b / 8) bytes
  - FP16 group scales: 2 * n_groups bytes
  - Packed b-bit integer zero-points: ceil(n_groups * b / 8) bytes
  - Optional compensating FP16 channel scales (when scale folding is invalid): 2 * d_in bytes
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

SUPPORTED_BITS = (2, 3, 4, 8)
DEFAULT_GROUP_SIZE = 128


def pack_bits(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack a 1D unsigned integer tensor with values in [0, 2^bits - 1] into uint8 bytes.

    Uses contiguous little-endian bit packing (exact 8*3 -> 3 bytes for 3-bit, etc.).
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"Unsupported bit width {bits}; expected one of {SUPPORTED_BITS}")
    flat = values.reshape(-1).to(torch.uint8)
    n = flat.numel()
    if bits == 8:
        return flat.contiguous()
    if bits == 4:
        if n % 2 != 0:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
        pairs = flat.reshape(-1, 2)
        return (pairs[:, 0] | (pairs[:, 1] << 4)).contiguous()
    if bits == 2:
        rem = n % 4
        if rem:
            flat = torch.cat([flat, torch.zeros(4 - rem, dtype=torch.uint8, device=flat.device)])
        quads = flat.reshape(-1, 4)
        return (
            quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)
        ).contiguous()
    # bits == 3: pack blocks of 8 3-bit values into 3 bytes (24 bits)
    rem = n % 8
    if rem:
        flat = torch.cat([flat, torch.zeros(8 - rem, dtype=torch.uint8, device=flat.device)])
    octs = flat.reshape(-1, 8).to(torch.int32)
    packed24 = (
        octs[:, 0]
        | (octs[:, 1] << 3)
        | (octs[:, 2] << 6)
        | (octs[:, 3] << 9)
        | (octs[:, 4] << 12)
        | (octs[:, 5] << 15)
        | (octs[:, 6] << 18)
        | (octs[:, 7] << 21)
    )
    b0 = (packed24 & 0xFF).to(torch.uint8)
    b1 = ((packed24 >> 8) & 0xFF).to(torch.uint8)
    b2 = ((packed24 >> 16) & 0xFF).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=-1).reshape(-1).contiguous()


def unpack_bits(packed: torch.Tensor, bits: int, num_elements: int) -> torch.Tensor:
    """Unpack a uint8 byte stream produced by `pack_bits` back into uint8 values."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"Unsupported bit width {bits}; expected one of {SUPPORTED_BITS}")
    if bits == 8:
        return packed[:num_elements].to(torch.uint8)
    if bits == 4:
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        return torch.stack([lo, hi], dim=-1).reshape(-1)[:num_elements]
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(-1)[:num_elements]
    # bits == 3
    triples = packed.reshape(-1, 3).to(torch.int32)
    packed24 = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    vals = [(packed24 >> (3 * k)) & 0x07 for k in range(8)]
    return torch.stack(vals, dim=-1).reshape(-1)[:num_elements].to(torch.uint8)


@dataclass(frozen=True)
class QuantizedMatrixBytes:
    d_out: int
    d_in: int
    bits: int
    group_size: int
    n_groups: int
    weight_bytes: int
    scale_bytes: int
    zero_bytes: int
    channel_scale_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.scale_bytes + self.zero_bytes + self.channel_scale_bytes


def calc_quantized_matrix_bytes(
    d_out: int,
    d_in: int,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    has_unfolded_channel_scale: bool = False,
) -> QuantizedMatrixBytes:
    """Compute exact packed byte footprint for one group-wise quantized linear weight matrix."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"Unsupported bit width {bits}")
    n_groups_per_row = math.ceil(d_in / group_size)
    d_pad = n_groups_per_row * group_size
    n_groups = d_out * n_groups_per_row
    weight_bytes = math.ceil((d_out * d_pad * bits) / 8)
    scale_bytes = 2 * n_groups
    zero_bytes = math.ceil((n_groups * bits) / 8)
    channel_scale_bytes = (2 * d_in) if has_unfolded_channel_scale else 0
    return QuantizedMatrixBytes(
        d_out=d_out,
        d_in=d_in,
        bits=bits,
        group_size=group_size,
        n_groups=n_groups,
        weight_bytes=weight_bytes,
        scale_bytes=scale_bytes,
        zero_bytes=zero_bytes,
        channel_scale_bytes=channel_scale_bytes,
    )


def calc_sparse_matrix_bytes(
    d_out: int,
    d_in: int,
    nnz: int,
    align_per_row: bool = False,
) -> tuple[int, int, int]:
    """Compute (mask_bytes, value_bytes, total_bytes) for 1-bit-mask + FP16-nonzero storage."""
    if align_per_row:
        mask_bytes = d_out * math.ceil(d_in / 8)
    else:
        mask_bytes = math.ceil((d_out * d_in) / 8)
    value_bytes = 2 * nnz
    return mask_bytes, value_bytes, mask_bytes + value_bytes


@dataclass
class PackedQuantizedWeight:
    d_out: int
    d_in: int
    bits: int
    group_size: int
    n_groups_per_row: int
    qweight: torch.Tensor          # uint8 packed b-bit integers
    scales: torch.Tensor           # float16 [d_out, n_groups_per_row]
    qzeros: torch.Tensor           # uint8 packed b-bit integer zero-points
    channel_inv_scales: Optional[torch.Tensor] = None  # optional float16 [d_in]

    def actual_bytes(self) -> int:
        extra = self.channel_inv_scales.numel() * self.channel_inv_scales.element_size() if self.channel_inv_scales is not None else 0
        return (
            self.qweight.numel() * self.qweight.element_size()
            + self.scales.numel() * self.scales.element_size()
            + self.qzeros.numel() * self.qzeros.element_size()
            + extra
        )

    def dequantize(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        d_pad = self.n_groups_per_row * self.group_size
        n_groups = self.d_out * self.n_groups_per_row
        q = unpack_bits(self.qweight, self.bits, self.d_out * d_pad).reshape(
            self.d_out, self.n_groups_per_row, self.group_size
        ).to(torch.float32)
        z = unpack_bits(self.qzeros, self.bits, n_groups).reshape(
            self.d_out, self.n_groups_per_row, 1
        ).to(torch.float32)
        s = self.scales.to(torch.float32).unsqueeze(-1)
        w = s * (q - z)
        w = w.reshape(self.d_out, d_pad)[:, : self.d_in]
        if self.channel_inv_scales is not None:
            w = w * self.channel_inv_scales.to(torch.float32).unsqueeze(0)
        return w.to(dtype)

