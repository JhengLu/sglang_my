"""
Microbenchmark for the tree-sparse decode selector path.

This script is intended for fast iteration on:
1. The fused CUDA kernel `fused_sparse_select_and_build`
2. The production-like cached wrapper path around that kernel

Default shapes are chosen to resemble the current BS=1 Qwen tree-sparse run.

Examples:
    python -m sglang.jit_kernel.benchmark_tree_sparse_selector
    python -m sglang.jit_kernel.benchmark_tree_sparse_selector --iters 2000
    python -m sglang.jit_kernel.benchmark_tree_sparse_selector --bs 4 --num-chunks 1024
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from sglang.jit_kernel.tree_sparse_topk import fused_sparse_select_and_build


@dataclass
class BenchmarkInputs:
    queries_full: torch.Tensor
    queries_grouped: torch.Tensor
    centroids: torch.Tensor
    chunk_offsets: torch.Tensor
    chunk_starts: torch.Tensor
    chunk_ends: torch.Tensor
    tile_offsets: torch.Tensor
    schedule_offsets: torch.Tensor
    schedule_req_indices: torch.Tensor
    schedule_tile_indices: torch.Tensor
    seq_lens: torch.Tensor
    sparse_req_mask: torch.Tensor
    req_pool_indices: torch.Tensor
    req_to_token: torch.Tensor
    top_k: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    scaling: float
    always_include_first: int
    always_include_recent: int


def _build_chunk_boundaries(bs: int, num_chunks: int, seq_len: int, device: torch.device):
    chunk_offsets = [0]
    starts_per_req = []
    ends_per_req = []

    for _ in range(bs):
        base = seq_len // num_chunks
        remainder = seq_len % num_chunks
        lengths = torch.full((num_chunks,), base, dtype=torch.int32, device=device)
        if remainder > 0:
            lengths[:remainder] += 1
        starts = torch.cumsum(lengths, dim=0) - lengths
        ends = starts + lengths - 1
        starts_per_req.append(starts)
        ends_per_req.append(ends)
        chunk_offsets.append(chunk_offsets[-1] + num_chunks)

    return (
        torch.tensor(chunk_offsets, dtype=torch.int32, device=device),
        torch.cat(starts_per_req, dim=0),
        torch.cat(ends_per_req, dim=0),
    )


def _build_schedule(bs: int, num_chunks: int, device: torch.device):
    tile_offsets = [0]
    schedule_req_indices = []
    schedule_tile_indices = []
    num_tiles_per_req = (num_chunks + 31) // 32
    for req_idx in range(bs):
        tile_offsets.append(tile_offsets[-1] + num_tiles_per_req)
        for tile_idx in range(num_tiles_per_req):
            schedule_req_indices.append(req_idx)
            schedule_tile_indices.append(tile_idx)
    total_tiles = tile_offsets[-1]
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_schedule_blocks = min(max(total_tiles, 1), num_sms)
    schedule_offsets = [
        (total_tiles * block_idx) // num_schedule_blocks
        for block_idx in range(num_schedule_blocks + 1)
    ]
    return (
        torch.tensor(tile_offsets, dtype=torch.int32, device=device),
        torch.tensor(schedule_offsets, dtype=torch.int32, device=device),
        torch.tensor(schedule_req_indices, dtype=torch.int32, device=device),
        torch.tensor(schedule_tile_indices, dtype=torch.int32, device=device),
    )


def create_inputs(args: argparse.Namespace) -> BenchmarkInputs:
    device = torch.device(args.device)
    if args.num_qo_heads % args.num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")

    queries_full = torch.randn(
        args.bs,
        args.num_qo_heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    group_size = args.num_qo_heads // args.num_kv_heads
    queries_grouped = queries_full.view(
        args.bs, args.num_kv_heads, group_size, args.head_dim
    ).mean(dim=2)

    centroids_bf16 = torch.randn(
        args.bs * args.num_chunks,
        args.num_kv_heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    centroids = centroids_bf16.to(torch.float8_e4m3fn)

    chunk_offsets, chunk_starts, chunk_ends = _build_chunk_boundaries(
        args.bs, args.num_chunks, args.seq_len, device
    )
    tile_offsets, schedule_offsets, schedule_req_indices, schedule_tile_indices = (
        _build_schedule(args.bs, args.num_chunks, device)
    )

    seq_lens = torch.full((args.bs,), args.seq_len, dtype=torch.int32, device=device)
    sparse_req_mask = torch.ones(args.bs, dtype=torch.int32, device=device)
    req_pool_indices = torch.arange(args.bs, dtype=torch.int32, device=device)
    req_to_token = torch.arange(
        args.bs * args.max_ctx_len, dtype=torch.int32, device=device
    ).view(args.bs, args.max_ctx_len)

    return BenchmarkInputs(
        queries_full=queries_full,
        queries_grouped=queries_grouped,
        centroids=centroids,
        chunk_offsets=chunk_offsets,
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        tile_offsets=tile_offsets,
        schedule_offsets=schedule_offsets,
        schedule_req_indices=schedule_req_indices,
        schedule_tile_indices=schedule_tile_indices,
        seq_lens=seq_lens,
        sparse_req_mask=sparse_req_mask,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        top_k=args.top_k,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        scaling=args.scaling,
        always_include_first=args.always_include_first,
        always_include_recent=args.always_include_recent,
    )


def measure_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def bench_query_grouping(data: BenchmarkInputs, warmup: int, iters: int) -> float:
    group_size = data.num_qo_heads // data.num_kv_heads

    def fn():
        return data.queries_full.view(
            data.queries_full.shape[0], data.num_kv_heads, group_size, data.head_dim
        ).mean(dim=2)

    return measure_ms(fn, warmup, iters)


def bench_fused_kernel(data: BenchmarkInputs, warmup: int, iters: int) -> float:
    def fn():
        return fused_sparse_select_and_build(
            data.queries_grouped,
            data.centroids,
            data.chunk_offsets,
            data.chunk_starts,
            data.chunk_ends,
            data.seq_lens,
            data.sparse_req_mask,
            data.req_pool_indices,
            data.req_to_token,
            num_kv_heads=data.num_kv_heads,
            head_dim=data.head_dim,
            top_k=data.top_k,
            scaling=data.scaling,
            always_include_first=data.always_include_first,
            always_include_recent=data.always_include_recent,
            tile_offsets=data.tile_offsets,
            schedule_offsets=data.schedule_offsets,
            schedule_req_indices=data.schedule_req_indices,
            schedule_tile_indices=data.schedule_tile_indices,
        )

    return measure_ms(fn, warmup, iters)


def bench_postprocess(data: BenchmarkInputs, warmup: int, iters: int) -> float:
    _, _, topk_indices = fused_sparse_select_and_build(
        data.queries_grouped,
        data.centroids,
        data.chunk_offsets,
        data.chunk_starts,
        data.chunk_ends,
        data.seq_lens,
        data.sparse_req_mask,
        data.req_pool_indices,
        data.req_to_token,
        num_kv_heads=data.num_kv_heads,
        head_dim=data.head_dim,
        top_k=data.top_k,
        scaling=data.scaling,
        always_include_first=data.always_include_first,
        always_include_recent=data.always_include_recent,
        tile_offsets=data.tile_offsets,
        schedule_offsets=data.schedule_offsets,
        schedule_req_indices=data.schedule_req_indices,
        schedule_tile_indices=data.schedule_tile_indices,
    )

    def fn():
        topk_ids_list = [None] * data.queries_grouped.shape[0]
        for i in range(data.queries_grouped.shape[0]):
            if data.sparse_req_mask[i].item() == 1:
                req_topk = topk_indices[i]
                valid_mask = req_topk >= 0
                topk_ids_list[i] = req_topk[valid_mask] if valid_mask.any() else None
        return topk_ids_list

    return measure_ms(fn, warmup, iters)


def bench_cached_wrapper(data: BenchmarkInputs, warmup: int, iters: int) -> float:
    group_size = data.num_qo_heads // data.num_kv_heads

    def fn():
        q_grouped = data.queries_full.view(
            data.queries_full.shape[0], data.num_kv_heads, group_size, data.head_dim
        ).mean(dim=2)
        kv_indptr, kv_indices, topk_indices = fused_sparse_select_and_build(
            q_grouped,
            data.centroids,
            data.chunk_offsets,
            data.chunk_starts,
            data.chunk_ends,
            data.seq_lens,
            data.sparse_req_mask,
            data.req_pool_indices,
            data.req_to_token,
            num_kv_heads=data.num_kv_heads,
            head_dim=data.head_dim,
            top_k=data.top_k,
            scaling=data.scaling,
            always_include_first=data.always_include_first,
            always_include_recent=data.always_include_recent,
            tile_offsets=data.tile_offsets,
            schedule_offsets=data.schedule_offsets,
            schedule_req_indices=data.schedule_req_indices,
            schedule_tile_indices=data.schedule_tile_indices,
        )
        topk_ids_list = [None] * data.queries_grouped.shape[0]
        for i in range(data.queries_grouped.shape[0]):
            if data.sparse_req_mask[i].item() == 1:
                req_topk = topk_indices[i]
                valid_mask = req_topk >= 0
                topk_ids_list[i] = req_topk[valid_mask] if valid_mask.any() else None
        return kv_indptr, kv_indices, topk_ids_list

    return measure_ms(fn, warmup, iters)


def main():
    parser = argparse.ArgumentParser(description="Benchmark tree-sparse selector path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-chunks", type=int, default=686)
    parser.add_argument("--seq-len", type=int, default=9661)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--always-include-first", type=int, default=4)
    parser.add_argument("--always-include-recent", type=int, default=128)
    parser.add_argument("--max-ctx-len", type=int, default=262144)
    parser.add_argument("--scaling", type=float, default=0.088)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    data = create_inputs(args)

    print("=" * 72)
    print("Tree-Sparse Selector Microbenchmark")
    print("=" * 72)
    print(f"device={args.device} bs={args.bs} qo_heads={args.num_qo_heads} kv_heads={args.num_kv_heads}")
    print(f"head_dim={args.head_dim} num_chunks={args.num_chunks} seq_len={args.seq_len} top_k={args.top_k}")
    print(f"always_include_first={args.always_include_first} always_include_recent={args.always_include_recent}")
    print(f"warmup={args.warmup} iters={args.iters}")
    print()

    grouping_ms = bench_query_grouping(data, args.warmup, args.iters)
    kernel_ms = bench_fused_kernel(data, args.warmup, args.iters)
    post_ms = bench_postprocess(data, args.warmup, args.iters)
    cached_wrapper_ms = bench_cached_wrapper(data, args.warmup, args.iters)

    print(f"{'query_grouping':24s} {grouping_ms:8.3f} ms")
    print(f"{'fused_kernel_total':24s} {kernel_ms:8.3f} ms")
    print(f"{'topk_postprocess':24s} {post_ms:8.3f} ms")
    print(f"{'cached_wrapper_total':24s} {cached_wrapper_ms:8.3f} ms")
    print()
    print("Approximate composition:")
    print(f"  grouped query + kernel + postprocess ~= {grouping_ms + kernel_ms + post_ms:8.3f} ms")
    print(f"  wrapper measured total             ~= {cached_wrapper_ms:8.3f} ms")


if __name__ == "__main__":
    main()
