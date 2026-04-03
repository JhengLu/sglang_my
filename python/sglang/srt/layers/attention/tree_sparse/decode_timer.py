"""
CUDA event-based decode step profiler for tree-sparse attention.

Enable: TREE_SPARSE_TIMING=1
Log interval: TREE_SPARSE_TIMING_INTERVAL=10 (default)

Records fine-grained timing within each decode step using CUDA event pairs.
Reports aggregated timing breakdown periodically to the logger.
"""

import logging
import os
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


class DecodeStepTimer:
    """
    Records CUDA event pairs (start/stop) for named intervals during decode.
    Reports aggregated timing breakdown periodically.

    Usage:
        timer.start("L0:kv_cache_save")
        # ... operation ...
        timer.stop("L0:kv_cache_save")
        # ... more operations ...
        timer.finish_step()  # sync + aggregate + maybe report
    """

    def __init__(self, num_layers: int = 28):
        self.enabled = os.environ.get("TREE_SPARSE_TIMING", "0") == "1"
        self.num_layers = num_layers
        self.log_interval = int(
            os.environ.get("TREE_SPARSE_TIMING_INTERVAL", "10")
        )
        self._step_count = 0

        # Current step state
        self._pending_starts: Dict[str, torch.cuda.Event] = {}
        self._completed: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

        # Wall-clock total (avoids cross-stream CUDA event issues)
        self._total_wall_start: float = 0.0
        self._total_wall_end: float = 0.0

        # History for averaging
        self._history: Dict[str, List[float]] = defaultdict(list)

        if self.enabled:
            logger.info(
                f"[DecodeTimer] ENABLED (TREE_SPARSE_TIMING=1, "
                f"report every {self.log_interval} steps)"
            )

    def start_total(self):
        """Record wall-clock start for the total decode step.

        Uses time.perf_counter instead of CUDA events because the total
        timer (in model_runner) and per-layer timers (in attention backend)
        may record on different CUDA streams, making cross-stream
        elapsed_time() unreliable.  Wall-clock + synchronize in
        finish_step() gives accurate results.
        """
        if not self.enabled:
            return
        self._total_wall_start = time.perf_counter()

    def stop_total(self):
        """Snapshot wall-clock end for model forward without calling finish_step.

        Call this at the end of model_runner.forward_decode() so the total
        captures only model forward time.  finish_step() is called later
        from the scheduler after scheduler-level events are recorded.
        """
        if not self.enabled or self._total_wall_start <= 0:
            return
        self._total_wall_end = time.perf_counter()

    def start(self, name: str):
        """Record the start of a named interval."""
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._pending_starts[name] = ev

    def stop(self, name: str):
        """Record the end of a named interval."""
        if not self.enabled:
            return
        start_ev = self._pending_starts.pop(name, None)
        if start_ev is None:
            return
        end_ev = torch.cuda.Event(enable_timing=True)
        end_ev.record()
        self._completed.append((name, start_ev, end_ev))

    def finish_step(self):
        """Synchronize GPU, compute elapsed times, optionally report."""
        if not self.enabled or not self._completed:
            self._pending_starts.clear()
            self._completed.clear()
            return

        self._step_count += 1
        sync_start = time.perf_counter()
        torch.cuda.synchronize()
        sync_ms = (time.perf_counter() - sync_start) * 1000
        self._history["sched:sync"].append(sync_ms)

        # Wall-clock total for model forward
        if self._total_wall_start > 0:
            end = self._total_wall_end if self._total_wall_end > 0 else time.perf_counter()
            total_ms = (end - self._total_wall_start) * 1000
            self._history["total"].append(total_ms)
            self._total_wall_start = 0.0
            self._total_wall_end = 0.0

        for name, start_ev, end_ev in self._completed:
            elapsed = start_ev.elapsed_time(end_ev)  # milliseconds
            self._history[name].append(elapsed)

        self._completed.clear()
        self._pending_starts.clear()

        if self._step_count % self.log_interval == 0:
            self._report()

    def _avg(self, name: str, n: int) -> float:
        vals = self._history.get(name, [])
        recent = vals[-n:]
        return sum(recent) / len(recent) if recent else 0.0

    # Fine-grained op categories — every op must be listed explicitly.
    # If a new op appears that is NOT listed here, it shows as "UNCLASSIFIED"
    # so you can immediately see it and decide where it belongs.
    ATTN_PROJ_OPS = {"input_ln", "qkv_proj", "o_proj"}
    KV_CACHE_OPS = {"kv_cache_save", "kv_cache_load"}
    ATTENTION_OPS = {"attention", "begin_forward"}
    SPARSE_SELECTION_OPS = {
        "sparse_get_centroids", "sparse_score_topk",
        "sparse_build_mask", "sparse_extract_indices",
        "sparse_batch_meta", "sparse_build_indices",
        "centroid_update",
    }
    MLP_BLOCK_OPS = {"post_attn_ln", "mlp"}

    # Map: category_key -> (display_name, op_set)
    OP_CATEGORIES = [
        ("attn_proj", "Attention projections", ATTN_PROJ_OPS),
        ("kv_cache", "KV cache ops", KV_CACHE_OPS),
        ("attention", "Attention compute", ATTENTION_OPS),
        ("sparse", "Sparse selection", SPARSE_SELECTION_OPS),
        ("mlp", "MLP block", MLP_BLOCK_OPS),
    ]

    def _classify_op(self, op: str) -> str:
        for key, _, op_set in self.OP_CATEGORIES:
            if op in op_set:
                return key
        return "UNCLASSIFIED"

    def _group_ops_by_category(self, ops: OrderedDict) -> OrderedDict:
        """Group ops into named categories. Returns {cat_key: (display_name, [(op, time)])}."""
        grouped = OrderedDict()
        for key, display_name, _ in self.OP_CATEGORIES:
            grouped[key] = (display_name, [])
        grouped["UNCLASSIFIED"] = ("UNCLASSIFIED", [])

        for op, t in ops.items():
            cat = self._classify_op(op)
            grouped[cat][1].append((op, t))

        # Remove empty categories
        return OrderedDict((k, v) for k, v in grouped.items() if v[1])

    def _report(self):
        n = min(self.log_interval, self._step_count)
        if n == 0:
            return

        avgs: Dict[str, float] = {}
        for name in self._history:
            avgs[name] = self._avg(name, n)

        init_time = avgs.get("init_metadata", 0.0)
        total_time = avgs.get("total", 0.0)

        # Group per-layer ops: "L{id}:{op}" -> {layer_id: {op: time}}
        layer_ops: Dict[int, OrderedDict] = defaultdict(OrderedDict)
        # Scheduler-level ops: "sched:{op}" -> {op: time}
        sched_ops: OrderedDict = OrderedDict()

        for name, avg in avgs.items():
            if name in ("init_metadata", "total"):
                continue
            if name.startswith("L") and ":" in name:
                try:
                    layer_str, op = name.split(":", 1)
                    layer_id = int(layer_str[1:])
                    layer_ops[layer_id][op] = avg
                except (ValueError, IndexError):
                    pass
            elif name.startswith("sched:"):
                op = name.split(":", 1)[1]
                sched_ops[op] = avg

        # Compute totals
        timed_total = init_time
        for lid in layer_ops:
            timed_total += sum(layer_ops[lid].values())

        sched_total = sum(sched_ops.values())

        untimed = max(total_time - timed_total, 0.0)

        # Build report
        W = 78
        lines = ["", "=" * W]
        lines.append(
            f"  DECODE STEP TIMING (avg over {n} steps, "
            f"{self.num_layers} layers)"
        )
        lines.append("=" * W)

        if init_time > 0:
            lines.append(f"  {'init_forward_metadata':48s} {init_time:8.3f} ms")
            lines.append("")

        # Accumulate per-category totals across all layers
        # {cat_key: total_ms}
        all_cat_totals: Dict[str, float] = defaultdict(float)

        # Layer 0 detail
        if 0 in layer_ops:
            l0 = layer_ops[0]
            l0_total = sum(l0.values())
            has_sparse = any("sparse" in op for op in l0)
            tag = " (sparse)" if has_sparse else ""
            lines.append(f"  Layer 0{tag}:")

            grouped = self._group_ops_by_category(l0)
            for cat_key, (display_name, op_list) in grouped.items():
                cat_total = sum(t for _, t in op_list)
                all_cat_totals[cat_key] += cat_total
                lines.append(f"    [{display_name}]  {cat_total:40.3f} ms")
                for op, t in op_list:
                    lines.append(f"      {op:44s} {t:8.3f} ms")
            lines.append(f"    {'── layer total':46s} {l0_total:8.3f} ms")

        # Layers 1+ aggregated
        other_ids = sorted(lid for lid in layer_ops if lid > 0)
        if other_ids:
            num = len(other_ids)
            first_id, last_id = other_ids[0], other_ids[-1]

            # Sum ops across other layers
            agg_ops: OrderedDict = OrderedDict()
            for lid in other_ids:
                for op, t in layer_ops[lid].items():
                    agg_ops[op] = agg_ops.get(op, 0.0) + t

            lines.append(
                f"\n  Layers {first_id}-{last_id} (avg per layer x {num}):"
            )
            grouped = self._group_ops_by_category(agg_ops)
            for cat_key, (display_name, op_list) in grouped.items():
                cat_total = sum(t for _, t in op_list)
                all_cat_totals[cat_key] += cat_total
                avg_cat = cat_total / num
                lines.append(
                    f"    [{display_name}]  "
                    f"{avg_cat:7.3f} x {num:2d} = {cat_total:8.3f} ms"
                )
                for op, total_t in op_list:
                    avg_t = total_t / num
                    lines.append(
                        f"      {op:28s} {avg_t:7.3f} x {num:2d} = "
                        f"{total_t:8.3f} ms"
                    )

        # Scheduler ops section (exclude summary-only entries)
        _sched_summary_keys = {"sync", "loop_wall"}
        sched_detail = {k: v for k, v in sched_ops.items()
                        if k not in _sched_summary_keys}
        if sched_detail:
            lines.append(f"\n  Scheduler:")
            for op, t in sched_detail.items():
                lines.append(f"    {op:46s} {t:8.3f} ms")

        # Summary
        lines.append(f"\n  {'─' * 58}")

        # Scheduler timing components
        run_batch_time = sched_ops.get("run_batch", 0.0)
        process_result_time = sched_ops.get("process_result", 0.0)
        sampling_time = sched_ops.get("sampling", 0.0)
        sync_time = sched_ops.get("sync", 0.0)
        loop_wall_time = sched_ops.get("loop_wall", 0.0)

        # Per-category summary lines
        for cat_key, display_name, _ in self.OP_CATEGORIES:
            cat_ms = all_cat_totals.get(cat_key, 0.0)
            if cat_ms > 0:
                pct = 100 * cat_ms / total_time if total_time > 0 else 0.0
                label = f"{display_name} (all layers)"
                lines.append(
                    f"  {label:42s} {cat_ms:8.3f} ms ({pct:.1f}%)"
                )
        # Show UNCLASSIFIED ops prominently so they're easy to spot
        unclassified_ms = all_cat_totals.get("UNCLASSIFIED", 0.0)
        if unclassified_ms > 0:
            pct = 100 * unclassified_ms / total_time if total_time > 0 else 0.0
            lines.append(
                f"  {'*** UNCLASSIFIED (all layers)':42s} "
                f"{unclassified_ms:8.3f} ms ({pct:.1f}%)"
            )
        if init_time > 0:
            pct_i = 100 * init_time / total_time if total_time > 0 else 0.0
            lines.append(
                f"  {'init_forward_metadata':42s} "
                f"{init_time:8.3f} ms ({pct_i:.1f}%)"
            )
        pct_u = 100 * untimed / total_time if total_time > 0 else 0.0
        lines.append(
            f"  {'Untimed (embed/logits/overhead)':42s} "
            f"{untimed:8.3f} ms ({pct_u:.1f}%)"
        )
        lines.append(
            f"  {'TOTAL DECODE STEP (model forward)':42s} {total_time:8.3f} ms"
        )
        if sched_ops:
            lines.append(
                f"  {'+ sampling':42s} {sampling_time:8.3f} ms"
            )
            lines.append(
                f"  {'sched:run_batch + sync':42s} "
                f"{run_batch_time + sync_time:8.3f} ms"
            )
            if loop_wall_time > 0:
                sched_overhead = max(
                    loop_wall_time - run_batch_time - sync_time, 0.0
                )
                lines.append(
                    f"  {'+ scheduler overhead':42s} "
                    f"{sched_overhead:8.3f} ms"
                )
                lines.append(
                    f"    {'(recv_reqs, get_batch, process_prev_result, etc.)':s}"
                )
                lines.append(
                    f"  {'FULL LOOP ITERATION (wall-clock ≈ ITL)':42s} "
                    f"{loop_wall_time:8.3f} ms"
                )
        lines.append("=" * W)

        logger.info("\n".join(lines))
