"""Data pipeline monitor — DataLoader workers, batch timing, dataset health."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from rf2_swarm import config as C

from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.data_pipeline")


class DataPipelineAgent(BaseAgent):
    """Monitors data loading health: workers, timing, cache, dataset sizes."""

    def __init__(self) -> None:
        super().__init__("data_pipeline", "DataLoader workers, batch timing, cache hits, dataset sizes")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        log_tail = datastore.get("log_tail", [])
        state = datastore.get("state", {})
        metrics = datastore.get("metrics", [])

        log_text = "\n".join(log_tail)

        # 1. DataLoader timeout or worker errors
        dl_errors = [l for l in log_tail if "DataLoader" in l and ("error" in l.lower() or "timeout" in l.lower())]
        if dl_errors:
            checks.append(CheckResult(
                name="dataloader_errors",
                verdict=Verdict.FAIL,
                summary=f"{len(dl_errors)} DataLoader errors in log",
                detail=f"Recent: {dl_errors[-3:]}",
            ))
        else:
            checks.append(CheckResult(name="dataloader_errors", verdict=Verdict.PASS,
                                       summary="No DataLoader errors"))

        # 2. Dataset size information
        ds_sizes = re.findall(r"(?:dataset|Dataset).*?(\d+)", log_text)
        if ds_sizes:
            size = int(ds_sizes[-1])
            if size > 0:
                checks.append(CheckResult(
                    name="dataset_size",
                    verdict=Verdict.PASS,
                    summary=f"Dataset size: {size} samples",
                    metric=float(size),
                    dimension="dataset_size",
                ))
            else:
                checks.append(CheckResult(name="dataset_size", verdict=Verdict.CRIT,
                                           summary=f"Dataset size is {size}",
                                           detail="Empty dataset = no training possible"))
        else:
            checks.append(CheckResult(name="dataset_size", verdict=Verdict.SKIP,
                                       summary="No dataset size info in log"))

        # 3. Batch timing (seconds per batch)
        batch_times = re.findall(r"(?:batch|step).*?(\d+\.?\d*)s", log_text, re.IGNORECASE)
        if batch_times:
            times = [float(t) for t in batch_times[-20:]]
            avg_time = sum(times) / len(times)
            if avg_time < 1.0:
                v = Verdict.PASS
            elif avg_time < 3.0:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="batch_timing",
                verdict=v,
                summary=f"Avg batch time: {avg_time:.2f}s over {len(times)} samples",
                detail=f"Slow batches = I/O or compute bottleneck",
                metric=avg_time,
                threshold=1.0,
                dimension="batch_time_avg",
            ))
        else:
            checks.append(CheckResult(name="batch_timing", verdict=Verdict.SKIP,
                                       summary="No batch timing info in log"))

        # 4. Subprocess log health
        subprocess_errors = state.get("subprocess_errors", 0)
        if subprocess_errors:
            checks.append(CheckResult(
                name="subprocess_health",
                verdict=Verdict.WARN if subprocess_errors < 5 else Verdict.FAIL,
                summary=f"{subprocess_errors} subprocess errors",
                detail="Subprocess errors can indicate dataloader crashes",
                metric=float(subprocess_errors),
                dimension="subprocess_errors",
            ))
        else:
            checks.append(CheckResult(name="subprocess_health", verdict=Verdict.PASS,
                                       summary="No subprocess errors"))

        # 5. Batch size consistency
        batch_sizes = re.findall(r"batch_size[=:]\s*(\d+)", log_text, re.IGNORECASE)
        if batch_sizes:
            bs = int(batch_sizes[-1])
            checks.append(CheckResult(
                name="batch_size",
                verdict=Verdict.PASS if bs >= 2 else Verdict.WARN,
                summary=f"Batch size: {bs}",
                metric=float(bs),
                dimension="batch_size",
            ))
        else:
            checks.append(CheckResult(name="batch_size", verdict=Verdict.SKIP,
                                       summary="No batch_size in log"))

        # 6. GPU memory allocated for data
        gpu = datastore.get("gpu", {})
        mem_used = gpu.get("mem_used_gb", -1)
        if mem_used > 0:
            data_mem_ratio = mem_used / C.GPU_TOTAL_MEM_GB
            if data_mem_ratio < 0.5:
                v = Verdict.PASS
            elif data_mem_ratio < 0.8:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="data_memory_usage",
                verdict=v,
                summary=f"GPU mem: {mem_used:.1f}/{C.GPU_TOTAL_MEM_GB} GB ({data_mem_ratio:.0%})",
                metric=mem_used,
                threshold=C.GPU_TOTAL_MEM_GB * 0.8,
                dimension="data_mem_usage",
            ))
        else:
            checks.append(CheckResult(name="data_memory_usage", verdict=Verdict.SKIP,
                                       summary="GPU data not available"))

        # 7. Image loading errors
        img_errors = [l for l in log_tail if "image" in l.lower() and ("error" in l.lower() or "fail" in l.lower())]
        if img_errors:
            checks.append(CheckResult(
                name="image_load_errors",
                verdict=Verdict.FAIL,
                summary=f"{len(img_errors)} image loading errors",
                detail=f"Recent: {img_errors[-3:]}",
            ))
        else:
            checks.append(CheckResult(name="image_load_errors", verdict=Verdict.PASS,
                                       summary="No image loading errors"))

        # 8. Cache directory size from ckpt
        ckpt = datastore.get("ckpt", {})
        total_files = ckpt.get("total_files", 0)
        total_size = ckpt.get("total_size_mb", 0)
        if total_files > 0:
            if total_size < C.GPU_TOTAL_MEM_GB * 512:
                v = Verdict.PASS
            elif total_size < C.GPU_TOTAL_MEM_GB * 1024:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="checkpoint_disk_usage",
                verdict=v,
                summary=f"Ckpt: {total_size:.0f} MB / {total_files} files",
                metric=total_size,
                dimension="ckpt_disk_mb",
            ))
        else:
            checks.append(CheckResult(name="checkpoint_disk_usage", verdict=Verdict.SKIP,
                                       summary="No checkpoint data"))

        # 9. Batch composition (augmentation info)
        aug_lines = [l for l in log_tail if "augment" in l.lower() or "mosaic" in l.lower()]
        checks.append(CheckResult(
            name="augmentation_active",
            verdict=Verdict.PASS if aug_lines else Verdict.WARN,
            summary=f"Augmentation log lines: {len(aug_lines)}",
            detail="Missing augmentation lines may mean no augmentation applied",
        ))

        # 10. Metrics.jsonl timestamps (data freshness)
        if metrics:
            latest_ts = metrics[0].get("timestamp") or metrics[0].get("time")
            if latest_ts:
                checks.append(CheckResult(
                    name="data_freshness",
                    verdict=Verdict.PASS,
                    summary=f"Latest metric: {str(latest_ts)[:19]}",
                    dimension="data_freshness",
                ))
            else:
                checks.append(CheckResult(name="data_freshness", verdict=Verdict.SKIP,
                                           summary="No timestamp in metrics"))
        else:
            checks.append(CheckResult(name="data_freshness", verdict=Verdict.WARN,
                                       summary="No metrics data available"))

        return AgentResult(agent_name=self.name, checks=checks)
