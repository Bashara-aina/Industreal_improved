"""Checkpoint monitor — file age, sizes, disk usage, corruption check."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.checkpoint")


class CheckpointAgent(BaseAgent):
    """Monitors checkpoint health: freshness, size, disk, corruption risk."""

    def __init__(self) -> None:
        super().__init__("checkpoint", "File age, sizes, disk usage, corruption check, cleanup")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        ckpt_info = datastore.get("ckpt", {})
        ckpt_files = ckpt_info.get("files", [])
        now = time.time()

        # 1. Latest checkpoint freshness
        if ckpt_files:
            latest = ckpt_files[0]  # sorted by mtime desc
            age_hours = (now - latest["mtime"]) / 3600
            if age_hours < 1:
                v = Verdict.PASS
            elif age_hours < 6:
                v = Verdict.WARN
            elif age_hours < 24:
                v = Verdict.FAIL
            else:
                v = Verdict.CRIT
            checks.append(CheckResult(
                name="latest_ckpt_age",
                verdict=v,
                summary=f"Latest ckpt: {latest['name']}, {age_hours:.1f}h old",
                detail=f"mtime: {time.strftime('%Y-%m-%d %H:%M', time.localtime(latest['mtime']))}",
                metric=age_hours,
                threshold=1.0,
                dimension="ckpt_age_hours",
            ))
        else:
            checks.append(CheckResult(name="latest_ckpt_age", verdict=Verdict.WARN,
                                       summary="No checkpoint files found",
                                       detail="Checkpoints may not be saving"))

        # 2. Total disk usage
        total_size = ckpt_info.get("total_size_mb", 0)
        if total_size > 0:
            if total_size < 5000:  # 5 GB
                v = Verdict.PASS
            elif total_size < 20000:  # 20 GB
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="ckpt_disk_usage",
                verdict=v,
                summary=f"Total: {total_size:.0f} MB / {ckpt_info.get('total_files', 0)} files",
                metric=total_size,
                threshold=5000,
                dimension="ckpt_total_mb",
            ))
        else:
            checks.append(CheckResult(name="ckpt_disk_usage", verdict=Verdict.SKIP,
                                       summary="No checkpoint data"))

        # 3. Stale checkpoint cleanup needed
        if len(ckpt_files) > 10:
            oldest = ckpt_files[-1]
            oldest_age = (now - oldest["mtime"]) / 3600
            if oldest_age > 48 and total_size > 10000:
                v = Verdict.WARN
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="ckpt_cleanup_needed",
                verdict=v,
                summary=f"{len(ckpt_files)} checkpoints, oldest {oldest_age:.0f}h old",
                detail=f"Oldest: {oldest['name']}",
                metric=float(len(ckpt_files)),
                threshold=10,
                dimension="ckpt_count",
            ))
        else:
            checks.append(CheckResult(name="ckpt_cleanup_needed", verdict=Verdict.PASS,
                                       summary=f"{len(ckpt_files)} checkpoints, no cleanup needed"))

        # 4. Interrupted save detection (partial files)
        partial = [f for f in ckpt_files if f["name"].endswith(".tmp") or f["name"].endswith(".part")]
        if partial:
            checks.append(CheckResult(
                name="partial_checkpoints",
                verdict=Verdict.WARN,
                summary=f"{len(partial)} partial/tmp checkpoint files",
                detail="May indicate interrupted saves",
            ))
        else:
            checks.append(CheckResult(name="partial_checkpoints", verdict=Verdict.PASS,
                                       summary="No partial checkpoint files"))

        # 5. Checkpoint size consistency
        if len(ckpt_files) >= 3:
            sizes = [f["size_mb"] for f in ckpt_files[:5]]
            avg_size = sum(sizes) / len(sizes)
            outliers = [s for s in sizes if s < avg_size * 0.5 or s > avg_size * 2]
            if outliers:
                checks.append(CheckResult(
                    name="ckpt_size_consistency",
                    verdict=Verdict.WARN,
                    summary=f"Size outliers: {[f'{s:.1f}' for s in outliers]} MB, avg={avg_size:.0f} MB",
                    detail="Inconsistent sizes = possible corruption or config change",
                    metric=avg_size,
                    dimension="ckpt_size_consistency",
                ))
            else:
                checks.append(CheckResult(name="ckpt_size_consistency", verdict=Verdict.PASS,
                                           summary=f"Ckpt sizes consistent, avg={avg_size:.0f} MB"))
        else:
            checks.append(CheckResult(name="ckpt_size_consistency", verdict=Verdict.SKIP,
                                       summary=f"Only {len(ckpt_files)} checkpoints"))

        # 6. Free disk space
        try:
            stat = os.statvfs(C.CKPT_DIR)
            free_gb = stat.f_frsize * stat.f_bavail / (1024 ** 3)
            total_gb = stat.f_frsize * stat.f_blocks / (1024 ** 3)
            free_ratio = free_gb / total_gb if total_gb > 0 else 0
            if free_gb < 10:
                v = Verdict.CRIT
            elif free_gb < 50:
                v = Verdict.FAIL
            elif free_ratio < 0.1:
                v = Verdict.WARN
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="free_disk_space",
                verdict=v,
                summary=f"Free: {free_gb:.1f} / {total_gb:.0f} GB ({free_ratio:.0%})",
                metric=free_gb,
                threshold=10.0,
                dimension="free_disk_gb",
            ))
        except OSError:
            checks.append(CheckResult(name="free_disk_space", verdict=Verdict.SKIP,
                                       summary="Could not get disk space"))

        # 7. Best checkpoint weight saved
        state = datastore.get("state", {})
        best_epoch = state.get("best_epoch")
        if best_epoch is not None:
            best_ckpt = [f for f in ckpt_files if str(best_epoch) in f["name"]]
            if best_ckpt:
                checks.append(CheckResult(
                    name="best_ckpt_saved",
                    verdict=Verdict.PASS,
                    summary=f"Best epoch {best_epoch} checkpoint saved: {best_ckpt[0]['name']}",
                    dimension="best_ckpt",
                ))
            else:
                checks.append(CheckResult(name="best_ckpt_saved", verdict=Verdict.WARN,
                                           summary=f"Best epoch {best_epoch} checkpoint not found",
                                           detail="Best model may not be persisted"))
        else:
            checks.append(CheckResult(name="best_ckpt_saved", verdict=Verdict.SKIP,
                                       summary="best_epoch not in state"))

        # 8. Resume file exists
        resume_file = os.path.join(C.CKPT_DIR, "last_checkpoint")
        if os.path.isfile(resume_file):
            checks.append(CheckResult(name="resume_file", verdict=Verdict.PASS,
                                       summary="Resume checkpoint exists"))
        else:
            checks.append(CheckResult(name="resume_file", verdict=Verdict.SKIP,
                                       summary="No resume file (ok for fresh training)"))

        # 9. Epoch frequency estimation
        # Only best.pth is saved (on val improvement), so file-mtime intervals
        # are NOT epoch intervals — they reflect how often val improves.
        # Use datastore epoch info as ground truth; skip if training is active.
        epoch = datastore.get("epoch", 0) or 0
        step = datastore.get("step", 0) or 0
        pid_alive = datastore.get("pid_alive", False)
        training_active = pid_alive and (epoch > 0 or step > 0)
        if training_active and len(ckpt_files) >= 3:
            # Training is running — file intervals only reflect val improvement cadence
            times = [f["mtime"] for f in ckpt_files[:5]]
            if len(times) >= 2:
                intervals = [times[i] - times[i + 1] for i in range(min(len(times) - 1, 4))]
                avg_interval = sum(intervals) / len(intervals)
                interval_min = avg_interval / 60
                # With best-only saves, long intervals are expected
                v = Verdict.WARN if interval_min > 240 else Verdict.PASS
                checks.append(CheckResult(
                    name="ckpt_save_interval",
                    verdict=v,
                    summary=f"Avg save interval: {interval_min:.0f} min (best-only ckpt strategy)",
                    detail="Long interval = no new best model found, not a stall",
                    metric=interval_min,
                    threshold=30,
                    dimension="ckpt_interval_min",
                ))
            else:
                checks.append(CheckResult(name="ckpt_save_interval", verdict=Verdict.SKIP,
                                           summary="Not enough checkpoints"))
        elif len(ckpt_files) >= 3:
            # Not training but has old checkpoints — may be stale
            v = Verdict.FAIL
            checks.append(CheckResult(
                name="ckpt_save_interval",
                verdict=v,
                summary="Training not active — checkpoints are stale",
                dimension="ckpt_interval_min",
            ))
        else:
            checks.append(CheckResult(name="ckpt_save_interval", verdict=Verdict.SKIP,
                                       summary="Not enough checkpoints"))

        return AgentResult(agent_name=self.name, checks=checks)
