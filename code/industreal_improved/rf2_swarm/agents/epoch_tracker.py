"""Epoch tracker — progression rate, ETA, batch throughput."""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.epoch_tracker")


class EpochTrackerAgent(BaseAgent):
    """Tracks epoch progression: rate, ETA to completion, batch throughput."""

    def __init__(self) -> None:
        super().__init__("epoch_tracker", "Epoch progression rate, ETA, batch throughput")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        state = datastore.get("state", {})
        log_tail = datastore.get("log_tail", [])
        metrics = datastore.get("metrics", [])

        epoch = datastore.get("epoch", 0) or 0
        step = datastore.get("step", 0) or 0

        # Read full log for completed epoch tqdm lines (log_tail only has partial in-progress lines)
        full_log_lines = self._read_full_log_timing()

        # 1. Epoch progression
        max_epochs = int(state.get("max_epochs", C.DEFAULT_MAX_EPOCHS))
        if epoch > 0:
            pct = epoch / max_epochs * 100
            remaining = max_epochs - epoch
            if pct >= 100:
                v = Verdict.CRIT
            elif pct >= 90:
                v = Verdict.WARN
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="epoch_progress",
                verdict=v,
                summary=f"Epoch {epoch}/{max_epochs} ({pct:.0f}%, {remaining} remaining)",
                metric=float(epoch),
                threshold=float(max_epochs),
                dimension="epoch_progress",
            ))
        else:
            checks.append(CheckResult(name="epoch_progress", verdict=Verdict.WARN,
                                       summary="Epoch data not available"))

        # 2. Epoch timing from log (use full log for completed epochs, tail for current)
        epoch_times = self._parse_epoch_timing(full_log_lines or log_tail)
        if epoch_times:
            recent = epoch_times[-5:]
            avg_time = sum(recent) / len(recent)
            if avg_time < 5400:  # < 90 min = normal for ConvNeXt-Tiny + RTX 3060
                v = Verdict.PASS
            elif avg_time < 7200:  # < 2h = slow but acceptable
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="epoch_duration",
                verdict=v,
                summary=f"Avg epoch time: {avg_time / 60:.1f} min over last {len(recent)} epochs",
                detail=f"Recent: {[f'{t / 60:.1f}m' for t in recent]}",
                metric=avg_time,
                threshold=5400,
                dimension="epoch_time_sec",
            ))
        else:
            checks.append(CheckResult(name="epoch_duration", verdict=Verdict.SKIP,
                                       summary="Epoch timing not available"))

        # 3. ETA to completion
        if epoch_times and epoch > 0:
            avg_epoch_sec = sum(epoch_times[-5:]) / len(epoch_times[-5:]) if len(epoch_times) >= 5 else epoch_times[-1]
            remaining = max_epochs - epoch
            eta_sec = remaining * avg_epoch_sec
            eta_hours = eta_sec / 3600
            if eta_hours < 12:
                v = Verdict.PASS
            elif eta_hours < 48:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="training_eta",
                verdict=v,
                summary=f"ETA: ~{eta_hours:.1f}h ({remaining} epochs × {avg_epoch_sec / 60:.1f} min avg)",
                metric=eta_hours,
                threshold=12,
                dimension="eta_hours",
            ))
        else:
            checks.append(CheckResult(name="training_eta", verdict=Verdict.SKIP,
                                       summary="Cannot compute ETA"))

        # 4. Batch throughput (steps per second)
        if step > 0 and epoch_times:
            avg_epoch_sec = sum(epoch_times[-3:]) / len(epoch_times[-3:]) if len(epoch_times) >= 3 else epoch_times[-1]
            steps_per_epoch = 3302  # observed from training log at subset_ratio=0.50
            throughput = steps_per_epoch / avg_epoch_sec if avg_epoch_sec > 0 else 0
            if throughput > 2.0:
                v = Verdict.PASS
            elif throughput > 0.5:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="batch_throughput",
                verdict=v,
                summary=f"Throughput: {throughput:.2f} steps/s ({steps_per_epoch} steps/epoch)",
                metric=throughput,
                threshold=0.5,
                dimension="throughput_steps_per_sec",
            ))
        else:
            checks.append(CheckResult(name="batch_throughput", verdict=Verdict.SKIP,
                                       summary="Cannot compute throughput"))

        # 5. Step consistency (no stalls or jumps)
        if metrics and len(metrics) >= 5:
            steps = []
            for m in metrics[:10]:
                s = m.get("step") or m.get("global_step")
                if s is not None:
                    steps.append(int(s))
            if len(steps) >= 5:
                diffs = [steps[i] - steps[i + 1] for i in range(len(steps) - 1)]
                avg_diff = sum(diffs) / len(diffs)
                max_diff = max(diffs)
                if max_diff > avg_diff * 3:
                    v = Verdict.WARN
                else:
                    v = Verdict.PASS
                checks.append(CheckResult(
                    name="step_consistency",
                    verdict=v,
                    summary=f"Step diffs: avg={avg_diff:.0f}, max={max_diff}, (len={len(steps)})",
                    detail="Large step gaps = possible stalls or skips",
                    metric=float(max_diff),
                    threshold=avg_diff * 3,
                    dimension="step_consistency",
                ))
            else:
                checks.append(CheckResult(name="step_consistency", verdict=Verdict.SKIP,
                                           summary="Not enough steps"))
        else:
            checks.append(CheckResult(name="step_consistency", verdict=Verdict.SKIP,
                                       summary="Not enough metrics"))

        return AgentResult(agent_name=self.name, checks=checks)

    def _read_full_log_timing(self) -> List[str] | None:
        """Read the full train.log to extract completed epoch tqdm lines.

        Tqdm lines are written with \r and only the final output per epoch
        survives in the file. log_tail (500 lines) may only have partial
        in-progress epoch lines — the full log captures all completed epochs.
        """
        try:
            if not os.path.isfile(C.TRAIN_LOG):
                return None
            lines = []
            with open(C.TRAIN_LOG, "r") as f:
                for line in f:
                    line = line.rstrip("\n").strip("\r")
                    if line:
                        lines.append(line)
            return lines
        except (OSError, IOError) as e:
            logger.warning("Could not read train.log for timing: %s", e)
            return None

    def _parse_epoch_timing(self, lines: List[str]) -> List[float]:
        """Parse epoch elapsed time from tqdm output: ``[MM:SS<...`` or ``[H:MM:SS<...``.

        Tqdm lines look like:
          Epoch 17/30: 100%|████| 3302/3302 [47:51<00:00, 1.45s/it]

        Returns epoch duration in seconds (e.g., 47:51 → 2871.0).
        """
        times = []
        for line in lines:
            # Match [MM:SS< or [H:MM:SS< — the elapsed time before <
            m = re.search(r"\[(\d+):(\d+)(?::(\d+))?\s*<", line)
            if m:
                try:
                    h = int(m.group(1)) if m.group(3) else 0
                    m_val = int(m.group(2)) if m.group(3) else int(m.group(1))
                    s = int(m.group(3)) if m.group(3) else int(m.group(2))
                    total_sec = h * 3600 + m_val * 60 + s
                    if total_sec > 0:
                        times.append(float(total_sec))
                except ValueError:
                    pass
        return times
