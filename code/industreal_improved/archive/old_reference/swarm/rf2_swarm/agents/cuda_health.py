"""CUDA health monitor — CUDA errors, OOM, NCCL, GPU visibility."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.cuda_health")


class CudaHealthAgent(BaseAgent):
    """Monitors CUDA subsystem: errors, OOM events, NCCL failures, GPU visibility."""

    def __init__(self) -> None:
        super().__init__("cuda_health", "CUDA errors, OOM events, NCCL failures, GPU visibility")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        log_tail = datastore.get("log_tail", [])
        gpu = datastore.get("gpu", {})

        log_text = "\n".join(log_tail)

        # 1. CUDA errors
        cuda_errors = re.findall(r"CUDA\s+Error|RuntimeError.*CUDA|cuda\s+error", log_text, re.IGNORECASE)
        if cuda_errors:
            checks.append(CheckResult(
                name="cuda_errors",
                verdict=Verdict.CRIT,
                summary=f"{len(cuda_errors)} CUDA errors in log",
                detail=f"Recent: {cuda_errors[-3:]}",
                metric=float(len(cuda_errors)),
                threshold=0,
                dimension="cuda_errors",
            ))
        else:
            checks.append(CheckResult(name="cuda_errors", verdict=Verdict.PASS,
                                       summary="No CUDA errors"))

        # 2. OOM events
        oom_events = re.findall(r"out of memory|OOM|CUDA_OUT_OF_MEMORY", log_text, re.IGNORECASE)
        if oom_events:
            checks.append(CheckResult(
                name="oom_events",
                verdict=Verdict.CRIT,
                summary=f"{len(oom_events)} OOM events",
                detail=f"OOM events: {oom_events}",
                metric=float(len(oom_events)),
                threshold=0,
                dimension="oom_events",
            ))
        else:
            checks.append(CheckResult(name="oom_events", verdict=Verdict.PASS,
                                       summary="No OOM events"))

        # 3. NCCL errors (distributed training)
        nccl_errors = re.findall(r"NCCL\s+Error|nccl\s+error|nccl.*fail", log_text, re.IGNORECASE)
        if nccl_errors:
            checks.append(CheckResult(
                name="nccl_errors",
                verdict=Verdict.FAIL,
                summary=f"{len(nccl_errors)} NCCL errors",
                detail=f"NCCL errors: {nccl_errors[-3:]}",
                metric=float(len(nccl_errors)),
                dimension="nccl_errors",
            ))
        else:
            checks.append(CheckResult(name="nccl_errors", verdict=Verdict.PASS,
                                       summary="No NCCL errors (single-GPU OK)"))

        # 4. GPU visibility (nvidia-smi working)
        gpu_error = gpu.get("error")
        if gpu_error:
            checks.append(CheckResult(
                name="gpu_visibility",
                verdict=Verdict.FAIL,
                summary=f"nvidia-smi failed: {gpu_error}",
                detail="GPU monitoring unavailable — check drivers",
            ))
        else:
            util = gpu.get("util_pct", -1)
            if util >= 0:
                checks.append(CheckResult(
                    name="gpu_visibility",
                    verdict=Verdict.PASS,
                    summary=f"GPU visible: util={util:.0f}%, mem={gpu.get('mem_used_gb', '?')} GB",
                ))
            else:
                checks.append(CheckResult(name="gpu_visibility", verdict=Verdict.WARN,
                                           summary="GPU visible but no util data"))

        # 5. CUDA version / compatibility
        cuda_version_lines = re.findall(r"CUDA\s+(?:Version|version|v)\s*[=:]\s*([\d.]+)", log_text)
        if cuda_version_lines:
            cuda_ver = cuda_version_lines[-1]
            major = int(cuda_ver.split(".")[0]) if "." in cuda_ver else 0
            if major >= 11:
                v = Verdict.PASS
            elif major >= 10:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="cuda_version",
                verdict=v,
                summary=f"CUDA version: {cuda_ver}",
                dimension="cuda_version",
            ))
        else:
            checks.append(CheckResult(name="cuda_version", verdict=Verdict.SKIP,
                                       summary="CUDA version not in log"))

        # 6. CUDNN status
        cudnn_lines = [l for l in log_tail if "cuDNN" in l or "CUDNN" in l]
        if cudnn_lines:
            checks.append(CheckResult(
                name="cudnn_status",
                verdict=Verdict.PASS,
                summary=f"cuDNN: {cudnn_lines[-1][:80]}",
                dimension="cudnn_status",
            ))
        else:
            checks.append(CheckResult(name="cudnn_status", verdict=Verdict.SKIP,
                                       summary="cuDNN not mentioned in log"))

        return AgentResult(agent_name=self.name, checks=checks)
