"""GPU resource monitor — VRAM, util%, temperature, power, ECC."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.gpu_resource")


class GPUResourceAgent(BaseAgent):
    """Monitors GPU health: VRAM usage, util, temperature, power, ECC errors."""

    def __init__(self) -> None:
        super().__init__("gpu_resource", "VRAM usage, util %, temperature, power, ECC errors")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        gpu = datastore.get("gpu", {})
        gpu_error = gpu.get("error")

        if gpu_error:
            checks.append(CheckResult(
                name="gpu_query",
                verdict=Verdict.FAIL,
                summary=f"nvidia-smi failed: {gpu_error}",
                detail="GPU monitoring unavailable",
            ))
            return AgentResult(agent_name=self.name, checks=checks)

        # 1. VRAM usage
        mem_used = gpu.get("mem_used_gb", -1)
        mem_total = gpu.get("mem_total_gb", C.GPU_TOTAL_MEM_GB)
        vram_frac = mem_used / mem_total if mem_total > 0 else 0
        if mem_used >= 0:
            if vram_frac >= C.VRAM_FAIL_FRACTION:
                v = Verdict.CRIT
            elif vram_frac >= C.VRAM_WARN_FRACTION:
                v = Verdict.FAIL
            elif vram_frac >= 0.7:
                v = Verdict.WARN
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="vram_usage",
                verdict=v,
                summary=f"VRAM: {mem_used:.1f}/{mem_total:.0f} GB ({vram_frac:.0%})",
                detail=f"WARN at {C.VRAM_WARN_FRACTION:.0%}, FAIL at {C.VRAM_FAIL_FRACTION:.0%}",
                metric=mem_used,
                threshold=mem_total * C.VRAM_WARN_FRACTION,
                dimension="vram_used_gb",
            ))
        else:
            checks.append(CheckResult(name="vram_usage", verdict=Verdict.SKIP,
                                       summary="VRAM data not available"))

        # 2. GPU utilization (training-context aware)
        # nvidia-smi snapshot is instantaneous — can land between training steps
        # showing 0% even when training is actively running at 100% during steps.
        util = gpu.get("util_pct", -1)
        pid_alive = datastore.get("pid_alive", False)
        epoch = datastore.get("epoch", 0) or 0
        step = datastore.get("step", 0) or 0
        training_active = pid_alive and (epoch > 0 or step > 0)
        if util >= 0:
            if util > 90:
                v = Verdict.PASS
            elif util > 50:
                v = Verdict.WARN
            elif training_active and util < 10:
                # Training is running but snapshot caught it between steps — sampling artifact
                v = Verdict.WARN
            elif util > 10:
                v = Verdict.FAIL
            else:
                v = Verdict.CRIT
            checks.append(CheckResult(
                name="gpu_utilization",
                verdict=v,
                summary=f"GPU util: {util:.0f}%",
                detail="Low utilization = compute bottleneck or small batch"
                if not (training_active and util < 10)
                else "0% is sampling artifact (read between training steps)",
                metric=util,
                threshold=50,
                dimension="gpu_util_pct",
            ))
        else:
            checks.append(CheckResult(name="gpu_utilization", verdict=Verdict.SKIP,
                                       summary="Util data not available"))

        # 3. Temperature
        temp = gpu.get("temp_c", -1)
        if temp >= 0:
            if temp < 70:
                v = Verdict.PASS
            elif temp < 85:
                v = Verdict.WARN
            else:
                v = Verdict.CRIT
            checks.append(CheckResult(
                name="gpu_temperature",
                verdict=v,
                summary=f"GPU temp: {temp:.0f}°C",
                detail="High temp = thermal throttling risk",
                metric=temp,
                threshold=85,
                dimension="gpu_temp_c",
            ))
        else:
            checks.append(CheckResult(name="gpu_temperature", verdict=Verdict.SKIP,
                                       summary="Temp data not available"))

        # 4. Power draw
        power = gpu.get("power_w", -1)
        if power >= 0:
            max_power = 170.0  # RTX 3060 typical TDP
            power_frac = power / max_power
            if power_frac > 0.5:
                v = Verdict.PASS
            elif power_frac > 0.3:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="gpu_power",
                verdict=v,
                summary=f"Power: {power:.0f}/{max_power:.0f} W ({power_frac:.0%})",
                detail="Low power = GPU underutilized",
                metric=power,
                threshold=max_power * 0.3,
                dimension="gpu_power_w",
            ))
        else:
            checks.append(CheckResult(name="gpu_power", verdict=Verdict.SKIP,
                                       summary="Power data not available"))

        # 5. Available RAM
        ram_gb = datastore.get("ram_gb", -1)
        if ram_gb >= 0:
            if ram_gb > 16:
                v = Verdict.PASS
            elif ram_gb > 8:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="system_ram",
                verdict=v,
                summary=f"Available RAM: {ram_gb:.1f} GB",
                metric=ram_gb,
                threshold=8.0,
                dimension="ram_available_gb",
            ))
        else:
            checks.append(CheckResult(name="system_ram", verdict=Verdict.SKIP,
                                       summary="RAM data not available"))

        # 6. ECC errors (from nvidia-smi if available)
        ecc_errors = gpu.get("ecc_errors", 0)
        if ecc_errors:
            checks.append(CheckResult(
                name="ecc_errors",
                verdict=Verdict.WARN if ecc_errors < 100 else Verdict.FAIL,
                summary=f"ECC errors: {ecc_errors}",
                metric=float(ecc_errors),
                dimension="ecc_errors",
            ))
        else:
            checks.append(CheckResult(name="ecc_errors", verdict=Verdict.PASS,
                                       summary="No ECC errors detected"))

        # 7. VRAM allocation vs batch size
        state = datastore.get("state", {})
        batch_size = state.get("batch_size", 2)
        expected_vram = batch_size * 1.5  # rough estimate per sample
        if mem_used > 0 and mem_used < expected_vram * 0.5:
            checks.append(CheckResult(
                name="vram_vs_batch",
                verdict=Verdict.WARN,
                summary=f"VRAM {mem_used:.1f}GB low for batch_size={batch_size}",
                detail="Possible small batch or model not fully loaded",
                dimension="vram_vs_batch",
            ))
        else:
            checks.append(CheckResult(name="vram_vs_batch", verdict=Verdict.PASS,
                                       summary=f"VRAM {mem_used:.1f}GB appropriate for batch_size={batch_size}"))

        # 8. Power cap throttling
        power_limit = gpu.get("power_limit_w", -1)
        if power_limit > 0 and power > 0:
            if power >= power_limit * 0.9:
                checks.append(CheckResult(
                    name="power_throttling",
                    verdict=Verdict.WARN,
                    summary=f"Power ({power:.0f}W) near limit ({power_limit:.0f}W)",
                    detail="GPU may be throttling",
                    dimension="power_throttling",
                ))
            else:
                checks.append(CheckResult(name="power_throttling", verdict=Verdict.PASS,
                                           summary=f"Power ({power:.0f}W) below limit ({power_limit:.0f}W)"))

        # 9. GPU clock info from metrics
        clock_gpu = gpu.get("clock_gpu_mhz", -1)
        clock_mem = gpu.get("clock_mem_mhz", -1)
        if clock_gpu > 0:
            checks.append(CheckResult(
                name="gpu_clock",
                verdict=Verdict.PASS if clock_gpu > 500 else Verdict.WARN,
                summary=f"GPU clock: {clock_gpu} MHz, Mem clock: {clock_mem} MHz",
                dimension="gpu_clock",
            ))

        return AgentResult(agent_name=self.name, checks=checks)
