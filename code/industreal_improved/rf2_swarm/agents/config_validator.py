"""Config validator — training config consistency, model architecture params."""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.config_validator")


class ConfigValidatorAgent(BaseAgent):
    """Validates training configuration consistency and model architecture.

    Reads config from three sources:
      1. Full train.log (config lines printed at startup, not just log_tail)
      2. state.json (active_heads, etc.)
      3. log_tail for fallback patterns
    """

    def __init__(self) -> None:
        super().__init__("config_validator", "Training config consistency, model arch params")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        log_tail = datastore.get("log_tail", [])
        state = datastore.get("state", {})

        # Build search text from log_tail (fast) and supplemental full-log read
        log_text = "\n".join(log_tail)
        full_log_text = self._read_full_log()
        combined_text = full_log_text if full_log_text else log_text

        # 1. Batch size vs gradient accumulation
        # Log formats:  BATCH_SIZE=4  or  Batch size : 4 x 8 = 32  or  "BATCH_SIZE": 4
        batch_size = self._find_value(combined_text, r"(?:batch[_ ]?size)\s*\"?\s*[:=]\s*(\d+)")
        # Accum steps may be explicit or embedded in batch_size line as "4 x 8"
        accum_steps = self._find_value(combined_text, r"(?:accum[_\s]*steps?|gradient_accumulation)\s*\"?\s*[:=]\s*(\d+)")
        if accum_steps is None:
            accum_steps = self._find_accum_steps(combined_text)
        if batch_size is not None and accum_steps is not None:
            effective_bs = int(batch_size * accum_steps)
            if effective_bs >= 16:
                v = Verdict.PASS
            elif effective_bs >= 8:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="effective_batch_size",
                verdict=v,
                summary=f"Effective batch: {batch_size} × {accum_steps} = {effective_bs}",
                detail="Small effective batch = noisy gradients",
                metric=float(effective_bs),
                threshold=16,
                dimension="effective_batch",
            ))
        else:
            checks.append(CheckResult(name="effective_batch_size", verdict=Verdict.SKIP,
                                       summary="batch_size or accum_steps not found"))

        # 2. Learning rate sanity
        # Log formats:  BASE_LR: 0.0005  or  "BASE_LR": 0.0005  or  lr = backbone=5.0e-05
        lr = self._find_value(combined_text, r"(?:base_lr|learning_rate|learning rate|LR|lr)\s*\"?\s*[:=]\s*([\d.e+\-]+)")
        if lr is not None:
            if 1e-6 <= lr <= 1e-3:
                v = Verdict.PASS
            elif lr < 1e-6:
                v = Verdict.FAIL
            else:
                v = Verdict.WARN
            checks.append(CheckResult(
                name="learning_rate_sanity",
                verdict=v,
                summary=f"Learning rate: {lr:.2e}",
                metric=lr,
                threshold=(1e-6, 1e-3),
                dimension="lr_sanity",
            ))
        else:
            checks.append(CheckResult(name="learning_rate_sanity", verdict=Verdict.SKIP,
                                       summary="LR not found in log"))

        # 3. Head configuration
        active_heads = state.get("active_heads", "")
        if active_heads:
            if "det" in active_heads.lower():
                v = Verdict.PASS
            else:
                v = Verdict.CRIT
            checks.append(CheckResult(
                name="active_heads_config",
                verdict=v,
                summary=f"Active heads: {active_heads}",
                dimension="active_heads",
            ))
        else:
            checks.append(CheckResult(name="active_heads_config", verdict=Verdict.SKIP,
                                       summary="active_heads not in state"))

        # 4. Subset ratio
        # Log format:  Subset ratio=0.5: train 36->18 recs
        subset = self._find_value(combined_text, r"(?:subset[_\s]*ratio|subset[_\s]*size)\s*\"?\s*[:=]\s*([\d.]+)")
        if subset is not None:
            v = Verdict.PASS if subset >= 0.3 else Verdict.WARN
            checks.append(CheckResult(
                name="subset_ratio",
                verdict=v,
                summary=f"Subset ratio: {subset}",
                detail="Low subset = fewer training samples per epoch",
                metric=subset,
                threshold=0.3,
                dimension="subset_ratio",
            ))
        else:
            checks.append(CheckResult(name="subset_ratio", verdict=Verdict.SKIP,
                                       summary="subset_ratio not in log"))

        # 5. Model architecture / backbone
        backbone_lines = [l for l in log_tail if "backbone" in l.lower()]
        if not backbone_lines and full_log_text:
            # Search full log for backbone config printed at startup
            lines = full_log_text.split("\n")
            backbone_lines = [l for l in lines if "backbone" in l.lower()]
        if backbone_lines:
            checks.append(CheckResult(
                name="backbone_config",
                verdict=Verdict.PASS,
                summary=f"Backbone config found: {backbone_lines[-1][:80]}",
                dimension="backbone_config",
            ))
        else:
            checks.append(CheckResult(name="backbone_config", verdict=Verdict.SKIP,
                                       summary="No backbone config in log"))

        # 6. Precision / AMP
        amp_lines = [l for l in log_tail if "amp" in l.lower() or "precision" in l.lower() or "fp16" in l.lower()]
        if not amp_lines and full_log_text:
            lines = full_log_text.split("\n")
            amp_lines = [l for l in lines if "amp" in l.lower() or "precision" in l.lower() or "fp16" in l.lower()]
        if amp_lines:
            checks.append(CheckResult(
                name="training_precision",
                verdict=Verdict.PASS,
                summary=f"Precision/AMP: {amp_lines[-1][:80]}",
                dimension="training_precision",
            ))
        else:
            checks.append(CheckResult(name="training_precision", verdict=Verdict.SKIP,
                                       summary="No precision info in log"))

        # 7. Max epochs consistency
        # Log formats:  Epochs: 100  or  "EPOCHS": 100
        max_epochs = self._find_value(combined_text, r"(?:max_epochs?|epochs?)\s*\"?\s*[:=]\s*(\d+)")
        stage_max = 30
        if max_epochs is not None:
            if max_epochs >= stage_max:
                v = Verdict.PASS
            else:
                v = Verdict.WARN
            checks.append(CheckResult(
                name="max_epochs_config",
                verdict=v,
                summary=f"Max epochs: {max_epochs} (stage RF2: {stage_max})",
                metric=float(max_epochs),
                threshold=stage_max,
                dimension="max_epochs",
            ))
        else:
            checks.append(CheckResult(name="max_epochs_config", verdict=Verdict.SKIP,
                                       summary="max_epochs not found"))

        return AgentResult(agent_name=self.name, checks=checks)

    def _read_full_log(self) -> str | None:
        """Read the first portion of train.log where config is printed at startup."""
        try:
            if not os.path.isfile(C.TRAIN_LOG):
                return None
            # Config is printed in the first ~500 lines of training startup
            with open(C.TRAIN_LOG, "r") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 500:
                        break
                    lines.append(line.rstrip("\n"))
            return "\n".join(lines)
        except (OSError, IOError) as e:
            logger.warning("Could not read train.log for config: %s", e)
            return None

    def _find_value(self, text: str, pattern: str) -> float | None:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    def _find_accum_steps(self, text: str) -> float | None:
        """Parse gradient accumulation steps from batch_size line like '4 x 8 = 32'."""
        m = re.search(r"(?:batch|accum)[_\s]*size[^=]*[:=].*?x\s*(\d+)", text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None
