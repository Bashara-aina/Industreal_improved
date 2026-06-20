"""DET_PROBE analyzer — per-epoch mAP progress, class-level APs, probe consistency."""
from __future__ import annotations

import ast
import logging
import re
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.probe_analyzer")


class ProbeAnalyzerAgent(BaseAgent):
    """Parses DET_PROBE lines from validation output for mAP progress.

    DET_PROBE lines have JSON dicts with score/IoU distributions (not mAP).
    Actual mAP values come from metrics.jsonl val blocks.
    """

    def __init__(self) -> None:
        super().__init__("probe_analyzer", "DET_PROBE results per epoch, mAP progress, class-level APs")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        metrics = datastore.get("metrics", [])
        probe_lines = datastore.get("full_det_probe", [])

        # Extract per-class AP from the latest completed val epoch in metrics.jsonl
        class_aps: Dict[str, float] = {}
        latest_val_map = None
        latest_val_epoch = 0
        for m in metrics:
            val = m.get("val", {})
            det_map = val.get("det_mAP50")
            if det_map is not None:
                latest_val_map = float(det_map)
                latest_val_epoch = m.get("epoch", 0)
                pc_aps = val.get("det_per_class_ap", {})
                if pc_aps:
                    class_aps = {f"class_{k}": float(v) for k, v in pc_aps.items() if v is not None}
                break

        # 1. Overall mAP50 from latest val
        if latest_val_map is not None:
            if latest_val_map >= C.GATE.det_mAP50 * 0.4:
                v = Verdict.PASS
            elif latest_val_map >= C.GATE.det_mAP50 * 0.2:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="val_mAP50",
                verdict=v,
                summary=f"Latest val det_mAP50={latest_val_map:.4f} (epoch {latest_val_epoch})",
                metric=latest_val_map,
                threshold=C.GATE.det_mAP50 * 0.2,
                dimension="val_mAP50",
            ))
        else:
            checks.append(CheckResult(name="val_mAP50", verdict=Verdict.SKIP,
                                       summary="No val det_mAP50 in metrics.jsonl"))

        # 2. mAP progression across val epochs
        # metrics is newest-first; reverse to chronological
        val_maps = [float(m.get("val", {}).get("det_mAP50", 0)) for m in metrics
                    if m.get("val", {}).get("det_mAP50") is not None]
        val_maps = [v for v in val_maps if v > 0]
        if len(val_maps) >= 3:
            recent = val_maps[:3][::-1]  # 3 most recent, chronological
            trend = recent[-1] - recent[0]
            if trend > 0.01:
                v = Verdict.PASS
            elif trend > 0.0:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="val_trend_3epoch",
                verdict=v,
                summary=f"Val mAP trend over 3 epochs: {trend:+.4f}",
                detail=f"Recent values: {[f'{x:.4f}' for x in recent]}",
                metric=trend,
                threshold=0.0,
                dimension="val_trend",
            ))
        else:
            checks.append(CheckResult(name="val_trend_3epoch", verdict=Verdict.SKIP,
                                       summary=f"Only {len(val_maps)} val mAP entries, need 3"))

        # 3. Class-level APs from metrics.jsonl val.det_per_class_ap
        low_ap_classes = {}
        zero_ap_classes = {}
        if class_aps:
            for cls_name, ap_val in class_aps.items():
                if ap_val <= 0.0 and ap_val == 0.0:
                    zero_ap_classes[cls_name] = ap_val
                elif ap_val < 0.1:
                    low_ap_classes[cls_name] = ap_val
                checks.append(CheckResult(
                    name=f"class_AP_{cls_name}",
                    verdict=Verdict.PASS if ap_val >= 0.3 else (Verdict.WARN if ap_val >= 0.1 else Verdict.FAIL),
                    summary=f"{cls_name} AP={ap_val:.4f}",
                    metric=ap_val,
                    threshold=0.1,
                    dimension=f"class_AP_{cls_name}",
                ))
            if zero_ap_classes:
                checks.append(CheckResult(
                    name="zero_ap_classes",
                    verdict=Verdict.FAIL,
                    summary=f"{len(zero_ap_classes)} classes with AP=0: {list(zero_ap_classes.keys())[:5]}...",
                    detail=f"Zero-AP classes: {zero_ap_classes}",
                    dimension="zero_ap_classes",
                ))
        else:
            checks.append(CheckResult(name="class_level_APs", verdict=Verdict.SKIP,
                                       summary="No per-class AP in metrics.jsonl"))

        # 4. Parse DET_PROBE statistics (score distribution, IoU stats)
        probe_records = self._parse_probe_json(probe_lines)
        if probe_records:
            # Average bestIoU_mean across recent probes (higher = better localization)
            recent_probes = probe_records[-20:]
            avg_iou = sum(p.get("bestIoU_mean", 0) for p in recent_probes) / len(recent_probes)
            avg_score_p50 = sum(p.get("score_p50", 0) for p in recent_probes) / len(recent_probes)
            total_preds_above_05 = sum(p.get("bestIoU>0.5", 0) for p in recent_probes)
            total_gt = sum(p.get("n_gt", 0) for p in recent_probes)

            checks.append(CheckResult(
                name="probe_avg_iou",
                verdict=Verdict.PASS if avg_iou > 0.06 else (Verdict.WARN if avg_iou > 0.03 else Verdict.FAIL),
                summary=f"Avg bestIoU_mean: {avg_iou:.4f} (recent {len(recent_probes)} probes)",
                metric=avg_iou,
                threshold=0.03,
                dimension="probe_avg_iou",
            ))
            checks.append(CheckResult(
                name="probe_score_p50",
                verdict=Verdict.WARN if avg_score_p50 < 0.01 else Verdict.PASS,
                summary=f"Avg score_p50: {avg_score_p50:.4f}",
                metric=avg_score_p50,
                dimension="probe_score_p50",
            ))
            checks.append(CheckResult(
                name="probe_preds_per_gt",
                verdict=Verdict.WARN if total_gt > 0 and total_preds_above_05 < total_gt else Verdict.PASS,
                summary=f"Preds@IoU>0.5 per GT: {total_preds_above_05}/{total_gt} ({total_preds_above_05 / max(total_gt, 1):.1f}x)",
                metric=total_preds_above_05 / max(total_gt, 1),
                dimension="probe_preds_per_gt",
            ))
        else:
            checks.append(CheckResult(name="probe_statistics", verdict=Verdict.SKIP,
                                       summary="No DET_PROBE lines parsed"))

        # 5. Probe consistency (std of score_p50 across recent probes)
        if len(probe_records) >= 3:
            import statistics
            score_vals = [p.get("score_p50", 0) for p in probe_records[-10:]]
            std = statistics.stdev(score_vals) if len(score_vals) >= 2 else 0
            checks.append(CheckResult(
                name="probe_consistency",
                verdict=Verdict.PASS if std < 0.02 else (Verdict.WARN if std < 0.05 else Verdict.FAIL),
                summary=f"Probe score_p50 std: {std:.4f} over {len(score_vals)} entries",
                metric=std,
                threshold=0.02,
                dimension="probe_stability",
            ))
        else:
            checks.append(CheckResult(name="probe_consistency", verdict=Verdict.SKIP,
                                       summary="Need ≥3 probe samples for std"))

        # 6. Parse DET_PROBE verdict counts
        verdicts = self._parse_probe_verdicts(probe_lines)
        if verdicts:
            localizing_pct = verdicts.get("LOCALIZING", 0) / max(sum(verdicts.values()), 1) * 100
            checks.append(CheckResult(
                name="probe_verdict_localizing",
                verdict=Verdict.PASS if localizing_pct > 50 else Verdict.WARN,
                summary=f"LOCALIZING verdicts: {localizing_pct:.0f}% ({verdicts})",
                detail="LOCALIZING = model finds GT boxes above chance",
                metric=localizing_pct / 100,
                dimension="probe_localizing_pct",
            ))

        return AgentResult(agent_name=self.name, checks=checks)

    def _parse_probe_json(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse Python dict literals embedded in DET_PROBE lines.

        DET_PROBE lines use Python format: ``{'tag': 'b695', 'n_gt': 4}``
        not JSON. Use ast.literal_eval for safe parsing.
        """
        records = []
        for line in lines:
            m = re.search(r"\{.*\}", line)
            if m:
                try:
                    data = ast.literal_eval(m.group(0))
                    if isinstance(data, dict):
                        records.append(data)
                except (ValueError, SyntaxError, MemoryError):
                    pass
        return records

    def _parse_probe_verdicts(self, lines: List[str]) -> Dict[str, int]:
        """Count verdict types (LOCALIZING, DETECTING, etc.) in DET_PROBE lines."""
        verdicts: Dict[str, int] = {}
        for line in lines:
            m = re.search(r"verdict:\s*(\w+)", line)
            if m:
                v = m.group(1)
                verdicts[v] = verdicts.get(v, 0) + 1
        return verdicts
