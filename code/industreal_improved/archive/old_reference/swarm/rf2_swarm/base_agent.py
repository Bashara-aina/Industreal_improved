"""Base classes for all monitoring agents."""
from __future__ import annotations

import enum
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class Verdict(enum.Enum):
    """Standardised check outcome."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    CRIT = "CRIT"
    SKIP = "SKIP"  # check was not applicable


@dataclass
class CheckResult:
    """Outcome of a single check within an agent."""
    name: str
    verdict: Verdict
    summary: str                       # one-line human-readable
    detail: str = ""                   # extended evidence
    metric: Optional[float] = None     # numeric value if applicable
    threshold: Optional[float] = None  # comparison threshold
    dimension: str = ""                # e.g. "det_mAP50", "VRAM_GB"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict.value,
            "summary": self.summary,
            "detail": self.detail,
            "metric": self.metric,
            "threshold": self.threshold,
            "dimension": self.dimension,
        }


@dataclass
class AgentResult:
    """Aggregated output from one agent run."""
    agent_name: str
    checks: List[CheckResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    error: Optional[str] = None

    @property
    def elapsed(self) -> float:
        return self.finished_at - self.started_at

    @property
    def passed(self) -> bool:
        return all(c.verdict in (Verdict.PASS, Verdict.SKIP) for c in self.checks)

    @property
    def worst(self) -> Verdict:
        order = [Verdict.PASS, Verdict.SKIP, Verdict.WARN, Verdict.FAIL, Verdict.CRIT]
        best = {v: i for i, v in enumerate(order)}
        return max(self.checks, key=lambda c: best.get(c.verdict, 0)).verdict if self.checks else Verdict.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "elapsed_s": round(self.elapsed, 2),
            "worst": self.worst.value,
            "passed": self.passed,
            "error": self.error,
            "num_checks": len(self.checks),
            "checks": [c.to_dict() for c in self.checks],
        }


class BaseAgent(ABC):
    """Every monitoring agent subclasses this."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        ...
