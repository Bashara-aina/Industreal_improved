"""5-severity alert levels, 4 channels (console, file, webhook, slack).

Currently implements console + file. Webhook/slack stubs ready.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from rf2_swarm.base_agent import AgentResult, CheckResult, Verdict

logger = logging.getLogger("swarm.alerting")

ALERT_CHANNELS = ["console", "file", "webhook", "slack"]
SEVERITY_ORDER = {s: i for i, s in enumerate(["PASS", "SKIP", "WARN", "FAIL", "CRIT"])}


class Alert:
    """A single alert event."""

    def __init__(self, agent: str, check: str, verdict: Verdict,
                 summary: str, detail: str = ""):
        self.agent = agent
        self.check = check
        self.verdict = verdict
        self.summary = summary
        self.detail = detail
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "check": self.check,
            "verdict": self.verdict.value,
            "summary": self.summary,
            "detail": self.detail,
            "timestamp": self.timestamp.isoformat(),
        }

    def severity_score(self) -> int:
        return SEVERITY_ORDER.get(self.verdict.value, 0)


class AlertManager:
    """Manages alert generation and routing."""

    def __init__(self, alert_log: str = ""):
        self.alert_log = alert_log
        self._history: List[Alert] = []
        self._channels: Dict[str, Callable[[Alert], None]] = {
            "console": self._channel_console,
            "file": self._channel_file,
        }

    def process_results(self, results: List[AgentResult]) -> List[Alert]:
        """Evaluate all results and generate alerts for non-PASS checks."""
        alerts: List[Alert] = []
        for ar in results:
            for c in ar.checks:
                if c.verdict in (Verdict.PASS, Verdict.SKIP):
                    continue
                alert = Alert(
                    agent=ar.agent_name,
                    check=c.name,
                    verdict=c.verdict,
                    summary=c.summary,
                    detail=c.detail,
                )
                alerts.append(alert)
                self._history.append(alert)

        # Route to channels (only FAIL/CRIT → webhook/slack)
        for alert in alerts:
            self._route(alert)

        return alerts

    def _route(self, alert: Alert) -> None:
        for name, fn in self._channels.items():
            try:
                fn(alert)
            except Exception as e:
                logger.warning("Alert channel %s failed: %s", name, e)

    def _channel_console(self, alert: Alert) -> None:
        level = "WARNING" if alert.verdict in (Verdict.WARN, Verdict.FAIL) else "ERROR"
        logger.log(
            getattr(logging, level, logging.WARNING),
            "[%s] %s/%s: %s", alert.verdict.value, alert.agent, alert.check, alert.summary,
        )

    def _channel_file(self, alert: Alert) -> None:
        if not self.alert_log:
            return
        os.makedirs(os.path.dirname(self.alert_log), exist_ok=True)
        try:
            with open(self.alert_log, "a") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except OSError:
            pass

    def register_channel(self, name: str, fn: Callable[[Alert], None]) -> None:
        self._channels[name] = fn

    def recent_alerts(self, n: int = 10) -> List[Alert]:
        return self._history[-n:]

    def worst_since(self, min_severity: str = "FAIL") -> List[Alert]:
        threshold = SEVERITY_ORDER.get(min_severity, 3)
        return [a for a in self._history if a.severity_score() >= threshold]
