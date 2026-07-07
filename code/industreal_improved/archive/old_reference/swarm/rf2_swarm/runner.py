"""Main monitoring loop — signal handling, 5-min interval."""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import List, Optional

from rf2_swarm import config as C
from rf2_swarm.alerting import AlertManager
from rf2_swarm.base_agent import BaseAgent
from rf2_swarm.coordinator import Coordinator
from rf2_swarm.data_sources import gather_all
from rf2_swarm.reporter import write_reports

logger = logging.getLogger("swarm.runner")


class SwarmRunner:
    """Main loop that orchestrates monitoring cycles."""

    def __init__(self, agents: List[BaseAgent],
                 interval: int = C.POLL_INTERVAL_SECONDS,
                 oneshot: bool = False):
        self.agents = agents
        self.interval = interval
        self.oneshot = oneshot
        self.coordinator = Coordinator(agents, max_workers=40, agent_timeout=C.AGENT_TIMEOUT_SECONDS)
        self.alert_mgr = AlertManager(alert_log=os.path.join(C.SWARM_OUTPUT_DIR, "alerts.jsonl"))
        self.cycle_num = 0
        self._shutdown = False

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Received signal %d — shutting down after current cycle", signum)
        self._shutdown = True

    def run(self) -> None:
        """Run monitoring loop (or single cycle if oneshot)."""
        logger.info("Swarm runner started — %d agents, interval=%ds, oneshot=%s",
                    len(self.agents), self.interval, self.oneshot)

        os.makedirs(C.SWARM_OUTPUT_DIR, exist_ok=True)

        while not self._shutdown:
            self.cycle_num += 1
            self._run_one_cycle()

            if self.oneshot or self._shutdown:
                break

            logger.info("Cycle %d complete. Next cycle in %ds.", self.cycle_num, self.interval)
            time.sleep(self.interval)

        logger.info("Swarm runner stopped after %d cycles.", self.cycle_num)

    def _run_one_cycle(self) -> None:
        """Execute a single monitoring cycle."""
        logger.info("=" * 60)
        logger.info("Cycle %d starting — gathering data...", self.cycle_num)

        # Fresh data snapshot
        datastore = gather_all()
        logger.info("Epoch=%s step=%s pid_alive=%s",
                    datastore.get("epoch"), datastore.get("step"),
                    datastore.get("pid_alive"))

        # Run all agents
        results = self.coordinator.run_cycle(datastore)

        # Delta tracking
        deltas = self.coordinator.get_deltas(results)

        # Alerting
        alerts = self.alert_mgr.process_results(results)

        # Write reports
        write_reports(
            results=results,
            datastore=datastore,
            cycle_num=self.cycle_num,
            json_path=C.SWARM_RESULTS_JSON,
            txt_path=C.SWARM_REPORT_TXT,
            alerts=alerts,
            deltas=deltas,
        )

        fail_count = sum(1 for ar in results for c in ar.checks
                         if c.verdict.value in ("FAIL", "CRIT"))
        logger.info("Cycle %d done — %d FAIL/CRIT checks", self.cycle_num, fail_count)
