"""ThreadPoolExecutor dispatch, delta tracking, timeouts."""
from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from rf2_swarm.base_agent import AgentResult, BaseAgent, Verdict

logger = logging.getLogger("swarm.coordinator")


class DeltaTracker:
    """Tracks per-check verdict changes between cycles."""

    def __init__(self):
        self._prev: Dict[str, str] = {}

    def update(self, results: List[AgentResult]) -> List[Dict[str, Any]]:
        """Compare results against previous cycle, return deltas."""
        deltas: List[Dict[str, Any]] = []
        current: Dict[str, str] = {}
        for ar in results:
            for c in ar.checks:
                key = f"{ar.agent_name}/{c.name}"
                current[key] = c.verdict.value
                prev = self._prev.get(key)
                if prev is not None and prev != c.verdict.value:
                    deltas.append({
                        "key": key,
                        "from": prev,
                        "to": c.verdict.value,
                        "summary": c.summary,
                    })
        self._prev = current
        return deltas


class Coordinator:
    """Runs all agents in parallel, collects results, tracks deltas."""

    def __init__(self, agents: List[BaseAgent], max_workers: int = 40,
                 agent_timeout: int = 120):
        self.agents = agents
        self.max_workers = max_workers
        self.agent_timeout = agent_timeout
        self.delta_tracker = DeltaTracker()
        self.datastore: Dict[str, Any] = {}

    def run_cycle(self, datastore: Dict[str, Any]) -> List[AgentResult]:
        """Execute all agents in parallel and return results."""
        self.datastore = datastore
        results: List[AgentResult] = []
        futures: Dict[concurrent.futures.Future, str] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for agent in self.agents:
                future = pool.submit(self._run_safe, agent)
                futures[future] = agent.name

            for future in concurrent.futures.as_completed(futures, timeout=300):
                name = futures[future]
                try:
                    result = future.result(timeout=10)
                    results.append(result)
                except Exception as e:
                    logger.error("Agent %s crashed: %s", name, e)
                    results.append(AgentResult(
                        agent_name=name,
                        error=f"Unhandled exception: {e}",
                    ))

        # Sort by agent name for deterministic output
        results.sort(key=lambda r: r.agent_name)
        return results

    def get_deltas(self, results: List[AgentResult]) -> List[Dict[str, Any]]:
        return self.delta_tracker.update(results)

    def _run_safe(self, agent: BaseAgent) -> AgentResult:
        try:
            result = agent.run(self.datastore)
            logger.info("Agent %s: %d checks, worst=%s, %.1fs",
                        agent.name, len(result.checks),
                        result.worst.value, result.elapsed)
            return result
        except Exception as e:
            logger.exception("Agent %s failed:", agent.name)
            return AgentResult(
                agent_name=agent.name,
                error=f"Exception: {e}",
            )
