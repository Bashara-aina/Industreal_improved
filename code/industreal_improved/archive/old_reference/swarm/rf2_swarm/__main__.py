"""CLI entry point for the RF2 monitoring swarm.

Usage:
    python -m rf2_swarm --oneshot          # single cycle, report, exit
    python -m rf2_swarm --interval 300     # continuous loop every 5 min
    python -m rf2_swarm --list-agents      # list all agents and exit
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import List

from rf2_swarm import config as C
from rf2_swarm.runner import SwarmRunner


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_agents() -> List:
    """Instantiate all 20 monitoring agents."""
    from rf2_swarm.agents import (
        GateTrackerAgent, ProbeAnalyzerAgent, HeadHealthAgent,
        LossHealthAgent, ConvergenceAgent, DataPipelineAgent,
        CheckpointAgent, GPUResourceAgent, ValidationAgent,
        HeadRecoveryAgent, MetricsLoggerAgent, GatePredictorAgent,
        ProcessHealthAgent, EpochTrackerAgent, NanDetectorAgent,
        CudaHealthAgent, ConfigValidatorAgent, LogAnomalyAgent,
        BlockerAssessmentAgent, SummaryAgent,
    )
    return [
        GateTrackerAgent(),
        ProbeAnalyzerAgent(),
        HeadHealthAgent(),
        LossHealthAgent(),
        ConvergenceAgent(),
        DataPipelineAgent(),
        CheckpointAgent(),
        GPUResourceAgent(),
        ValidationAgent(),
        HeadRecoveryAgent(),
        MetricsLoggerAgent(),
        GatePredictorAgent(),
        ProcessHealthAgent(),
        EpochTrackerAgent(),
        NanDetectorAgent(),
        CudaHealthAgent(),
        ConfigValidatorAgent(),
        LogAnomalyAgent(),
        BlockerAssessmentAgent(),
        SummaryAgent(),
    ]


def list_agents() -> None:
    print(f"{'Agent':<25} Description")
    print("-" * 90)
    for name, desc in sorted(C.AGENT_DEFINITIONS.items()):
        print(f"  {name:<23} {desc}")
    print(f"\nTotal: {len(C.AGENT_DEFINITIONS)} agents defined in config")


def main() -> None:
    parser = argparse.ArgumentParser(description="RF2 20-Agent Monitoring Swarm")
    parser.add_argument("--oneshot", action="store_true", help="Run single cycle and exit")
    parser.add_argument("--interval", type=int, default=C.POLL_INTERVAL_SECONDS,
                        help=f"Poll interval in seconds (default: {C.POLL_INTERVAL_SECONDS})")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--list-agents", action="store_true", help="List all agents and exit")
    args = parser.parse_args()

    if args.list_agents:
        list_agents()
        sys.exit(0)

    setup_logging(args.verbose)

    agents = build_agents()
    logger = logging.getLogger("swarm.main")
    logger.info("Initialized %d agents for RF2 monitoring swarm", len(agents))

    runner = SwarmRunner(
        agents=agents,
        interval=args.interval,
        oneshot=args.oneshot,
    )
    runner.run()


if __name__ == "__main__":
    main()
