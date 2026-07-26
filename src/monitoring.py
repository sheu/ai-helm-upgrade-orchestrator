"""
Health monitoring and evaluation.

Evaluates post-deployment health snapshots against quality gate thresholds.
UNKNOWN is never treated as PASS — this is the key safety invariant.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .models import GateResult, HealthEvaluationResult, HealthSnapshot


def load_quality_gates(config_path: str | Path) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_health(
    snapshot: Optional[HealthSnapshot],
    gates_config: Dict,
) -> HealthEvaluationResult:
    """
    Evaluate a health snapshot against configured thresholds.

    Critical invariant: if snapshot is None or evidence_available is False,
    all gates return UNKNOWN — never PASS. This prevents an absent monitoring
    system from being treated as a successful deployment.
    """
    cluster = snapshot.cluster if snapshot else "unknown"

    if snapshot is None or not snapshot.evidence_available:
        return HealthEvaluationResult(
            cluster=cluster,
            pod_readiness_gate=GateResult.UNKNOWN,
            restart_gate=GateResult.UNKNOWN,
            error_rate_gate=GateResult.UNKNOWN,
            latency_gate=GateResult.UNKNOWN,
            memory_gate=GateResult.UNKNOWN,
            overall=GateResult.UNKNOWN,
            notes=["No monitoring evidence available — UNKNOWN state. Human investigation required."],
        )

    health_cfg = gates_config.get("health", {})
    notes: List[str] = []

    # Pod readiness
    min_ready = health_cfg.get("minimum_ready_replicas_percent", 100)
    if snapshot.pod_ready_percent is None:
        pod_gate = GateResult.UNKNOWN
        notes.append("pod_ready_percent not available")
    elif snapshot.pod_ready_percent >= min_ready:
        pod_gate = GateResult.PASS
    else:
        pod_gate = GateResult.FAIL
        notes.append(f"pod_ready_percent={snapshot.pod_ready_percent} < {min_ready}")

    # Restart count
    max_restarts = health_cfg.get("maximum_new_restart_count", 2)
    if snapshot.restart_count is None:
        restart_gate = GateResult.UNKNOWN
        notes.append("restart_count not available")
    elif snapshot.restart_count <= max_restarts:
        restart_gate = GateResult.PASS
    else:
        restart_gate = GateResult.FAIL
        notes.append(f"restart_count={snapshot.restart_count} > {max_restarts}")

    # Error rate
    max_err = health_cfg.get("maximum_error_rate_increase_percent", 5.0)
    if snapshot.error_rate_change_pct is None:
        err_gate = GateResult.UNKNOWN
        notes.append("error_rate_change_pct not available")
    elif snapshot.error_rate_change_pct <= max_err:
        err_gate = GateResult.PASS
    else:
        err_gate = GateResult.FAIL
        notes.append(f"error_rate_change={snapshot.error_rate_change_pct}% > {max_err}%")

    # Latency
    max_lat = health_cfg.get("maximum_latency_increase_percent", 10.0)
    if snapshot.latency_change_pct is None:
        lat_gate = GateResult.UNKNOWN
        notes.append("latency_change_pct not available")
    elif snapshot.latency_change_pct <= max_lat:
        lat_gate = GateResult.PASS
    else:
        lat_gate = GateResult.FAIL
        notes.append(f"latency_change={snapshot.latency_change_pct}% > {max_lat}%")

    # Memory
    max_mem = health_cfg.get("maximum_memory_increase_percent", 20.0)
    if snapshot.memory_change_pct is None:
        mem_gate = GateResult.UNKNOWN
        notes.append("memory_change_pct not available")
    elif snapshot.memory_change_pct <= max_mem:
        mem_gate = GateResult.PASS
    else:
        mem_gate = GateResult.FAIL
        notes.append(f"memory_change={snapshot.memory_change_pct}% > {max_mem}%")

    gates = [pod_gate, restart_gate, err_gate, lat_gate, mem_gate]
    if any(g == GateResult.FAIL for g in gates):
        overall = GateResult.FAIL
    elif any(g == GateResult.UNKNOWN for g in gates):
        overall = GateResult.UNKNOWN
    else:
        overall = GateResult.PASS

    return HealthEvaluationResult(
        cluster=cluster,
        pod_readiness_gate=pod_gate,
        restart_gate=restart_gate,
        error_rate_gate=err_gate,
        latency_gate=lat_gate,
        memory_gate=mem_gate,
        overall=overall,
        notes=notes,
    )
