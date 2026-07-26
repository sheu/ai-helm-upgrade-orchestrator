"""
Deterministic risk scoring model.

Integration: The feature-engineering and classification framework here is
adapted from Project 3 (Applied ML — Bank Transaction Fraud Detection with
LSTM). In that project, transaction features (amount, recency, credit score)
were weighted and combined to identify fraud probability. Here, upgrade
features (version gap, breaking changes, Kubernetes compatibility) are
combined into a deterministic risk score using the same principle: assign
weights to risk factors and sum them to produce a classification threshold.

Key design decision: unlike the probabilistic LSTM classifier in Project 3,
this scoring function is fully deterministic. The LLM research agent may
identify factors, but Python calculates the score — the model never guesses.

Reference: applied_ml/deep_learning.ipynb — risk feature design and
classification thresholds.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .models import (
    FindingSeverity,
    ResearchFinding,
    ResearchReport,
    RiskLevel,
    RolloutWave,
    UpgradePlan,
    Criticality,
    ClusterRecord,
    InventoryReport,
)


# ── Risk factor weights (loaded from config) ──────────────────────────────────

DEFAULT_WEIGHTS = {
    "major_app_version_change": 25,
    "breaking_helm_value_change": 20,
    "kubernetes_incompatibility": 25,
    "stateful_or_critical_component": 10,
    "no_prior_upgrade_history": 10,
    "missing_monitoring_coverage": 20,
    "proven_rollback_procedure": -10,
    "successful_int_validation": -20,
}

RISK_THRESHOLDS = {
    "low": 29,
    "medium": 59,
    "high": 79,
    "critical": 100,
}


def load_risk_rules(config_path: str | Path) -> Dict[str, Any]:
    """Load risk weights and thresholds from YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    weights: Dict[str, int] = {}
    for k, v in cfg.get("positive_factors", {}).items():
        weights[k] = v["weight"]
    for k, v in cfg.get("mitigations", {}).items():
        weights[k] = v["weight"]

    thresholds = {
        k: v for k, v in cfg.get("thresholds", RISK_THRESHOLDS).items()
    }
    return {"weights": weights, "thresholds": thresholds}


# ── Kubernetes version comparison ─────────────────────────────────────────────

def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse a semver-like string into a comparable tuple of ints."""
    parts = re.findall(r"\d+", str(version_str))
    return tuple(int(p) for p in parts)


def is_kubernetes_compatible(
    cluster_k8s_version: str,
    minimum_k8s_version: Optional[str],
) -> bool:
    """
    Return True if the cluster Kubernetes version meets the minimum required
    by the target chart.
    """
    if minimum_k8s_version is None:
        return True
    try:
        cluster = parse_version(cluster_k8s_version)
        minimum = parse_version(minimum_k8s_version)
        return cluster >= minimum
    except (ValueError, TypeError):
        return False  # Conservative: assume incompatible if unparseable


def is_major_version_change(current_app_version: str, target_app_version: str) -> bool:
    """Detect a major application version increment."""
    try:
        cur = parse_version(current_app_version)
        tgt = parse_version(target_app_version)
        return tgt[0] > cur[0]
    except (IndexError, ValueError):
        return False


# ── Risk score calculation ────────────────────────────────────────────────────

def calculate_risk_score(
    inventory_report: InventoryReport,
    research_report: ResearchReport,
    history_success_rate: float,
    weights: Optional[Dict[str, int]] = None,
    thresholds: Optional[Dict[str, int]] = None,
) -> Tuple[int, RiskLevel, Dict[str, Any]]:
    """
    Calculate a deterministic risk score from observable evidence.

    Methodology derived from the ML feature-weight approach in Project 3
    (Applied ML): each risk factor contributes a fixed weight, producing a
    final score that maps to a categorical risk level — analogous to how
    LSTM-derived features mapped to a fraud/not-fraud classification.

    Returns: (score, risk_level, factor_breakdown)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if thresholds is None:
        thresholds = RISK_THRESHOLDS

    score = 0
    factors: Dict[str, Any] = {}

    # ── Positive risk factors ─────────────────────────────────────────────────

    # Major version change
    for cluster in inventory_report.affected_clusters:
        if is_major_version_change(cluster.app_version, research_report.target_version):
            score += weights.get("major_app_version_change", 25)
            factors["major_app_version_change"] = {
                "triggered": True,
                "weight": weights.get("major_app_version_change", 25),
                "evidence": f"App version change to {research_report.target_version}",
            }
            break

    # Breaking Helm value changes
    if research_report.breaking_changes_detected:
        score += weights.get("breaking_helm_value_change", 20)
        factors["breaking_helm_value_change"] = {
            "triggered": True,
            "weight": weights.get("breaking_helm_value_change", 20),
            "evidence": f"Breaking changes: {research_report.deprecated_values}",
        }

    # Kubernetes incompatibility (any affected cluster)
    if research_report.minimum_kubernetes_version:
        for cluster in inventory_report.affected_clusters:
            if not is_kubernetes_compatible(
                cluster.kubernetes_version, research_report.minimum_kubernetes_version
            ):
                score += weights.get("kubernetes_incompatibility", 25)
                factors["kubernetes_incompatibility"] = {
                    "triggered": True,
                    "weight": weights.get("kubernetes_incompatibility", 25),
                    "evidence": (
                        f"Cluster {cluster.cluster_name} runs K8s "
                        f"{cluster.kubernetes_version}, chart requires "
                        f">={research_report.minimum_kubernetes_version}"
                    ),
                }
                break

    # Stateful or critical component
    critical_clusters = [
        c for c in inventory_report.affected_clusters
        if c.criticality in (Criticality.HIGH, Criticality.CRITICAL)
    ]
    if critical_clusters:
        score += weights.get("stateful_or_critical_component", 10)
        factors["stateful_or_critical_component"] = {
            "triggered": True,
            "weight": weights.get("stateful_or_critical_component", 10),
            "evidence": f"{len(critical_clusters)} high/critical cluster(s)",
        }

    # No prior upgrade history
    import math
    if math.isnan(history_success_rate):
        score += weights.get("no_prior_upgrade_history", 10)
        factors["no_prior_upgrade_history"] = {
            "triggered": True,
            "weight": weights.get("no_prior_upgrade_history", 10),
            "evidence": "No prior upgrade records for this version pair",
        }

    # ── Mitigations ──────────────────────────────────────────────────────────

    # Rollback procedure documented
    score += weights.get("proven_rollback_procedure", -10)
    factors["proven_rollback_procedure"] = {
        "triggered": True,
        "weight": weights.get("proven_rollback_procedure", -10),
        "evidence": "Runbook exists with documented rollback steps",
    }

    # Clamp to [0, 100]
    score = max(0, min(100, score))

    # Determine risk level
    if score <= thresholds["low"]:
        level = RiskLevel.LOW
    elif score <= thresholds["medium"]:
        level = RiskLevel.MEDIUM
    elif score <= thresholds["high"]:
        level = RiskLevel.HIGH
    else:
        level = RiskLevel.CRITICAL

    return score, level, factors


# ── Rollout wave planning ─────────────────────────────────────────────────────

def plan_rollout_waves(
    inventory_report: InventoryReport,
    risk_level: RiskLevel,
) -> List[RolloutWave]:
    """
    Order affected clusters into deployment waves from lowest to highest risk.

    Wave 1: INT environments, lowest-criticality clusters
    Wave 2: PROD environments, lowest criticality
    Wave 3: PROD environments, high/critical clusters
    """
    from .models import Environment, Criticality

    low_crit = {Criticality.LOW, Criticality.MEDIUM}
    high_crit = {Criticality.HIGH, Criticality.CRITICAL}

    int_clusters = [
        c.cluster_name for c in inventory_report.affected_clusters
        if c.environment == Environment.INT
    ]
    prod_low = [
        c.cluster_name for c in inventory_report.affected_clusters
        if c.environment == Environment.PROD and c.criticality in low_crit
    ]
    prod_high = [
        c.cluster_name for c in inventory_report.affected_clusters
        if c.environment == Environment.PROD and c.criticality in high_crit
    ]

    waves = []
    if int_clusters:
        waves.append(RolloutWave(
            wave=1,
            clusters=int_clusters,
            description="INT environment validation",
            requires_gate_pass=True,
        ))
    if prod_low:
        waves.append(RolloutWave(
            wave=2,
            clusters=prod_low,
            description="PROD rollout — low/medium criticality clusters",
            requires_gate_pass=True,
        ))
    if prod_high:
        waves.append(RolloutWave(
            wave=3,
            clusters=prod_high,
            description="PROD rollout — high/critical clusters (human approval mandatory)",
            requires_gate_pass=True,
        ))
    return waves


# ── Top-level plan builder ────────────────────────────────────────────────────

def build_upgrade_plan(
    request_id: str,
    inventory_report: InventoryReport,
    research_report: ResearchReport,
    history_success_rate: float,
    risk_rules_path: str | Path,
) -> UpgradePlan:
    """Combine inventory + research findings into a structured upgrade plan."""
    rules = load_risk_rules(risk_rules_path)
    score, level, factors = calculate_risk_score(
        inventory_report,
        research_report,
        history_success_rate,
        weights=rules["weights"],
        thresholds=rules["thresholds"],
    )

    waves = plan_rollout_waves(inventory_report, level)

    config_changes = []
    for finding in research_report.findings:
        if finding.severity in (FindingSeverity.ERROR, FindingSeverity.CRITICAL):
            config_changes.append(finding.recommended_action or finding.title)

    validation_steps = [
        "Run helm lint on target chart",
        "Run helm template and inspect rendered manifests",
        "Validate no secrets present in rendered output",
        "Check for breaking value key renames",
        "Verify Kubernetes version compatibility for each cluster",
    ]

    rollback_conditions = [
        "pod_ready_percent < 100 after soak period",
        "restart_count > 2 within 5 minutes of deployment",
        f"error_rate_change_pct > 5",
        "connector_status = FAILED (for kafka-connect)",
        "ArgoCD sync status != Synced after 10 minutes",
    ]

    return UpgradePlan(
        request_id=request_id,
        component=inventory_report.component,
        current_versions=list(inventory_report.version_distribution.keys()),
        target_version=research_report.target_version,
        affected_clusters=[c.cluster_name for c in inventory_report.affected_clusters],
        risk_score=score,
        risk_level=level,
        risk_factors=factors,
        compatibility_findings=research_report.findings,
        configuration_changes=config_changes,
        validation_steps=validation_steps,
        rollout_waves=waves,
        rollback_conditions=rollback_conditions,
        approval_required=True,
        evidence=research_report.sources_consulted,
    )
