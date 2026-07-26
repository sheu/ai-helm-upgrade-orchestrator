"""
GitOps change generation.

Produces proposed changes as files under outputs/proposed_changes/.
The system proposes changes; it does not directly commit or push to production.
This separation is a core safety constraint: propose → human reviews → approves.
"""
from __future__ import annotations

import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

from .models import (
    ClusterRecord,
    Environment,
    ProposedChange,
    ResearchReport,
    UpgradePlan,
)


def _apply_value_migrations(values: Dict, target_version: str) -> Dict:
    """
    Apply known value key migrations for the target chart version.
    For kafka-connect 0.19.0: the old kafka.connect.config.storage.*
    keys are left in place but flagged (actual rename requires manual review).
    """
    migrated = dict(values)
    migrations_applied = []

    # kafka-connect 0.19.0 key rename detection
    connect_section = migrated.get("kafka", {}).get("connect", {})
    if connect_section:
        old_key = "config.storage.replication.factor"
        if old_key in connect_section:
            migrations_applied.append(
                f"Note: '{old_key}' detected — verify rename to "
                f"'connect.config.storage.replication.factor' for chart {target_version}"
            )

    migrated["_migration_notes"] = migrations_applied
    return migrated


def generate_proposed_change(
    cluster: ClusterRecord,
    plan: UpgradePlan,
    research: ResearchReport,
    base_project_path: str | Path,
) -> ProposedChange:
    """Generate a proposed change for a single cluster."""
    base = Path(base_project_path)
    values_path = base / cluster.values_path

    # Load current values
    try:
        with open(values_path) as f:
            current_content = f.read()
        current_values = yaml.safe_load(current_content) or {}
    except FileNotFoundError:
        current_content = "# values file not found"
        current_values = {}

    # Build proposed values
    proposed_values = dict(current_values)
    proposed_values["chart"] = {
        "repository": proposed_values.get("chart", {}).get("repository", "internal-platform"),
        "name": cluster.component,
        "version": plan.target_version,
    }
    proposed_values["image"] = {
        "repository": proposed_values.get("image", {}).get("repository",
            f"confluentinc/cp-{cluster.component}"),
        "tag": plan.target_version,
        "pullPolicy": "IfNotPresent",
    }

    # Apply known migrations
    proposed_values = _apply_value_migrations(proposed_values, plan.target_version)
    migration_notes = proposed_values.pop("_migration_notes", [])

    proposed_content = yaml.dump(proposed_values, default_flow_style=False)

    # Produce unified diff (safe — no secrets, no credentials)
    diff_lines = list(difflib.unified_diff(
        current_content.splitlines(keepends=True),
        proposed_content.splitlines(keepends=True),
        fromfile=f"current/{cluster.values_path}",
        tofile=f"proposed/{cluster.values_path}",
        lineterm="",
    ))
    diff_content = "".join(diff_lines) if diff_lines else "# No content changes detected"

    value_changes = migration_notes + [
        f.recommended_action for f in research.findings
        if f.recommended_action
    ]

    pr_description = (
        f"## Helm Upgrade: {cluster.component} {cluster.chart_version} → {plan.target_version}\n\n"
        f"**Cluster:** {cluster.cluster_name}\n"
        f"**Environment:** {cluster.environment.value}\n"
        f"**Risk Level:** {plan.risk_level.value.upper()}\n\n"
        f"### Changes\n"
        + "\n".join(f"- {c}" for c in value_changes or ["Chart and image version bump"])
        + f"\n\n### Validation Evidence\n"
        + "\n".join(f"- {e}" for e in plan.evidence or ["See upgrade plan"])
        + f"\n\n### Rollback\n"
        f"Rollback to chart version `{cluster.chart_version}` via ArgoCD.\n\n"
        f"**Requires human approval before PROD deployment.**\n"
    )

    return ProposedChange(
        cluster=cluster.cluster_name,
        environment=cluster.environment,
        values_path=str(cluster.values_path),
        current_chart_version=cluster.chart_version,
        target_chart_version=plan.target_version,
        current_app_version=cluster.app_version,
        target_app_version=research.target_version,
        value_migrations=value_changes,
        diff_content=diff_content,
        rollback_version=cluster.chart_version,
        pr_description=pr_description,
    )


def write_proposed_changes(
    changes: List[ProposedChange],
    output_dir: str | Path,
) -> List[str]:
    """
    Write proposed changes to output_dir/proposed_changes/ and return
    a list of written file paths.
    """
    out = Path(output_dir) / "proposed_changes"
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for change in changes:
        env_label = change.environment.value.lower()
        safe_cluster = change.cluster.replace("/", "-")
        diff_path = out / f"{safe_cluster}-{env_label}.diff"
        pr_path = out / f"{safe_cluster}-{env_label}-pr.md"

        diff_path.write_text(change.diff_content)
        pr_path.write_text(change.pr_description)
        written += [str(diff_path), str(pr_path)]

    return written
