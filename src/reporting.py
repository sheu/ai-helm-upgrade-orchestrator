"""
Audit logging and reporting.

Every agent action, tool call, and decision is recorded in a JSONL audit log.
The audit trail is the mechanism of accountability — it makes the AI system's
reasoning transparent and auditable by the platform engineer.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from .models import AuditEntry, UpgradeReport, UpgradeState, RiskLevel


def append_audit_entry(entry: AuditEntry, log_path: str | Path) -> None:
    """Append a single audit entry as a JSON line."""
    with open(log_path, "a") as f:
        f.write(entry.model_dump_json() + "\n")


def load_audit_log(log_path: str | Path) -> List[AuditEntry]:
    """Load all audit entries from a JSONL file."""
    path = Path(log_path)
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(AuditEntry.model_validate_json(line))
    return entries


def build_upgrade_report(
    request_id: str,
    component: str,
    final_state: UpgradeState,
    recommendation: str,
    risk_level: RiskLevel,
    inventory_summary: str,
    research_summary: str,
    validation_results: List[str],
    health_results: List[str],
    proposed_changes: List[str],
    audit_log_path: str | Path,
    requires_human_action: bool = True,
) -> UpgradeReport:
    """Build and return the final upgrade report."""
    entries = load_audit_log(audit_log_path)
    return UpgradeReport(
        request_id=request_id,
        component=component,
        final_state=final_state,
        recommendation=recommendation,
        risk_level=risk_level,
        inventory_summary=inventory_summary,
        research_summary=research_summary,
        validation_results=validation_results,
        health_results=health_results,
        proposed_changes=proposed_changes,
        audit_trail_length=len(entries),
        requires_human_action=requires_human_action,
    )


def format_report_markdown(report: UpgradeReport) -> str:
    """Format the upgrade report as a Markdown document."""
    lines = [
        f"# Upgrade Report — {report.component}",
        f"**Request ID:** {report.request_id}",
        f"**Generated:** {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Final State:** {report.final_state.value}",
        f"**Risk Level:** {report.risk_level.value.upper()}",
        f"**Recommendation:** {report.recommendation}",
        f"**Requires Human Action:** {'YES' if report.requires_human_action else 'NO'}",
        "",
        "## Inventory Summary",
        report.inventory_summary,
        "",
        "## Research Summary",
        report.research_summary,
        "",
        "## Validation Results",
        *[f"- {r}" for r in report.validation_results],
        "",
        "## Health Results",
        *[f"- {r}" for r in report.health_results],
        "",
        "## Proposed Changes",
        *[f"- {c}" for c in report.proposed_changes],
        "",
        f"## Audit Trail",
        f"{report.audit_trail_length} action(s) recorded.",
    ]
    return "\n".join(lines)
