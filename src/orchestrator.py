"""
Upgrade Coordinator — multi-agent orchestration.

Integration: The sequential multi-agent coordination pattern here is adapted
from the agentic-ai-beaver-choice-project, which implemented a four-agent
workflow for paper supply quoting. Here the same principle — structured
inter-agent message passing, state-machine progression, and coordinated
decision-making — is applied to Kubernetes platform upgrade orchestration.

Key difference: the Coordinator controls state transitions. The LLM must not
invent or skip workflow states. Only deterministic gate results may advance
the state machine to Validated, AwaitingApproval, or Completed.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .agents import build_research_agent
from .gitops import generate_proposed_change, write_proposed_changes
from .helm_tools import run_validation
from .inventory import run_inventory_analysis, analyse_upgrade_history, upgrade_success_rate
from .models import (
    AuditEntry,
    ClusterRecord,
    Environment,
    GateResult,
    HealthSnapshot,
    ResearchStatus,
    RiskLevel,
    UpgradePlan,
    UpgradeReport,
    UpgradeRequest,
    UpgradeState,
)
from .monitoring import evaluate_health, load_quality_gates
from .reporting import append_audit_entry, build_upgrade_report, format_report_markdown
from .risk import build_upgrade_plan

logger = logging.getLogger(__name__)


class UpgradeCoordinator:
    """
    Orchestrates the full upgrade workflow across one LLM-assisted research
    agent and three deterministic analytical/validation components:

    1. Inventory analysis  — cluster discovery and version EDA (deterministic)
    2. Research Agent      — release note and runbook research with LLM synthesis
    3. Planning/Risk       — risk scoring and upgrade plan construction (deterministic)
    4. Validation          — Helm lint, template, and value checking (deterministic)

    The coordinator enforces the state machine and quality gates.
    Components communicate through structured Pydantic models, not raw text.

    Design reference: agentic-ai-beaver-choice-project sequential agent pattern.
    """

    def __init__(
        self,
        project_root: str | Path,
        llm_client=None,
        verbose: bool = True,
    ):
        self.root = Path(project_root)
        self.llm_client = llm_client
        self.verbose = verbose

        # Paths
        self.inventory_path = self.root / "data" / "cluster_inventory.csv"
        self.history_path = self.root / "data" / "upgrade_history.csv"
        self.dep_path = self.root / "data" / "component_dependencies.csv"
        self.release_notes_dir = self.root / "data" / "release_notes"
        self.runbooks_dir = self.root / "data" / "runbooks"
        self.risk_rules_path = self.root / "config" / "risk_rules.yaml"
        self.quality_gates_path = self.root / "config" / "quality_gates.yaml"
        self.output_dir = self.root / "outputs"
        self.output_dir.mkdir(exist_ok=True)

        self.state = UpgradeState.REQUESTED
        self.audit_log_path = self.output_dir / "audit_log.jsonl"

    def _log(self, agent: str, action: str, result: str,
             tool: Optional[str] = None, evidence: Optional[str] = None):
        entry = AuditEntry(
            request_id=getattr(self, '_request_id', 'unknown'),
            agent=agent,
            action=action,
            tool_used=tool,
            result=result[:500],
            evidence=(evidence or "")[:200] if evidence else None,
        )
        append_audit_entry(entry, self.audit_log_path)
        if self.verbose:
            print(f"  [{agent}] {action}: {result[:100]}")

    def _transition(self, new_state: UpgradeState, reason: str):
        """Advance the state machine — only deterministic code calls this."""
        old = self.state.value
        self.state = new_state
        self._log("Coordinator", "state_transition",
                  f"{old} → {new_state.value}: {reason}")

    def run(
        self,
        request: UpgradeRequest,
        health_snapshot: Optional[HealthSnapshot] = None,
    ) -> UpgradeReport:
        """
        Execute the full upgrade orchestration workflow for a given request.

        Returns an UpgradeReport with the final recommendation.
        """
        self._request_id = request.request_id
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"UPGRADE ORCHESTRATION: {request.component}")
            print(f"Request: {request.request_id}")
            print(f"Target: {request.target_chart_version} (app {request.target_app_version})")
            print(f"Scope: {request.departments} × {request.regions}")
            print(f"{'='*60}")

        gates_config = load_quality_gates(self.quality_gates_path)

        # ── Phase 1: Inventory Analysis ───────────────────────────────────────
        print("\n[Phase 1] Inventory Agent — analysing cluster inventory...")
        inventory_report, df_clusters = run_inventory_analysis(
            request_id=request.request_id,
            component=request.component,
            departments=request.departments,
            regions=request.regions,
            inventory_path=self.inventory_path,
            history_path=self.history_path,
            dep_path=self.dep_path,
        )

        if inventory_report.total_clusters == 0:
            self._transition(UpgradeState.BLOCKED,
                             "No affected clusters found matching request scope")
            return self._build_report(
                request, inventory_report, None, [], [], [],
                "No clusters found for the requested scope.",
                UpgradeState.BLOCKED, RiskLevel.LOW, requires_human=False
            )

        self._log("InventoryAgent", "inventory_complete",
                  f"{inventory_report.total_clusters} cluster(s) identified",
                  evidence=str(inventory_report.version_distribution))
        self._transition(UpgradeState.ANALYSED, f"Inventory complete: {inventory_report.total_clusters} clusters")

        # ── Phase 2: Upgrade Research ─────────────────────────────────────────
        print("\n[Phase 2] Research Agent — analysing release notes and runbooks...")
        research_agent = build_research_agent(
            request_id=request.request_id,
            audit_log_path=self.audit_log_path,
            release_notes_dir=self.release_notes_dir,
            runbooks_dir=self.runbooks_dir,
            llm_client=self.llm_client,
        )
        research_report = research_agent.run(request)

        # ── Research completeness gate ────────────────────────────────────────
        if research_report.status == ResearchStatus.INCOMPLETE:
            self._transition(
                UpgradeState.PAUSED,
                f"Upgrade research incomplete — missing: {research_report.missing_evidence}",
            )
            return self._build_report(
                request, inventory_report, None, [], [], [],
                (
                    "PAUSED: Mandatory upgrade evidence was not collected by the research agent. "
                    f"Missing evidence: {research_report.missing_evidence}. "
                    "Human investigation is required before proceeding."
                ),
                UpgradeState.PAUSED, RiskLevel.UNKNOWN, requires_human=True
            )

        # ── Phase 3: Risk Scoring & Planning ─────────────────────────────────
        print("\n[Phase 3] Planning Agent — calculating risk score and upgrade plan...")
        history_df = analyse_upgrade_history(self.history_path, request.component)
        success_rate = upgrade_success_rate(history_df)

        plan = build_upgrade_plan(
            request_id=request.request_id,
            inventory_report=inventory_report,
            research_report=research_report,
            history_success_rate=success_rate,
            risk_rules_path=self.risk_rules_path,
        )

        self._log("PlanningAgent", "plan_created",
                  f"Risk: {plan.risk_score}/100 ({plan.risk_level.value.upper()})",
                  evidence=f"Factors: {list(plan.risk_factors.keys())}")
        self._transition(UpgradeState.PLANNED, f"Risk score: {plan.risk_score}/100 ({plan.risk_level.value})")

        # ── Kubernetes incompatibility gate (hard block) ──────────────────────
        if research_report.minimum_kubernetes_version:
            from .risk import is_kubernetes_compatible
            incompatible = [
                c for c in inventory_report.affected_clusters
                if not is_kubernetes_compatible(
                    c.kubernetes_version, research_report.minimum_kubernetes_version
                )
            ]
            if incompatible:
                cluster_list = [c.cluster_name for c in incompatible]
                self._transition(UpgradeState.BLOCKED,
                                 f"K8s incompatibility: {cluster_list}")
                return self._build_report(
                    request, inventory_report, plan, [], [], [],
                    f"BLOCKED: Kubernetes version incompatibility. "
                    f"Chart requires K8s >={research_report.minimum_kubernetes_version}. "
                    f"Incompatible clusters: {cluster_list}. "
                    f"Upgrade Kubernetes before proceeding.",
                    UpgradeState.BLOCKED, plan.risk_level, requires_human=True
                )

        # ── Risk gate for automatic INT ───────────────────────────────────────
        max_auto_risk = gates_config["global"]["maximum_risk_for_automatic_int"]
        if plan.risk_score > max_auto_risk:
            self._transition(
                UpgradeState.AWAITING_APPROVAL,
                f"Risk score {plan.risk_score} exceeds automatic INT threshold ({max_auto_risk}) — human approval required"
            )
            return self._build_report(
                request, inventory_report, plan, [], [], [],
                f"AWAITING_APPROVAL: Risk score {plan.risk_score}/100 exceeds the automatic INT "
                f"threshold of {max_auto_risk}. Human approval is required before INT deployment "
                f"may proceed. Risk level: {plan.risk_level.value.upper()}.",
                UpgradeState.AWAITING_APPROVAL, plan.risk_level, requires_human=True
            )

        # ── Phase 4: Validation ───────────────────────────────────────────────
        print("\n[Phase 4] Validation Agent — helm lint, template, and value checks...")
        chart_path = self.root / "charts" / request.component
        validation_results = []

        # Validate one representative cluster (INT first)
        int_clusters = [
            c for c in inventory_report.affected_clusters
            if c.environment == Environment.INT
        ]
        representative_clusters = int_clusters[:1] or inventory_report.affected_clusters[:1]

        for cluster in representative_clusters:
            values_path = self.root / cluster.values_path
            val_result = run_validation(
                request_id=request.request_id,
                cluster_name=cluster.cluster_name,
                chart_path=chart_path,
                values_path=values_path,
                target_version=request.target_chart_version,
            )
            validation_results.append(val_result)
            self._log("ValidationAgent", "validation_complete",
                      f"{cluster.cluster_name}: {val_result.overall.value}",
                      tool="helm_lint+helm_template",
                      evidence=f"lint={val_result.helm_lint.value}, "
                               f"template={val_result.helm_template.value}, "
                               f"breaking_vals={val_result.breaking_values_detected}")

        # Check overall validation gate
        if any(v.overall == GateResult.FAIL for v in validation_results):
            self._transition(UpgradeState.BLOCKED, "Static validation failed")
            val_summary = [
                f"{v.cluster}: {v.overall.value} "
                f"(lint={v.helm_lint.value}, template={v.helm_template.value})"
                for v in validation_results
            ]
            return self._build_report(
                request, inventory_report, plan, val_summary, [], [],
                "BLOCKED: Static validation failed. Review helm lint and template errors.",
                UpgradeState.BLOCKED, plan.risk_level, requires_human=True
            )

        self._transition(UpgradeState.VALIDATED, "Static validation passed")

        # ── Phase 5: INT health evaluation ────────────────────────────────────
        print("\n[Phase 5] Health evaluation — evaluating INT deployment metrics...")
        health_result = evaluate_health(health_snapshot, gates_config)
        self._log("Coordinator", "health_evaluation",
                  f"Health gate: {health_result.overall.value}",
                  evidence=f"Notes: {health_result.notes}")

        # Determine INT outcome
        if health_result.overall == GateResult.FAIL:
            self._transition(UpgradeState.INT_FAILED, "INT health gates failed")
            health_summary = [
                f"Overall: {health_result.overall.value}",
                *health_result.notes,
            ]
            return self._build_report(
                request, inventory_report, plan,
                [f"{v.cluster}: {v.overall.value}" for v in validation_results],
                health_summary, [],
                "ROLLBACK RECOMMENDED: INT deployment failed health gates. "
                "Do not promote to PROD.",
                UpgradeState.INT_FAILED, plan.risk_level, requires_human=True
            )

        if health_result.overall == GateResult.UNKNOWN:
            self._transition(UpgradeState.PAUSED,
                             "Missing monitoring evidence — UNKNOWN state, human investigation required")
            return self._build_report(
                request, inventory_report, plan,
                [f"{v.cluster}: {v.overall.value}" for v in validation_results],
                [f"Health: UNKNOWN — {'; '.join(health_result.notes)}"], [],
                "PAUSED: Monitoring evidence unavailable. "
                "UNKNOWN is not PASS. Human investigation required before any promotion decision.",
                UpgradeState.PAUSED, plan.risk_level, requires_human=True
            )

        # INT passed — transition to awaiting human approval for PROD
        self._transition(UpgradeState.INT_DEPLOYED, "INT health gates passed")

        # ── Phase 6: GitOps change generation ─────────────────────────────────
        print("\n[Phase 6] Generating proposed GitOps changes...")
        proposed_changes = []
        for cluster in inventory_report.affected_clusters:
            try:
                change = generate_proposed_change(
                    cluster=cluster,
                    plan=plan,
                    research=research_report,
                    base_project_path=self.root,
                )
                proposed_changes.append(change)
            except Exception as e:
                self._log("Coordinator", "gitops_warning",
                          f"Could not generate change for {cluster.cluster_name}: {e}")

        written_files = write_proposed_changes(proposed_changes, self.output_dir)
        self._log("Coordinator", "gitops_generated",
                  f"{len(written_files)} file(s) written to outputs/proposed_changes/")

        self._transition(UpgradeState.AWAITING_APPROVAL,
                         "PROD changes proposed — awaiting human approval")

        # ── Final report ──────────────────────────────────────────────────────
        val_summary = [
            f"{v.cluster}: {v.overall.value} (lint={v.helm_lint.value}, "
            f"template={v.helm_template.value}, breaking={v.breaking_values_detected})"
            for v in validation_results
        ]
        health_summary = [
            f"Overall: {health_result.overall.value}",
            *health_result.notes,
        ]

        return self._build_report(
            request, inventory_report, plan, val_summary, health_summary,
            written_files,
            "AWAITING HUMAN APPROVAL: INT validation passed. "
            "Review proposed changes and approve PROD deployment.",
            UpgradeState.AWAITING_APPROVAL, plan.risk_level, requires_human=True
        )

    def _build_report(
        self,
        request: UpgradeRequest,
        inventory_report,
        plan: Optional[UpgradePlan],
        val_summary: List[str],
        health_summary: List[str],
        written_files: List[str],
        recommendation: str,
        final_state: UpgradeState,
        risk_level: RiskLevel,
        requires_human: bool,
    ) -> UpgradeReport:

        inventory_summary = (
            f"{inventory_report.total_clusters} cluster(s) affected. "
            f"Version distribution: {inventory_report.version_distribution}. "
            f"Data quality issues: {inventory_report.data_quality_issues or 'none'}."
        )

        research_summary = "No release notes found for this version." if plan is None else (
            f"Chart {plan.target_chart_version} / app {plan.target_app_version}. "
            f"Risk score: {plan.risk_score}/100 ({plan.risk_level.value.upper()}). "
            f"Breaking changes: {plan.compatibility_findings[0].title if plan.compatibility_findings else 'none'}. "
            f"Min K8s: {plan.risk_factors.get('kubernetes_incompatibility', {}).get('evidence', 'not specified')}."
        )

        report = build_upgrade_report(
            request_id=request.request_id,
            component=request.component,
            final_state=final_state,
            recommendation=recommendation,
            risk_level=risk_level,
            inventory_summary=inventory_summary,
            research_summary=research_summary,
            validation_results=val_summary or ["No validation run"],
            health_results=health_summary or ["No health evaluation"],
            proposed_changes=written_files or ["No changes proposed"],
            audit_log_path=self.audit_log_path,
            requires_human_action=requires_human,
        )

        # Write the report
        report_md = format_report_markdown(report)
        report_path = self.output_dir / f"upgrade_report_{request.request_id}.md"
        report_path.write_text(report_md)

        plan_path = self.output_dir / "upgrade_plan.json"
        if plan:
            plan_path.write_text(plan.model_dump_json(indent=2))

        return report
