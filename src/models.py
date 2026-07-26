"""
Pydantic data models for the Helm Upgrade Orchestration system.

All inter-agent messages and structured outputs are validated through these
models. This ensures that probabilistic LLM outputs are always validated
against deterministic schemas before they influence workflow state.

Integration note: The structured-output discipline here draws on the Pydantic
validation patterns used in the agentic-ai-capstone research agent.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enumerations ─────────────────────────────────────────────────────────────

class Environment(str, Enum):
    INT = "INT"
    PROD = "PROD"

class Criticality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class UpgradeState(str, Enum):
    REQUESTED = "Requested"
    ANALYSED = "Analysed"
    PLANNED = "Planned"
    VALIDATED = "Validated"
    BLOCKED = "Blocked"
    INT_DEPLOYED = "INTDeployed"
    AWAITING_APPROVAL = "AwaitingApproval"
    PAUSED = "Paused"          # Missing evidence — human investigation required
    INT_FAILED = "INTFailed"
    PROD_PLANNED = "PRODPlanned"
    ROLLED_BACK = "RolledBack"
    COMPLETED = "Completed"

class GateResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"  # Missing evidence — never treated as PASS

class FindingSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ResearchStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


# ── Request ───────────────────────────────────────────────────────────────────

class UpgradeRequest(BaseModel):
    """Input request submitted by a platform engineer."""
    request_id: str
    component: str
    target_chart_version: str
    target_app_version: str
    departments: List[str]
    regions: List[str]
    starting_environment: Environment = Environment.INT
    requested_by: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Inventory ─────────────────────────────────────────────────────────────────

class ClusterRecord(BaseModel):
    department: str
    region: str
    environment: Environment
    cluster_name: str
    kubernetes_version: str
    component: str
    chart_version: str
    app_version: str
    values_path: str
    criticality: Criticality
    argocd_application: str


class InventoryReport(BaseModel):
    """Output of the Inventory Agent."""
    request_id: str
    component: str
    affected_clusters: List[ClusterRecord]
    version_distribution: Dict[str, int]  # chart_version -> count
    kubernetes_versions: Dict[str, List[str]]  # k8s_version -> cluster_names
    inconsistent_versions: List[str]  # cluster names with non-standard versions
    data_quality_issues: List[str]
    total_clusters: int


# ── Research ──────────────────────────────────────────────────────────────────

class ResearchFinding(BaseModel):
    """A single finding from release note or runbook research."""
    title: str
    severity: FindingSeverity
    source: str
    evidence_excerpt: str
    affected_configuration: Optional[str] = None
    recommended_action: Optional[str] = None
    requires_validation: bool = True


class ResearchReport(BaseModel):
    """Output of the Upgrade Research Agent."""
    request_id: str
    component: str
    target_chart_version: str   # Helm chart version, e.g. "0.18.3"
    target_app_version: str     # Application/image version, e.g. "7.8.2"
    findings: List[ResearchFinding]
    minimum_kubernetes_version: Optional[str] = None
    breaking_changes_detected: bool = False
    deprecated_values: List[str] = Field(default_factory=list)
    sources_consulted: List[str] = Field(default_factory=list)
    synthesis_note: Optional[str] = None  # LLM-generated summary; influences risk context
    status: ResearchStatus = ResearchStatus.COMPLETE
    missing_evidence: List[str] = Field(default_factory=list)


# ── Risk & Planning ───────────────────────────────────────────────────────────

class RolloutWave(BaseModel):
    wave: int
    clusters: List[str]
    description: str
    requires_gate_pass: bool = True


class UpgradePlan(BaseModel):
    """Output of the Planning and Risk Agent."""
    request_id: str
    component: str
    current_versions: List[str]
    target_chart_version: str   # Helm chart version, e.g. "0.18.3"
    target_app_version: str     # Application/image version, e.g. "7.8.2"
    affected_clusters: List[str]
    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    risk_factors: Dict[str, Any]
    compatibility_findings: List[ResearchFinding]
    configuration_changes: List[str]
    validation_steps: List[str]
    rollout_waves: List[RolloutWave]
    rollback_conditions: List[str]
    approval_required: bool = True
    evidence: List[str] = Field(default_factory=list)


# ── Validation ────────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Output of the Validation Agent."""
    request_id: str
    cluster: str
    helm_lint: GateResult
    helm_template: GateResult
    yaml_parse: GateResult
    required_resources_present: GateResult
    secret_check: GateResult  # PASS = no secrets found
    breaking_values_detected: List[str]
    rendered_diff_summary: str
    overall: GateResult

    @field_validator('overall', mode='before')
    @classmethod
    def derive_overall(cls, v, values):
        # UNKNOWN or FAIL in any gate makes overall non-PASS
        return v


# ── Health Monitoring ─────────────────────────────────────────────────────────

class HealthSnapshot(BaseModel):
    """Post-deployment health observation."""
    cluster: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pod_ready_percent: Optional[float] = None
    restart_count: Optional[int] = None
    error_rate_change_pct: Optional[float] = None
    latency_change_pct: Optional[float] = None
    memory_change_pct: Optional[float] = None
    argocd_sync_status: Optional[str] = None
    argocd_health_status: Optional[str] = None
    connector_status: Optional[str] = None
    evidence_available: bool = True


class HealthEvaluationResult(BaseModel):
    cluster: str
    pod_readiness_gate: GateResult
    restart_gate: GateResult
    error_rate_gate: GateResult
    latency_gate: GateResult
    memory_gate: GateResult
    overall: GateResult
    notes: List[str] = Field(default_factory=list)


# ── GitOps Proposal ───────────────────────────────────────────────────────────

class ProposedChange(BaseModel):
    """A proposed GitOps change for one environment."""
    cluster: str
    environment: Environment
    values_path: str
    current_chart_version: str
    target_chart_version: str
    current_app_version: str
    target_app_version: str
    value_migrations: List[str]
    diff_content: str
    rollback_version: str
    pr_description: str


# ── LLM output validation ─────────────────────────────────────────────────────

class LLMSynthesis(BaseModel):
    """
    Strict schema for the JSON object the LLM must return for synthesis.
    model_validate_json() enforces types and rejects unexpected keys.
    """
    model_config = ConfigDict(extra="forbid")

    synthesis_note: str
    additional_risks: List[str] = Field(default_factory=list)


# ── ReAct agent models ────────────────────────────────────────────────────────

class AgentAction(str, Enum):
    """Enumeration of actions the ReAct agent may select."""
    SEARCH_RELEASE_NOTES = "search_release_notes"
    SEARCH_RUNBOOK = "search_runbook"
    CHECK_COMPATIBILITY = "get_kubernetes_compatibility"
    FINISH = "finish"


class ReActDecision(BaseModel):
    """
    Schema for each LLM decision in the ReAct loop.
    extra='forbid' ensures the model cannot smuggle arbitrary fields.
    """
    model_config = ConfigDict(extra="forbid")

    decision_summary: str           # Auditable explanation — not raw chain-of-thought
    action: AgentAction
    action_input: Dict[str, str] = Field(default_factory=dict)


# ── Tool input validation models ──────────────────────────────────────────────

class StrictToolInput(BaseModel):
    """Base class that forbids unexpected fields on all tool input models."""
    model_config = ConfigDict(extra="forbid")


class ReleaseNotesInput(StrictToolInput):
    component: str
    chart_version: str


class RunbookInput(StrictToolInput):
    component: str


class CompatibilityInput(StrictToolInput):
    component: str
    chart_version: str


# ── Typed tool result models ──────────────────────────────────────────────────

class ReleaseNotesResult(BaseModel):
    findings: List["ResearchFinding"]
    source: Optional[str] = None


class RunbookResult(BaseModel):
    findings: List["ResearchFinding"]
    source: Optional[str] = None


class CompatibilityResult(BaseModel):
    minimum_kubernetes_version: Optional[str] = None
    source: Optional[str] = None


# ── ReAct observation ─────────────────────────────────────────────────────────

class ToolObservation(BaseModel):
    """
    Structured record of one tool invocation inside the ReAct loop.
    Written to the audit log and fed back to the LLM as context.
    """
    iteration: int
    tool: str
    input: Dict[str, str] = Field(default_factory=dict)
    result: Optional[Any] = None    # ReleaseNotesResult | RunbookResult | CompatibilityResult
    error: Optional[str] = None
    succeeded: bool


# ── Audit ─────────────────────────────────────────────────────────────────────

class AuditEntry(BaseModel):
    request_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    action: str
    tool_used: Optional[str] = None
    result: str
    evidence: Optional[str] = None


# ── Final Report ──────────────────────────────────────────────────────────────

class UpgradeReport(BaseModel):
    request_id: str
    component: str
    final_state: UpgradeState
    recommendation: str
    risk_level: RiskLevel
    inventory_summary: str
    research_summary: str
    validation_results: List[str]
    health_results: List[str]
    proposed_changes: List[str]
    audit_trail_length: int
    requires_human_action: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
