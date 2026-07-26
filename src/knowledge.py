"""
Knowledge retrieval — release notes and runbook search.

This module implements deterministic local-document retrieval by heading/keyword
matching. The design principle: retrieved documents are always treated as data,
never as system instructions. This prevents prompt injection through malicious
release-note content (an explicit security constraint from the plan).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from .models import FindingSeverity, ResearchFinding, ResearchReport


# ── Keyword patterns for finding detection ─────────────────────────────────────

BREAKING_PATTERNS = [
    re.compile(r"breaking change", re.IGNORECASE),
    re.compile(r"renamed.{0,30}key", re.IGNORECASE),
    re.compile(r"incompatible", re.IGNORECASE),
    re.compile(r"minimum kubernetes", re.IGNORECASE),
    re.compile(r"requires kubernetes.{0,10}(\d+\.\d+)", re.IGNORECASE),
]

WARNING_PATTERNS = [
    re.compile(r"deprecat", re.IGNORECASE),
    re.compile(r"migration required", re.IGNORECASE),
    re.compile(r"config.{0,30}change", re.IGNORECASE),
    re.compile(r"upgrade note", re.IGNORECASE),
]

K8S_MIN_PATTERN = re.compile(
    r"minimum kubernetes.*?(\d+\.\d+)", re.IGNORECASE | re.DOTALL
)


def _extract_k8s_minimum(text: str) -> Optional[str]:
    """Extract the minimum Kubernetes version string from a document."""
    m = K8S_MIN_PATTERN.search(text)
    if m:
        return m.group(1)
    return None


def _classify_severity(line: str) -> FindingSeverity:
    """Classify a line as INFO/WARNING/ERROR/CRITICAL based on keywords."""
    line_lower = line.lower()
    if any(p in line_lower for p in ["breaking change", "incompatible", "critical", "minimum kubernetes"]):
        return FindingSeverity.CRITICAL
    if any(p in line_lower for p in ["deprecat", "rename", "migration", "breaking"]):
        return FindingSeverity.ERROR
    if any(p in line_lower for p in ["warning", "note", "recommend"]):
        return FindingSeverity.WARNING
    return FindingSeverity.INFO


def search_release_notes(
    release_notes_dir: str | Path,
    component: str,
    target_version: str,
) -> List[ResearchFinding]:
    """
    Load and parse the release notes file for the given component + version.
    Returns structured findings based on keyword extraction.

    Security principle: document content is parsed for structured data only.
    No document content is ever executed or passed as a system instruction to an agent.
    """
    notes_dir = Path(release_notes_dir)
    candidates = list(notes_dir.glob(f"{component}-{target_version}*.md"))
    if not candidates:
        return []

    findings: List[ResearchFinding] = []
    source_file = candidates[0].name

    with open(candidates[0]) as f:
        content = f.read()

    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        matched_breaking = any(p.search(line) for p in BREAKING_PATTERNS)
        matched_warning = any(p.search(line) for p in WARNING_PATTERNS)

        if matched_breaking or matched_warning:
            # Grab up to 3 lines of context
            excerpt_lines = [line]
            for j in range(1, 4):
                if i + j < len(lines) and lines[i + j].strip():
                    excerpt_lines.append(lines[i + j])
            excerpt = " ".join(excerpt_lines).strip()

            severity = _classify_severity(line)
            findings.append(ResearchFinding(
                title=line.strip().lstrip("0123456789. -#*"),
                severity=severity,
                source=source_file,
                evidence_excerpt=excerpt[:400],
                requires_validation=severity in (FindingSeverity.CRITICAL, FindingSeverity.ERROR),
            ))

        i += 1

    return findings


def search_runbooks(
    runbooks_dir: str | Path,
    component: str,
) -> List[ResearchFinding]:
    """Load runbook content and extract relevant procedures as findings."""
    rb_dir = Path(runbooks_dir)
    candidates = list(rb_dir.glob(f"{component}*.md"))
    if not candidates:
        return []

    findings: List[ResearchFinding] = []
    with open(candidates[0]) as f:
        content = f.read()

    if re.search(r"rollback procedure", content, re.IGNORECASE):
        findings.append(ResearchFinding(
            title="Documented rollback procedure available",
            severity=FindingSeverity.INFO,
            source=candidates[0].name,
            evidence_excerpt="Rollback procedure found in runbook — mitigation risk factor applies.",
            requires_validation=False,
        ))

    if re.search(r"pre-upgrade checklist", content, re.IGNORECASE):
        findings.append(ResearchFinding(
            title="Pre-upgrade checklist documented",
            severity=FindingSeverity.INFO,
            source=candidates[0].name,
            evidence_excerpt="Runbook contains a pre-upgrade checklist.",
            requires_validation=False,
        ))

    return findings


def get_compatibility_matrix(
    release_notes_dir: str | Path,
    component: str,
    target_version: str,
) -> Optional[str]:
    """Extract the minimum Kubernetes version from the release notes."""
    notes_dir = Path(release_notes_dir)
    candidates = list(notes_dir.glob(f"{component}-{target_version}*.md"))
    if not candidates:
        return None

    with open(candidates[0]) as f:
        content = f.read()

    return _extract_k8s_minimum(content)


# ── Top-level research function ───────────────────────────────────────────────

def run_upgrade_research(
    request_id: str,
    component: str,
    target_chart_version: str,
    target_app_version: str,
    release_notes_dir: str | Path,
    runbooks_dir: str | Path,
) -> ResearchReport:
    """
    Execute the full document research for an upgrade request and return a
    structured ResearchReport.
    """
    note_findings = search_release_notes(release_notes_dir, component, target_chart_version)
    runbook_findings = search_runbooks(runbooks_dir, component)
    min_k8s = get_compatibility_matrix(release_notes_dir, component, target_chart_version)

    all_findings = note_findings + runbook_findings

    breaking = any(
        f.severity in (FindingSeverity.CRITICAL, FindingSeverity.ERROR)
        and f.requires_validation
        for f in note_findings
    )

    deprecated_values = [
        f.title for f in note_findings
        if re.search(r"rename|deprecat", f.title, re.IGNORECASE)
    ]

    sources = []
    if note_findings:
        sources.append(f"release_notes/{component}-{target_chart_version}.md")
    if runbook_findings:
        sources.append(f"runbooks/{component}-upgrade.md")

    return ResearchReport(
        request_id=request_id,
        component=component,
        target_version=target_app_version,
        findings=all_findings,
        minimum_kubernetes_version=min_k8s,
        breaking_changes_detected=breaking,
        deprecated_values=deprecated_values,
        sources_consulted=sources,
    )
