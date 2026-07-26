"""
Helm chart validation tools.

Wraps helm lint and helm template in a safe, deterministic interface.
The agent may request validation; Python executes it and returns structured
results — the LLM never sees raw command output as a system instruction.

Security rule: any rendered manifest containing patterns matching Kubernetes
Secret values, tokens, or credentials is rejected before being passed to agents.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from .models import GateResult, ValidationResult


# ── Secret detection patterns ─────────────────────────────────────────────────

SECRET_PATTERNS = [
    re.compile(r"kind:\s*Secret", re.IGNORECASE),
    re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"token\s*[:=]\s*[A-Za-z0-9+/]{20,}", re.IGNORECASE),
    re.compile(r"private.?key", re.IGNORECASE),
    re.compile(r"-----BEGIN", re.IGNORECASE),
    re.compile(r"[A-Za-z0-9+/]{64,}={0,2}"),  # Base64 blobs
]

# Values we expect to change between versions
BREAKING_VALUE_PATTERNS = [
    re.compile(r"kafka\.connect\.config\.storage", re.IGNORECASE),  # renamed in 0.19.0
]


def _detect_secrets(rendered: str) -> bool:
    """Return True if any secret patterns are found in rendered output."""
    for pattern in SECRET_PATTERNS:
        if pattern.search(rendered):
            return True
    return False


def _detect_breaking_values(values_content: str, target_version: str) -> List[str]:
    """Detect known deprecated or renamed value keys in an existing values file."""
    issues: List[str] = []
    for pattern in BREAKING_VALUE_PATTERNS:
        if pattern.search(values_content):
            issues.append(
                f"Deprecated value key pattern '{pattern.pattern}' found — "
                f"check rename in chart {target_version} release notes"
            )
    return issues


def validate_yaml(values_path: str | Path) -> Tuple[GateResult, str]:
    """Parse a YAML values file and return PASS/FAIL."""
    try:
        with open(values_path) as f:
            content = f.read()
        yaml.safe_load(content)
        return GateResult.PASS, "YAML parsed successfully"
    except FileNotFoundError:
        return GateResult.FAIL, f"File not found: {values_path}"
    except yaml.YAMLError as e:
        return GateResult.FAIL, f"YAML parse error: {e}"


def helm_lint(chart_path: str | Path) -> Tuple[GateResult, str]:
    """
    Run helm lint on a chart directory.
    Returns (PASS/FAIL, output_summary).
    """
    try:
        result = subprocess.run(
            ["helm", "lint", str(chart_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Return summary — not the full output — to avoid exposing internals
            passed_line = [
                l for l in result.stdout.splitlines() if "chart(s) linted" in l
            ]
            return GateResult.PASS, passed_line[0] if passed_line else "helm lint passed"
        else:
            error_lines = result.stderr.strip().splitlines()[:5]
            return GateResult.FAIL, "; ".join(error_lines)
    except FileNotFoundError:
        return GateResult.FAIL, "helm binary not found"
    except subprocess.TimeoutExpired:
        return GateResult.FAIL, "helm lint timed out"


def helm_template(
    chart_path: str | Path,
    values_path: Optional[str | Path] = None,
    release_name: str = "test-release",
) -> Tuple[GateResult, str, str]:
    """
    Run helm template and return (gate_result, summary, rendered_manifest).
    The rendered manifest is checked for secrets before any further use.
    """
    cmd = ["helm", "template", release_name, str(chart_path)]
    if values_path and Path(values_path).exists():
        cmd += ["-f", str(values_path)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return GateResult.FAIL, f"helm template failed: {result.stderr[:200]}", ""

        rendered = result.stdout

        # Immediately check for secrets
        if _detect_secrets(rendered):
            return GateResult.FAIL, "Secret or credential detected in rendered manifest — rejected", ""

        # Count rendered resource kinds
        kinds = re.findall(r"^kind:\s*(\w+)", rendered, re.MULTILINE)
        summary = f"Rendered {len(kinds)} resource(s): {', '.join(sorted(set(kinds)))}"
        return GateResult.PASS, summary, rendered

    except FileNotFoundError:
        return GateResult.FAIL, "helm binary not found", ""
    except subprocess.TimeoutExpired:
        return GateResult.FAIL, "helm template timed out", ""


def compare_rendered_manifests(current: str, target: str) -> str:
    """
    Produce a human-readable diff summary between two rendered manifests.
    Returns a descriptive string, not a raw diff, to avoid leaking sensitive values.
    """
    if not current or not target:
        return "Comparison unavailable — one or both manifests are empty"

    current_kinds = set(re.findall(r"^kind:\s*(\w+)", current, re.MULTILINE))
    target_kinds = set(re.findall(r"^kind:\s*(\w+)", target, re.MULTILINE))

    added = target_kinds - current_kinds
    removed = current_kinds - target_kinds
    unchanged = current_kinds & target_kinds

    parts = []
    if unchanged:
        parts.append(f"Unchanged resource kinds: {sorted(unchanged)}")
    if added:
        parts.append(f"New resource kinds: {sorted(added)}")
    if removed:
        parts.append(f"Removed resource kinds: {sorted(removed)}")

    lines_current = len(current.splitlines())
    lines_target = len(target.splitlines())
    parts.append(f"Manifest size: {lines_current} → {lines_target} lines")

    return "; ".join(parts)


def check_required_resources(rendered: str, required_kinds: List[str]) -> Tuple[GateResult, str]:
    """Verify that all required Kubernetes resource kinds are present."""
    found_kinds = set(re.findall(r"^kind:\s*(\w+)", rendered, re.MULTILINE))
    missing = [k for k in required_kinds if k not in found_kinds]
    if missing:
        return GateResult.FAIL, f"Missing required resources: {missing}"
    return GateResult.PASS, f"All required resources present: {required_kinds}"


# ── Top-level validation runner ───────────────────────────────────────────────

def run_validation(
    request_id: str,
    cluster_name: str,
    chart_path: str | Path,
    values_path: str | Path,
    target_version: str,
) -> ValidationResult:
    """
    Execute the full validation suite for one cluster and return a
    structured ValidationResult.
    """
    # 1. YAML parse
    yaml_result, yaml_msg = validate_yaml(values_path)

    # 2. Helm lint
    lint_result, lint_msg = helm_lint(chart_path)

    # 3. Helm template (with values)
    tmpl_result, tmpl_msg, rendered = helm_template(chart_path, values_path)

    # 4. Required resources
    required = ["Deployment", "Service"]
    if tmpl_result == GateResult.PASS and rendered:
        res_result, res_msg = check_required_resources(rendered, required)
    else:
        res_result = GateResult.UNKNOWN
        res_msg = "Skipped — template step failed"

    # 5. Secret check (already done in helm_template; mark PASS if no failure)
    secret_result = GateResult.PASS if tmpl_result != GateResult.FAIL or "Secret" not in tmpl_msg else GateResult.FAIL

    # 6. Breaking values
    breaking_vals: List[str] = []
    try:
        with open(values_path) as f:
            vals_content = f.read()
        breaking_vals = _detect_breaking_values(vals_content, target_version)
    except FileNotFoundError:
        pass

    # 7. Diff summary
    diff_summary = tmpl_msg if tmpl_result == GateResult.PASS else "N/A"

    # Overall gate
    gates = [yaml_result, lint_result, tmpl_result, res_result, secret_result]
    if any(g == GateResult.FAIL for g in gates):
        overall = GateResult.FAIL
    elif any(g == GateResult.UNKNOWN for g in gates):
        overall = GateResult.UNKNOWN
    else:
        overall = GateResult.PASS

    return ValidationResult(
        request_id=request_id,
        cluster=cluster_name,
        helm_lint=lint_result,
        helm_template=tmpl_result,
        yaml_parse=yaml_result,
        required_resources_present=res_result,
        secret_check=secret_result,
        breaking_values_detected=breaking_vals,
        rendered_diff_summary=diff_summary,
        overall=overall,
    )
