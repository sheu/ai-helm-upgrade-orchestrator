"""
Inventory analysis module.

Integration: This module directly adapts the statistical EDA methodology
from Project 2 (Statistical Data Analysis — Bank Fraud Detection). The same
Pandas-based workflow used to profile fraud patterns (distribution analysis,
missing-value detection, groupby aggregations) is here applied to cluster
inventory data to detect version drift, configuration inconsistencies, and
data quality issues.

Reference: Statistical Analysis Project — analysis.ipynb
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .models import (
    ClusterRecord,
    Criticality,
    Environment,
    InventoryReport,
)


# ── Schema validation ─────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {
    "department", "region", "environment", "cluster_name",
    "kubernetes_version", "component", "chart_version",
    "app_version", "values_path", "criticality", "argocd_application",
}


def validate_inventory_schema(df: pd.DataFrame) -> List[str]:
    """
    Validate that the inventory DataFrame has the required columns and no
    critical null values.  Returns a list of data-quality issue strings.

    Statistical approach: applies the same completeness checks used in the
    Project 2 fraud dataset EDA (df.isnull().sum(), column existence checks).
    """
    issues: List[str] = []

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {sorted(missing_cols)}")

    for col in REQUIRED_COLUMNS & set(df.columns):
        null_count = df[col].isnull().sum()
        if null_count > 0:
            issues.append(f"Column '{col}' has {null_count} null value(s)")

    # Check for duplicate cluster+component combos
    if "cluster_name" in df.columns and "component" in df.columns:
        dupes = df.duplicated(subset=["cluster_name", "component"]).sum()
        if dupes > 0:
            issues.append(f"{dupes} duplicate cluster/component record(s) detected")

    return issues


# ── Version distribution analysis ─────────────────────────────────────────────

def summarise_version_distribution(df: pd.DataFrame, component: str) -> Dict[str, int]:
    """
    Return a dict mapping chart_version -> cluster count for the given component.

    Mirrors the groupby + value_counts pattern from the Project 2 fraud analysis
    (merchant_category / fraud_type distributions).
    """
    subset = df[df["component"] == component]
    return subset["chart_version"].value_counts().to_dict()


def detect_configuration_drift(df: pd.DataFrame, component: str) -> List[str]:
    """
    Identify clusters running non-standard chart versions for the component.
    The modal version is considered the standard; all others are flagged.
    """
    subset = df[df["component"] == component]
    if subset.empty:
        return []

    modal_version = subset["chart_version"].mode().iloc[0]
    drifted = subset[subset["chart_version"] != modal_version]
    return drifted["cluster_name"].tolist()


def kubernetes_version_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group clusters by Kubernetes version.  Used to identify which clusters may
    be incompatible with chart version requirements.
    """
    result: Dict[str, List[str]] = {}
    for k8s_ver, group in df.groupby("kubernetes_version"):
        result[str(k8s_ver)] = group["cluster_name"].tolist()
    return result


# ── Upgrade history statistics ─────────────────────────────────────────────────

def analyse_upgrade_history(history_path: str | Path, component: str) -> pd.DataFrame:
    """
    Load and profile past upgrade records for a component.

    Statistical methods adapted from Project 2: descriptive statistics
    (mean, median, std), groupby aggregations, and outcome distributions —
    applied here to upgrade duration, restart counts, and error-rate changes
    rather than transaction amounts and fraud labels.
    """
    df = pd.read_csv(history_path)
    subset = df[df["component"] == component].copy()

    if subset.empty:
        return pd.DataFrame()

    # Numeric summary (analogous to df.describe() in the fraud EDA)
    numeric_cols = [
        "duration_minutes", "pod_restart_count",
        "error_rate_change_pct", "cpu_change_pct", "memory_change_pct",
    ]
    existing = [c for c in numeric_cols if c in subset.columns]
    return subset[existing + ["validation_result", "final_outcome", "rollback_performed"]].copy()


def upgrade_success_rate(history_df: pd.DataFrame) -> float:
    """Return the fraction of upgrades that did not require a rollback."""
    if history_df.empty:
        return float("nan")
    rollbacks = history_df["rollback_performed"].astype(str).str.lower()
    return float((rollbacks == "false").mean())


# ── Main load / filter functions ──────────────────────────────────────────────

def load_cluster_inventory(inventory_path: str | Path) -> pd.DataFrame:
    """Load cluster inventory CSV with basic type coercion."""
    df = pd.read_csv(inventory_path)
    df["environment"] = df["environment"].str.upper()
    return df


def filter_affected_clusters(
    df: pd.DataFrame,
    component: str,
    departments: List[str],
    regions: List[str],
) -> pd.DataFrame:
    """Return inventory rows matching the upgrade request scope."""
    mask = (
        (df["component"] == component)
        & (df["department"].isin(departments))
        & (df["region"].isin(regions))
    )
    return df[mask].copy()


def get_component_dependencies(dep_path: str | Path, component: str) -> List[str]:
    """Return list of components that the given component depends on."""
    df = pd.read_csv(dep_path)
    subset = df[df["component"] == component]
    return subset["dependency"].tolist()


# ── Top-level agent function ──────────────────────────────────────────────────

def run_inventory_analysis(
    request_id: str,
    component: str,
    departments: List[str],
    regions: List[str],
    inventory_path: str | Path,
    history_path: str | Path,
    dep_path: str | Path,
) -> Tuple[InventoryReport, pd.DataFrame]:
    """
    Execute the full inventory analysis and return a structured InventoryReport
    together with the filtered cluster DataFrame for downstream agents.
    """
    df_all = load_cluster_inventory(inventory_path)
    data_quality_issues = validate_inventory_schema(df_all)

    df_filtered = filter_affected_clusters(df_all, component, departments, regions)

    version_dist = summarise_version_distribution(df_filtered, component)
    k8s_groups = kubernetes_version_groups(df_filtered)
    drifted = detect_configuration_drift(df_all, component)

    cluster_records: List[ClusterRecord] = []
    for _, row in df_filtered.iterrows():
        cluster_records.append(
            ClusterRecord(
                department=row["department"],
                region=row["region"],
                environment=Environment(row["environment"]),
                cluster_name=row["cluster_name"],
                kubernetes_version=str(row["kubernetes_version"]),
                component=row["component"],
                chart_version=str(row["chart_version"]),
                app_version=str(row["app_version"]),
                values_path=row["values_path"],
                criticality=Criticality(row["criticality"]),
                argocd_application=row["argocd_application"],
            )
        )

    report = InventoryReport(
        request_id=request_id,
        component=component,
        affected_clusters=cluster_records,
        version_distribution=version_dist,
        kubernetes_versions=k8s_groups,
        inconsistent_versions=drifted,
        data_quality_issues=data_quality_issues,
        total_clusters=len(cluster_records),
    )

    return report, df_filtered
