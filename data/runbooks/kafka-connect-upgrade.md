# Kafka Connect Upgrade Runbook

## Pre-upgrade Checklist
1. Verify all connectors are in RUNNING state.
2. Check consumer group lag is within normal bounds.
3. Confirm Kafka broker version compatibility.
4. Review release notes for breaking changes.
5. Ensure monitoring dashboards are active.
6. Prepare rollback version in ArgoCD.

## Upgrade Procedure
1. Apply Helm values to INT environment via ArgoCD.
2. Monitor pod restart count — expected 0–1 rolling restarts.
3. Verify connector status via REST API: GET /connectors/{name}/status
4. Check error rate in Prometheus for 30 minutes.
5. Confirm consumer group lag returns to baseline.
6. If all gates pass, create PR for PROD values update.
7. Await human approval before PROD deployment.

## Rollback Procedure
1. Revert ArgoCD application to previous chart version.
2. Monitor connector recovery (typically < 5 minutes).
3. File incident report if rollback required.

## Known Issues
- Major version upgrades (e.g., CP 7 → CP 8) require config key migration.
- Kubernetes 1.26 or below is incompatible with chart versions 0.19.x+.
