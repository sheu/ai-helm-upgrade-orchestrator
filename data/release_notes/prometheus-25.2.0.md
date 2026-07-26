# Prometheus Helm Chart 25.2.0 — Release Notes
**App Version:** 2.52.0
**Chart Version:** 25.2.0
**Release Date:** 2026-06-01

## Summary
Minor release with new recording rules and improved storage configuration.

## Changes
- Added pre-built recording rules for Kubernetes API server latency.
- `server.persistentVolume.size` default increased from 8Gi to 16Gi.
- New `server.retentionSize` field added to cap storage by size rather than time.

## Breaking Changes
None.

## Upgrade Notes
- If you have existing PVCs, they will not be automatically resized.
- Review the new retention size defaults if storage capacity is constrained.

## Compatibility
| Requirement | Minimum |
|---|---|
| Kubernetes | 1.24 |
