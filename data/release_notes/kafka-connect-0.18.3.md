# Kafka Connect Helm Chart 0.18.3 — Release Notes
**App Version:** 7.8.2 (Confluent Platform 7.8)
**Chart Version:** 0.18.3
**Release Date:** 2026-05-20

## Summary
Patch release. No breaking changes.

## Bug Fixes
- Fixed liveness probe timeout on slow connector startup.
- Improved JVM heap allocation defaults for high-throughput environments.

## Upgrade Notes
- Drop-in replacement for 0.18.2. No configuration changes required.
- Standard rolling upgrade procedure applies.

## Compatibility
| Requirement | Minimum |
|---|---|
| Kubernetes | 1.25 |
| Kafka Broker CP version | 7.6+ |
