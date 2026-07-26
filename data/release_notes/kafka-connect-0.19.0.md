# Kafka Connect Helm Chart 0.19.0 — Release Notes
**App Version:** 8.0.2 (Confluent Platform 8.0)
**Chart Version:** 0.19.0
**Release Date:** 2026-06-15

## Summary
This is a **major application version upgrade** from Confluent Platform 7.x to 8.0.
It requires careful review of configuration keys and Kafka broker compatibility.

## Breaking Changes
1. **Configuration key renamed**: `kafka.connect.config.storage.replication.factor` has been
   renamed to `connect.config.storage.replication.factor`. Existing Helm values using the old
   key will be silently ignored, causing connectors to use the default replication factor of 1.
2. **Minimum Kubernetes version**: 1.27+ is now required. Clusters running 1.26 or below
   will not schedule pods correctly.
3. **Worker group ID format**: The `group.id` value must now match the pattern
   `^[a-z0-9-]{1,255}$`. Values with underscores will be rejected.

## Deprecations
- `image.pullPolicy: Always` is deprecated; use `image.pullPolicy: IfNotPresent` instead.

## New Features
- Native support for Kafka 3.7 protocol.
- Built-in connector status monitoring endpoint at `/connectors/{name}/status`.
- Improved dead-letter queue (DLQ) configuration.

## Upgrade Notes
- Run `helm diff upgrade` before applying to identify configuration key migrations.
- Validate that all existing connector configurations are compatible with CP 8.0.
- Recommended: Upgrade Kafka brokers to Kafka 3.7 before upgrading Kafka Connect.
- Test all connectors in INT for a minimum 30-minute soak period before promoting.

## Compatibility
| Requirement | Minimum |
|---|---|
| Kubernetes | 1.27 |
| Kafka Broker CP version | 7.8+ |
| Schema Registry CP version | 7.8+ |
