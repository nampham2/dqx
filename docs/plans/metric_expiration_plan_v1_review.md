# Architectural Review: Metric Expiration Implementation Plan v1

**Review Date**: October 30, 2024
**Reviewer**: Claude (Assistant Architect)
**Plan**: metric_expiration_plan_v1.md
**Status**: APPROVED WITH RECOMMENDATIONS

## Executive Summary

The metric expiration implementation plan introduces automatic cleanup of expired metrics based on TTL (Time To Live) values. The feature is well-designed and follows DQX's architectural patterns, but requires several improvements around default behavior, consistency, and performance before implementation.

## Architecture Assessment

### Alignment with DQX Principles

✅ **KISS/YAGNI**: Simple solution that solves the immediate need without over-engineering
✅ **Protocol-Based Extensions**: Uses existing patterns without requiring inheritance
✅ **Type Safety**: Maintains full type hints and mypy validation
✅ **Error Handling**: Uses Result types and non-blocking error handling
✅ **Testing**: Follows TDD with comprehensive test coverage planned

### Design Strengths

1. **Minimal Schema Impact**
   - Leverages existing `metadata` field with `ttl_hours` property
   - No database migrations required
   - Clean backward compatibility

2. **Clear Separation of Concerns**
   - `get_expired_metrics_stats()`: Analytics and reporting
   - `delete_expired_metrics()`: Actual cleanup operation
   - VerificationSuite: Orchestration only

3. **Robust Error Handling**
   - Cleanup failures don't break suite execution
   - Errors are logged for operational visibility
   - Graceful degradation ensures continuity

4. **Configurability**
   - Feature can be disabled via constructor parameter
   - Testable with `current_time` parameter injection
   - Clear API with sensible defaults

## Critical Issues

### 1. Race Condition Risk 🔴

**Problem**: Separate stats and delete queries could see different data if metrics are inserted concurrently.

```python
# Current approach (problematic)
stats = db.get_expired_metrics_stats()  # Query 1
# Metrics could be inserted here!
db.delete_expired_metrics()              # Query 2 sees different data
```

**Solution**: Use transactional consistency:

```python
def cleanup_expired_metrics(self) -> tuple[dict[dt.date, int], int]:
    """Get stats and delete in single transaction."""
    with self._mutex:
        session = self.new_session()
        with session.begin():
            # Get stats
            stats = self._get_expired_stats_locked(session)
            # Delete and get count
            deleted = self._delete_expired_locked(session)
            session.commit()
        return stats, deleted
```

### 2. Overly Aggressive Default Behavior 🔴

**Problem**: Deleting all metrics without TTL by default violates principle of least surprise.

**Current Logic**:
- No metadata → DELETE
- No ttl_hours → DELETE
- ttl_hours present → Check expiration

**Recommended Logic**:
- No metadata → KEEP (backward compatibility)
- ttl_hours = 0 → DELETE (explicit opt-in for cleanup)
- ttl_hours > 0 → Check expiration
- ttl_hours = None → KEEP (no TTL means keep forever)

### 3. Performance Concerns 🟡

**Problem**: No indexes for efficient expiration queries on large databases.

**Solution**: Add migration for performance:

```sql
CREATE INDEX idx_metric_expiration
ON dq_metric(created, json_extract(meta, '$.ttl_hours'));
```

Consider also:
- Batched deletes with LIMIT clauses
- Monitoring query performance
- Configurable batch sizes

## Recommendations

### High Priority

1. **Fix Race Condition**
   - Implement transactional cleanup as shown above
   - Return deletion count for verification
   - Add tests for concurrent operations

2. **Change Default Behavior**
   - Only delete metrics with explicit `ttl_hours=0`
   - Add `aggressive_cleanup` parameter for old behavior
   - Update documentation to clarify retention rules

3. **Use UTC Consistently**
   - Replace `datetime.now()` with `datetime.utcnow()`
   - Document timezone assumptions
   - Add timezone tests

### Medium Priority

4. **Add Performance Optimizations**
   - Create database indexes for expiration queries
   - Implement batched deletes for large cleanups
   - Add performance benchmarks

5. **Improve Operational Flexibility**
   - Create standalone cleanup utilities
   - Add CLI command: `dqx cleanup-metrics`
   - Enable scheduled cleanup outside suite runs

6. **Enhance Monitoring**
   - Use timer registry for cleanup duration
   - Add structured logging with counts
   - Expose cleanup metrics for observability

### Low Priority

7. **Future Enhancements**
   - Per-dataset retention policies
   - Archival instead of deletion option
   - Cleanup based on custom criteria

## Implementation Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Data loss from aggressive defaults | High | Change default behavior, add warnings |
| Performance degradation | Medium | Add indexes, batch operations |
| Timezone bugs | Medium | Use UTC everywhere |
| Concurrent operation issues | Low | Use transactions, add mutex |

## Test Coverage Gaps

The plan has good test coverage, but consider adding:

1. **Concurrency Tests**
   - Simultaneous metric insertion and cleanup
   - Multiple cleanup operations

2. **Performance Tests**
   - Large dataset cleanup benchmarks
   - Index effectiveness validation

3. **Timezone Tests**
   - Database in different timezone
   - DST boundary conditions

## Migration Path

For existing DQX users:

1. **Phase 1**: Deploy with cleanup disabled by default
   ```python
   # Opt-in for early adopters
   VerificationSuite(checks, db, "Suite", cleanup_expired_metrics=True)
   ```

2. **Phase 2**: Add warnings for metrics without TTL
   ```python
   logger.warning(f"Found {count} metrics without TTL - consider setting ttl_hours")
   ```

3. **Phase 3**: Enable by default with safe behavior
   - Only delete metrics with `ttl_hours=0`
   - Document in changelog

## Conclusion

The metric expiration plan is **approved with recommendations**. The core design is sound and aligns with DQX's architecture. However, the identified issues must be addressed before implementation to ensure data safety and system reliability.

### Next Steps

1. Update implementation plan with race condition fixes
2. Revise default behavior to be less aggressive
3. Add performance optimizations
4. Implement with recommended changes
5. Deploy in phases with careful monitoring

The feature will be a valuable addition to DQX once these improvements are incorporated.
