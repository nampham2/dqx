# Metric Expiration Refactoring Summary

## Overview
This refactoring improves the metric expiration implementation by extracting common SQL filter logic into a helper method, integrating metric cleanup into the VerificationSuite workflow, and providing visibility into cleanup operations through the plugin system.

## Key Changes

### 1. Helper Method for Expiration Filter
- Created `_build_expiration_filter()` in `MetricRepository` to eliminate duplicated SQL conditions
- Both `get_metrics_stats()` and `delete_expired_metrics()` now use this shared helper
- Improves maintainability and reduces risk of inconsistencies

### 2. MetricStats Dataclass
```python
@dataclass(frozen=True)
class MetricStats:
    """Statistics about metrics in the database."""
    total_metrics: int
    expired_metrics: int
```
- Simple, immutable data structure for metric statistics
- Replaces previous tuple return from `get_expired_metrics_stats()`

### 3. API Improvements
- Renamed `get_expired_metrics_stats()` to `get_metrics_stats()` for clarity
- Returns `MetricStats` dataclass instead of tuple
- Maintains backward compatibility through the refactoring

### 4. VerificationSuite Integration
- Added automatic metric cleanup before analysis in `_analyze()`
- Created cached `_metrics_stats` property to avoid duplicate DB queries
- Ensures expired metrics are cleaned up before running checks

### 5. Plugin System Enhancement
- Made `metrics_stats` a required field in `PluginExecutionContext`
- All plugins now have access to metric cleanup statistics
- Enhanced audit plugin to display cleanup information:
  ```
  Metrics Cleanup: 150 expired metrics removed
  ```

### 6. Test Updates
- Updated all test files that create `PluginExecutionContext` instances
- Added proper `MetricStats` objects to maintain test compatibility
- All 850+ tests passing successfully

## Benefits

1. **Code Quality**: Eliminated duplication in SQL filter logic
2. **Performance**: Cached metrics stats to avoid repeated DB queries
3. **Visibility**: Users can now see metric cleanup activity in audit reports
4. **Integration**: Metric cleanup is seamlessly integrated into the analysis workflow
5. **Type Safety**: Stronger typing with dedicated dataclass

## Example Usage

When running a verification suite:
```python
suite = VerificationSuite()
result = suite.analyze()  # Automatically cleans up expired metrics first
```

The audit plugin output now includes:
```
═══ DQX Audit Report ═══
Suite: Test Suite
Date: 2025-10-30
Tags: none
Duration: 100.00ms
Datasets: ds1

Execution Summary:
  Assertions: 0 total, 0 passed (0.0%), 0 failed (0.0%)
  Metrics Cleanup: 150 expired metrics removed
══════════════════════
```

## Technical Details

### SQL Filter Helper
```python
def _build_expiration_filter(self, cutoff_time: datetime) -> Any:
    """Build the SQL filter for expired metrics."""
    return and_(
        self.model.yyyy_mm_dd < cutoff_time.date(),
        func.json_extract(self.model.metadata_json, "$.ttl_hours").cast(Integer) > 0,
        (func.julianday(cutoff_time) - func.julianday(self.model.yyyy_mm_dd)) * 24
        > func.json_extract(self.model.metadata_json, "$.ttl_hours").cast(Integer),
    )
```

This helper encapsulates the complex SQL logic for identifying expired metrics based on:
- Date comparison
- TTL hours from metadata
- Time elapsed calculation using SQLite's julianday function

## Future Considerations

1. **Batch Operations**: Consider grouping metrics by TTL for more efficient batch cleanup
2. **API Exposure**: Could expose metrics stats in API responses for monitoring
3. **Notifications**: Potential for batch expiration notifications
4. **Performance Monitoring**: Track cleanup operation performance metrics

## Migration Notes

No migration required. The changes are backward compatible and the cleanup happens automatically.
