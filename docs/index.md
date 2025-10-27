# DQX - Data Quality Excellence

Welcome to DQX documentation! Transform data validation into mathematical expressions.

## What is DQX?

DQX is a Python framework that makes data validation as simple as writing math. Instead of complex SQL queries, you express your data quality rules as symbolic expressions that automatically compile to optimized SQL.

## Quick Example

```python
from dqx import mp

# Traditional SQL approach (complex, error-prone)
sql_query = """
WITH metrics AS (
  SELECT
    COUNT(*) as total_orders,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancellation_rate
  FROM orders
  WHERE order_date >= '2024-01-01'
)
SELECT
  CASE
    WHEN cancellation_rate > 0.15 THEN 'FAIL: High cancellation rate'
    WHEN unique_users < 1000 THEN 'FAIL: Low user count'
    ELSE 'PASS'
  END as validation_result
FROM metrics
"""

# DQX approach (clear, maintainable)
check = dq.check("orders").where("order_date >= '2024-01-01'")

check.assert_that(mp.cancellation_rate() <= 0.15).where(
    name="Cancellation rate check",
    cancellation_rate=mp.count().where("status = 'cancelled'") / mp.count(),
)
```

## Key Features

- **📊 Built-in Metrics**: Count, sum, average, min, max, and more
- **🔍 Smart Assertions**: Express complex rules in simple math
- **⚡ SQL Optimization**: Generates efficient queries automatically
- **📈 Time Analysis**: Compare metrics across time periods
- **🔌 Extensible**: Create custom metrics and validators

## Next Steps

- [Installation Guide](installation.md)
- [Quick Start Tutorial](quickstart.md)
- [API Reference](api-reference.md)
