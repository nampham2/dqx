#!/usr/bin/env python3
"""Demo of DuckDB query optimization using GROUP BY approach."""

import duckdb


def main() -> None:
    # Create a test database with sample data
    conn = duckdb.connect(":memory:")

    # Create sample data
    conn.execute("""
    CREATE TABLE commerce_data AS
    SELECT * FROM (VALUES
        ('2025-01-14', 'Alice', 100.0, 10.0, 5, true),
        ('2025-01-14', 'Bob', 200.0, 20.0, 3, false),
        ('2025-01-14', 'Alice', 150.0, 15.0, 2, true),
        ('2025-01-14', NULL, 300.0, 30.0, 1, false),
        ('2025-01-13', 'Charlie', 250.0, 25.0, 4, true),
        ('2025-01-13', 'Alice', 175.0, 17.5, 6, true),
        ('2025-01-13', 'Bob', 225.0, 22.5, 2, NULL)
    ) AS t(date, name, price, tax, quantity, delivered)
    """)

    print("=== CURRENT APPROACH (Multiple CTEs) ===")
    # Current approach - separate CTEs for each date
    current_query = """
    WITH
      source_2025_01_14 AS (
        SELECT * FROM commerce_data WHERE date = '2025-01-14'
      ),
      metrics_2025_01_14 AS (
        SELECT
          CAST(COUNT_IF(delivered IS NULL) AS DOUBLE) AS '_obwquu_null_count(delivered)',
          CAST(MIN(quantity) AS DOUBLE) AS '_pwvxmw_minimum(quantity)',
          CAST(COUNT(*) AS DOUBLE) AS '_kpjebz_num_rows()',
          CAST(AVG(price) AS DOUBLE) AS '_qzkzlx_average(price)',
          CAST(AVG(tax) AS DOUBLE) AS '_wzcxeu_average(tax)',
          CAST(COUNT(*) - COUNT(DISTINCT name) AS DOUBLE) AS '_oykmjq_duplicate_count(name)'
        FROM source_2025_01_14
      ),
      source_2025_01_13 AS (
        SELECT * FROM commerce_data WHERE date = '2025-01-13'
      ),
      metrics_2025_01_13 AS (
        SELECT
          CAST(COUNT(*) AS DOUBLE) AS '_bzykad_num_rows()',
          CAST(AVG(tax) AS DOUBLE) AS '_ypurlu_average(tax)'
        FROM source_2025_01_13
      )
    SELECT '2025-01-14' as date, '_obwquu_null_count(delivered)' as symbol, "_obwquu_null_count(delivered)" as value FROM metrics_2025_01_14
    UNION ALL
    SELECT '2025-01-14' as date, '_pwvxmw_minimum(quantity)' as symbol, "_pwvxmw_minimum(quantity)" as value FROM metrics_2025_01_14
    UNION ALL
    SELECT '2025-01-14' as date, '_kpjebz_num_rows()' as symbol, "_kpjebz_num_rows()" as value FROM metrics_2025_01_14
    UNION ALL
    SELECT '2025-01-14' as date, '_qzkzlx_average(price)' as symbol, "_qzkzlx_average(price)" as value FROM metrics_2025_01_14
    UNION ALL
    SELECT '2025-01-14' as date, '_wzcxeu_average(tax)' as symbol, "_wzcxeu_average(tax)" as value FROM metrics_2025_01_14
    UNION ALL
    SELECT '2025-01-14' as date, '_oykmjq_duplicate_count(name)' as symbol, "_oykmjq_duplicate_count(name)" as value FROM metrics_2025_01_14
    UNION ALL
    SELECT '2025-01-13' as date, '_bzykad_num_rows()' as symbol, "_bzykad_num_rows()" as value FROM metrics_2025_01_13
    UNION ALL
    SELECT '2025-01-13' as date, '_ypurlu_average(tax)' as symbol, "_ypurlu_average(tax)" as value FROM metrics_2025_01_13
    ORDER BY date DESC, symbol
    """

    result = conn.execute(current_query).fetchall()
    for row in result:
        print(f"{row[0]} | {row[1]:40s} | {row[2]:.1f}")

    print("\n=== OPTIMIZED APPROACH 1: GROUP BY Date ===")
    # Optimized approach - single CTE with GROUP BY
    optimized_query_1 = """
    WITH
      -- Keep source CTEs as user provides them
      source_2025_01_14 AS (
        SELECT * FROM commerce_data WHERE date = '2025-01-14'
      ),
      source_2025_01_13 AS (
        SELECT * FROM commerce_data WHERE date = '2025-01-13'
      ),
      -- Combine sources with date tag
      all_sources AS (
        SELECT '2025-01-14' as _date, * FROM source_2025_01_14
        UNION ALL
        SELECT '2025-01-13' as _date, * FROM source_2025_01_13
      ),
      -- Compute all metrics grouped by date in single pass
      metrics_by_date AS (
        SELECT
          _date,
          CAST(COUNT_IF(delivered IS NULL) AS DOUBLE) AS delivered_null_count,
          CAST(MIN(quantity) AS DOUBLE) AS quantity_minimum,
          CAST(COUNT(*) AS DOUBLE) AS num_rows,
          CAST(AVG(price) AS DOUBLE) AS price_average,
          CAST(AVG(tax) AS DOUBLE) AS tax_average,
          CAST(COUNT(*) - COUNT(DISTINCT name) AS DOUBLE) AS name_duplicate_count
        FROM all_sources
        GROUP BY _date
      )
    -- Unpivot using cross join with VALUES
    SELECT
      m._date as date,
      v.symbol,
      CASE v.metric
        WHEN 'delivered_null_count' THEN m.delivered_null_count
        WHEN 'quantity_minimum' THEN m.quantity_minimum
        WHEN 'num_rows' THEN m.num_rows
        WHEN 'price_average' THEN m.price_average
        WHEN 'tax_average' THEN m.tax_average
        WHEN 'name_duplicate_count' THEN m.name_duplicate_count
      END as value
    FROM metrics_by_date m
    CROSS JOIN (
      VALUES
        ('2025-01-14', '_obwquu_null_count(delivered)', 'delivered_null_count'),
        ('2025-01-14', '_pwvxmw_minimum(quantity)', 'quantity_minimum'),
        ('2025-01-14', '_kpjebz_num_rows()', 'num_rows'),
        ('2025-01-14', '_qzkzlx_average(price)', 'price_average'),
        ('2025-01-14', '_wzcxeu_average(tax)', 'tax_average'),
        ('2025-01-14', '_oykmjq_duplicate_count(name)', 'name_duplicate_count'),
        ('2025-01-13', '_bzykad_num_rows()', 'num_rows'),
        ('2025-01-13', '_ypurlu_average(tax)', 'tax_average')
    ) AS v(required_date, symbol, metric)
    WHERE m._date = v.required_date
    ORDER BY date DESC, symbol
    """

    result = conn.execute(optimized_query_1).fetchall()
    for row in result:
        print(f"{row[0]} | {row[1]:40s} | {row[2]:.1f}")

    print("\n=== VERIFICATION: Results Match ===")
    # Verify results are identical
    current_results = conn.execute(current_query).fetchall()
    optimized_results = conn.execute(optimized_query_1).fetchall()

    # Convert to dictionaries for easy comparison
    current_dict = {(row[0], row[1]): row[2] for row in current_results}
    optimized_dict = {(row[0], row[1]): row[2] for row in optimized_results}

    # Check if all keys and values match
    all_match = True
    for key in current_dict:
        if key not in optimized_dict:
            print(f"❌ Missing key in optimized: {key}")
            all_match = False
        elif abs(current_dict[key] - optimized_dict[key]) > 0.0001:
            print(f"❌ Value mismatch for {key}: current={current_dict[key]}, optimized={optimized_dict[key]}")
            all_match = False

    # Check for extra keys in optimized
    for key in optimized_dict:
        if key not in current_dict:
            print(f"❌ Extra key in optimized: {key}")
            all_match = False

    if all_match:
        print("✅ All results match! The optimization produces identical results.")
    else:
        print("❌ Results don't match!")

    print("\n=== PERFORMANCE COMPARISON ===")
    # Run a simple performance comparison
    import time

    # Current approach
    start = time.time()
    for _ in range(100):
        conn.execute(current_query).fetchall()
    current_time = time.time() - start

    # Optimized approach
    start = time.time()
    for _ in range(100):
        conn.execute(optimized_query_1).fetchall()
    optimized_time = time.time() - start

    print(f"Current approach (100 runs):   {current_time:.3f} seconds")
    print(f"Optimized approach (100 runs): {optimized_time:.3f} seconds")
    print(f"Speed improvement: {(current_time / optimized_time - 1) * 100:.1f}%")

    print("\n=== QUERY STRUCTURE COMPARISON ===")
    print("Current approach CTEs:", current_query.count("AS ("))
    print("Optimized approach CTEs:", optimized_query_1.count("AS ("))

    # Show query plans
    print("\n=== QUERY PLAN - CURRENT APPROACH ===")
    explain_current = conn.execute(f"EXPLAIN {current_query}").fetchall()
    for line in explain_current[:10]:  # First 10 lines
        print(line[0])
    print("...")

    print("\n=== QUERY PLAN - OPTIMIZED APPROACH ===")
    explain_optimized = conn.execute(f"EXPLAIN {optimized_query_1}").fetchall()
    for line in explain_optimized[:10]:  # First 10 lines
        print(line[0])
    print("...")

    conn.close()


if __name__ == "__main__":
    main()
