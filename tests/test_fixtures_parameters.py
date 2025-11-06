"""Tests for data fixtures with parameters support."""

from datetime import date

from dqx.common import SqlDataSource
from tests.fixtures.data_fixtures import CommercialDataSource


class TestFixturesParameters:
    """Test data fixtures with parameters."""

    def test_commercial_datasource_implements_protocol(self) -> None:
        """Test that CommercialDataSource properly implements SqlDataSource."""
        ds = CommercialDataSource(start_date=date(2024, 1, 1), end_date=date(2024, 1, 5), records_per_day=50, seed=42)

        # Verify it implements SqlDataSource protocol
        assert isinstance(ds, SqlDataSource)
        assert ds.name == "commerce"
        assert ds.dialect == "duckdb"

    def test_commercial_datasource_without_parameters(self) -> None:
        """Test CommercialDataSource CTE without parameters."""
        ds = CommercialDataSource(start_date=date(2024, 1, 1), end_date=date(2024, 1, 5), records_per_day=50, seed=42)

        # Test without parameters
        cte1 = ds.cte(date(2024, 1, 1))
        assert "WHERE order_date = DATE '2024-01-01'" in cte1
        assert "AND" not in cte1  # No additional filters

        # Test out of range date
        cte2 = ds.cte(date(2023, 12, 31))
        assert "WHERE 1=0" in cte2

    def test_commercial_datasource_with_parameters(self) -> None:
        """Test CommercialDataSource CTE with various parameters."""
        ds = CommercialDataSource(start_date=date(2024, 1, 1), end_date=date(2024, 1, 31), records_per_day=100, seed=42)

        # Test with quantity filter
        cte1 = ds.cte(date(2024, 1, 15), parameters={"min_quantity": 50})
        assert "WHERE order_date = DATE '2024-01-15'" in cte1
        assert "AND quantity >= 50" in cte1

        # Test with price filter
        cte2 = ds.cte(date(2024, 1, 15), parameters={"min_price": 100.0})
        assert "AND price >= 100.0" in cte2

        # Test with delivered filter
        cte3 = ds.cte(date(2024, 1, 15), parameters={"delivered": True})
        assert "AND delivered = True" in cte3

        # Test with item filter
        cte4 = ds.cte(date(2024, 1, 15), parameters={"item_contains": "Widget"})
        assert "AND item LIKE '%Widget%'" in cte4

        # Test with multiple parameters
        cte5 = ds.cte(
            date(2024, 1, 15),
            parameters={"min_quantity": 25, "min_price": 50.0, "delivered": False, "item_contains": "Premium"},
        )
        assert "AND quantity >= 25" in cte5
        assert "AND price >= 50.0" in cte5
        assert "AND delivered = False" in cte5
        assert "AND item LIKE '%Premium%'" in cte5

    def test_commercial_datasource_skip_dates(self) -> None:
        """Test CommercialDataSource with skip_dates parameter."""
        skip_dates = {date(2024, 1, 5), date(2024, 1, 10)}
        ds = CommercialDataSource(
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 15), skip_dates=skip_dates, seed=42
        )

        assert ds.skip_dates == skip_dates

    def test_commercial_datasource_edge_cases(self) -> None:
        """Test edge cases for CommercialDataSource."""
        ds = CommercialDataSource(start_date=date(2024, 1, 1), end_date=date(2024, 1, 10), seed=42)

        # Empty parameters dict should work like no parameters
        cte1 = ds.cte(date(2024, 1, 5), parameters={})
        assert "WHERE order_date = DATE '2024-01-05'" in cte1
        assert "AND" not in cte1

        # Unknown parameters should be ignored
        cte2 = ds.cte(date(2024, 1, 5), parameters={"unknown_param": "value"})
        assert "unknown_param" not in cte2

    def test_multiple_datasources_different_parameters(self) -> None:
        """Test multiple datasource instances with different configurations."""
        # First datasource with one set of dates
        ds1 = CommercialDataSource(
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 31), name="january_sales", seed=100
        )

        # Second datasource with different dates
        ds2 = CommercialDataSource(
            start_date=date(2024, 2, 1), end_date=date(2024, 2, 29), name="february_sales", seed=200
        )

        assert ds1.name == "january_sales"
        assert ds2.name == "february_sales"

        # Each should handle its own date range
        assert "WHERE order_date = DATE '2024-01-15'" in ds1.cte(date(2024, 1, 15))
        assert "WHERE 1=0" in ds1.cte(date(2024, 2, 15))  # Out of range

        assert "WHERE order_date = DATE '2024-02-15'" in ds2.cte(date(2024, 2, 15))
        assert "WHERE 1=0" in ds2.cte(date(2024, 1, 15))  # Out of range

    def test_parameter_sql_injection_safety(self) -> None:
        """Test that parameters are properly handled for SQL safety."""
        ds = CommercialDataSource(start_date=date(2024, 1, 1), end_date=date(2024, 1, 10), seed=42)

        # Test with potentially dangerous item_contains value
        # Note: In a real implementation, parameters should be properly escaped
        cte = ds.cte(date(2024, 1, 5), parameters={"item_contains": "Widget'; DROP TABLE--"})

        # The fixture currently just embeds the value directly
        # In production, this should be properly parameterized
        assert "Widget'; DROP TABLE--" in cte

    def test_datasource_query_execution(self) -> None:
        """Test that datasource can execute queries."""
        ds = CommercialDataSource(start_date=date(2024, 1, 1), end_date=date(2024, 1, 5), records_per_day=10, seed=42)

        # Test a simple query
        query = f"SELECT COUNT(*) as count FROM {ds._table_name}"
        result = ds.query(query)

        # Should be able to fetch results
        rows = result.fetchall()
        assert len(rows) == 1
        count = rows[0][0]
        # Approximate count (10 records/day * 5 days, +/- 20% variation)
        assert 30 <= count <= 70

    def test_commercial_datasource_reproducibility(self) -> None:
        """Test that same seed produces same data."""
        # Create two datasources with same seed
        ds1 = CommercialDataSource(
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 5), records_per_day=50, seed=12345
        )

        ds2 = CommercialDataSource(
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 5), records_per_day=50, seed=12345
        )

        # Query data from both
        query = f"SELECT COUNT(*) as count, AVG(price) as avg_price FROM {ds1._table_name}"
        result1 = ds1.query(query).fetchall()[0]

        query = f"SELECT COUNT(*) as count, AVG(price) as avg_price FROM {ds2._table_name}"
        result2 = ds2.query(query).fetchall()[0]

        # Should have same count and average price
        assert result1[0] == result2[0]  # count
        assert abs(result1[1] - result2[1]) < 0.01  # avg_price (float comparison)
