# CustomSQL and Universal Parameters Implementation Plan v1

## Overview

This plan introduces a `CustomSQL` operation to DQX that allows users to write custom SQL expressions with parameter templating. It also lays the foundation for universal parameter support across all operations, enabling dynamic CTE customization based on operation parameters.

## Goals

1. **Add CustomSQL Operation**: Enable users to write custom SQL metrics with templating support
2. **Universal Parameter Support**: Add optional parameters to all operations for CTE customization
3. **Parameter Grouping**: Optimize SQL generation by grouping operations with same parameters
4. **Backward Compatibility**: Ensure all existing code continues to work without changes

## Design Principles

- **YAGNI**: Start with minimal changes, add complexity only when needed
- **KISS**: Keep parameter handling simple and predictable
- **TDD**: Write tests first for each component
- **DRY**: Share parameter grouping logic between dialects

## Architecture Overview

### Current State
```
SqlOp (no parameters) → Dialect.translate_sql_op() → SQL Expression
DataSource.cte(date) → Single CTE for all operations
```

### Target State
```
SqlOp (with parameters) → Group by parameters → Multiple specialized CTEs
DataSource.cte(date, params) → Parameter-specific CTE generation
CustomSQL with templating → Render with parameters → SQL Expression
```

## Implementation Tasks

### Task Group 1: Protocol and Base Class Updates (Foundation)

#### Task 1.1: Extend OpValueMixin with Parameters Support
**File**: `src/dqx/ops.py`

**Current Code**:
```python
class OpValueMixin(Generic[T]):
    def __init__(self) -> None:
        self._value: T | None = None
```

**Target Code**:
```python
class OpValueMixin(Generic[T]):
    def __init__(self, parameters: Parameters | None = None) -> None:
        self._value: T | None = None
        self.parameters = parameters or {}
```

**Test First** (`tests/test_ops_parameters.py`):
```python
"""Test parameter support in operations."""
import pytest
from dqx.ops import OpValueMixin, Average, Parameters


def test_opvaluemixin_accepts_parameters():
    """OpValueMixin should accept and store parameters."""
    params = {"region": "US", "threshold": 100}

    class TestOp(OpValueMixin[float]):
        pass

    op = TestOp(parameters=params)
    assert op.parameters == params


def test_opvaluemixin_defaults_empty_parameters():
    """OpValueMixin should default to empty dict when no parameters."""
    class TestOp(OpValueMixin[float]):
        pass

    op = TestOp()
    assert op.parameters == {}


def test_opvaluemixin_converts_none_to_empty_dict():
    """OpValueMixin should convert None parameters to empty dict."""
    class TestOp(OpValueMixin[float]):
        pass

    op = TestOp(parameters=None)
    assert op.parameters == {}
```

#### Task 1.2: Add Parameters Property to SqlOp Protocol
**File**: `src/dqx/ops.py`

**Add to SqlOp Protocol**:
```python
@runtime_checkable
class SqlOp(Op[T], Protocol[T]):
    # ... existing properties ...

    @property
    def parameters(self) -> Parameters:
        """Operation parameters for CTE/SQL customization.

        Returns empty dict by default for backward compatibility.
        """
        return {}
```

**Test** (`tests/test_ops_parameters.py` - add to existing):
```python
def test_sqlop_protocol_has_parameters():
    """SqlOp protocol should include parameters property."""
    from dqx.ops import SqlOp, NumRows

    # Existing ops should satisfy protocol
    op = NumRows()
    assert isinstance(op, SqlOp)
    assert hasattr(op, 'parameters')
    assert op.parameters == {}
```

#### Task 1.3: Update SqlDataSource Protocol
**File**: `src/dqx/common.py`

**Current Code**:
```python
@runtime_checkable
class SqlDataSource(Protocol):
    @property
    def table_name(self) -> str: ...

    @property
    def dialect(self) -> str: ...

    def cte(self, nominal_date: datetime.date) -> str: ...
```

**Target Code**:
```python
@runtime_checkable
class SqlDataSource(Protocol):
    @property
    def table_name(self) -> str: ...

    @property
    def dialect(self) -> str: ...

    def cte(self, nominal_date: datetime.date, parameters: Parameters | None = None) -> str:
        """Generate CTE SQL for the given date with optional parameters.

        Args:
            nominal_date: Date for data extraction
            parameters: Optional parameters for CTE customization

        Returns:
            CTE SQL string
        """
        ...
```

**Test** (`tests/test_common_parameters.py`):
```python
"""Test parameter support in common protocols."""
import datetime
from dqx.common import SqlDataSource, Parameters


def test_sqldatasource_protocol_accepts_parameters():
    """SqlDataSource protocol should accept parameters in cte method."""

    class TestDataSource:
        table_name = "test_table"
        dialect = "duckdb"

        def cte(self, nominal_date: datetime.date, parameters: Parameters | None = None) -> str:
            base = f"SELECT * FROM {self.table_name} WHERE date = '{nominal_date}'"
            if parameters and "region" in parameters:
                base += f" AND region = '{parameters['region']}'"
            return base

    ds = TestDataSource()

    # Should work without parameters (backward compatible)
    sql1 = ds.cte(datetime.date(2024, 1, 1))
    assert "WHERE date = '2024-01-01'" in sql1
    assert "region" not in sql1

    # Should work with parameters
    sql2 = ds.cte(datetime.date(2024, 1, 1), {"region": "US"})
    assert "AND region = 'US'" in sql2

    # Should satisfy protocol
    assert isinstance(ds, SqlDataSource)
```

**Verification**: Run `uv run mypy src/dqx/common.py src/dqx/ops.py tests/test_ops_parameters.py tests/test_common_parameters.py` and `uv run pytest tests/test_ops_parameters.py tests/test_common_parameters.py -xvs`

### Task Group 2: Update All Operation Classes (12 Operations)

#### Task 2.1: Update Column-less Operations
**File**: `src/dqx/ops.py`

**NumRows Operation**:
```python
class NumRows(OpValueMixin[float], SqlOp[float]):
    """Count total number of rows."""
    __match_args__ = ("parameters",)  # Add for pattern matching

    def __init__(self, parameters: Parameters | None = None) -> None:
        """Initialize NumRows operation.

        Args:
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
        self._prefix = random_prefix()
```

**Test** (`tests/test_ops_parameters.py` - add):
```python
def test_numrows_accepts_parameters():
    """NumRows should accept parameters."""
    params = {"region": "US"}
    op = NumRows(parameters=params)
    assert op.parameters == params

    # Backward compatibility - no parameters
    op2 = NumRows()
    assert op2.parameters == {}
```

#### Task 2.2: Update Single-Column Operations
**File**: `src/dqx/ops.py`

**Pattern for all single-column operations** (Average, Sum, Minimum, Maximum, Variance, First, NullCount, NegativeCount, UniqueCount):
```python
class Average(OpValueMixin[float], SqlOp[float]):
    """Calculate average of a column."""
    __match_args__ = ("column", "parameters")

    def __init__(self, column: str, parameters: Parameters | None = None) -> None:
        """Initialize Average operation.

        Args:
            column: Column name to calculate average
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
        self.column = column
        self._prefix = random_prefix()
```

**Test** (`tests/test_ops_parameters.py` - add):
```python
@pytest.mark.parametrize("op_class,column", [
    (Average, "price"),
    (Sum, "amount"),
    (Minimum, "value"),
    (Maximum, "value"),
    (Variance, "score"),
    (First, "id"),
    (NullCount, "field"),
    (NegativeCount, "balance"),
    (UniqueCount, "user_id"),
])
def test_single_column_ops_accept_parameters(op_class, column):
    """All single-column operations should accept parameters."""
    params = {"region": "EU", "min_value": 50}

    # With parameters
    op1 = op_class(column, parameters=params)
    assert op1.column == column
    assert op1.parameters == params

    # Without parameters (backward compatibility)
    op2 = op_class(column)
    assert op2.column == column
    assert op2.parameters == {}
```

#### Task 2.3: Update Multi-Column Operations
**File**: `src/dqx/ops.py`

**DuplicateCount Operation**:
```python
class DuplicateCount(OpValueMixin[float], SqlOp[float]):
    """Count duplicate rows based on columns."""
    __match_args__ = ("columns", "parameters")

    def __init__(self, columns: list[str], parameters: Parameters | None = None) -> None:
        """Initialize DuplicateCount operation.

        Args:
            columns: List of columns to check for duplicates
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
        self.columns = sorted(columns)  # Sort for consistency
        self._prefix = random_prefix()
```

**Test** (`tests/test_ops_parameters.py` - add):
```python
def test_duplicate_count_accepts_parameters():
    """DuplicateCount should accept parameters."""
    cols = ["user_id", "date"]
    params = {"min_count": 2}

    op1 = DuplicateCount(cols, parameters=params)
    assert op1.columns == sorted(cols)
    assert op1.parameters == params

    # Backward compatibility
    op2 = DuplicateCount(cols)
    assert op2.parameters == {}
```

#### Task 2.4: Update CountValues Operation
**File**: `src/dqx/ops.py`

**CountValues Operation** (most complex due to value validation):
```python
class CountValues(OpValueMixin[float], SqlOp[float]):
    """Count occurrences of specific values."""
    __match_args__ = ("column", "values", "parameters")

    def __init__(
        self,
        column: str,
        values: CountableValues,
        parameters: Parameters | None = None
    ) -> None:
        """Initialize CountValues operation.

        Args:
            column: Column name to count values in
            values: Values to count (int, str, bool, or list)
            parameters: Optional parameters for CTE customization
        """
        OpValueMixin.__init__(self, parameters)
        self.column = column

        # Existing validation logic unchanged
        validated_values = self._validate_values(values)
        self._values = validated_values["values"]
        self._is_single = validated_values["is_single"]

        self._prefix = random_prefix()
```

**Test** (`tests/test_ops_parameters.py` - add):
```python
def test_count_values_accepts_parameters():
    """CountValues should accept parameters."""
    params = {"category": "electronics"}

    # Single value
    op1 = CountValues("status", "active", parameters=params)
    assert op1.parameters == params

    # Multiple values
    op2 = CountValues("status", ["active", "pending"], parameters=params)
    assert op2.parameters == params

    # Backward compatibility
    op3 = CountValues("status", "active")
    assert op3.parameters == {}
```

**Verification**: Run all tests and type checking:
```bash
uv run mypy src/dqx/ops.py tests/test_ops_parameters.py
uv run pytest tests/test_ops_parameters.py -xvs
```

### Task Group 3: Implement CustomSQL Operation

#### Task 3.1: Add CustomSQL Class
**File**: `src/dqx/ops.py`

**Add imports**:
```python
from string import Template
```

**Add CustomSQL class**:
```python
@dataclass(frozen=True)
class CustomSQL(OpValueMixin[float], SqlOp[float]):
    """Custom SQL operation with parameter templating.

    Allows writing custom SQL expressions with template variables
    that are substituted from operation parameters.

    Example:
        # Simple custom metric
        op = CustomSQL("COUNT(DISTINCT user_id)")

        # With parameters
        op = CustomSQL(
            "SUM(CASE WHEN amount > $threshold THEN amount ELSE 0 END)",
            parameters={"threshold": 1000}
        )
    """
    __match_args__ = ("sql_expression", "parameters")

    sql_expression: str
    _template: Template = field(init=False)
    _prefix: str = field(default_factory=random_prefix)

    def __init__(self, sql_expression: str, parameters: Parameters | None = None) -> None:
        """Initialize CustomSQL operation.

        Args:
            sql_expression: SQL expression with optional template variables ($var)
            parameters: Parameters for template substitution and CTE customization

        Raises:
            ValueError: If template variables are missing from parameters
        """
        # Can't use dataclass init with OpValueMixin
        object.__setattr__(self, 'sql_expression', sql_expression)
        object.__setattr__(self, '_template', Template(sql_expression))
        object.__setattr__(self, '_prefix', random_prefix())

        # Initialize mixin
        OpValueMixin.__init__(self, parameters)

    def render_sql(self) -> str:
        """Render SQL expression with parameter substitution.

        Returns:
            Rendered SQL string

        Raises:
            ValueError: If required template parameter is missing
        """
        try:
            return self._template.substitute(self.parameters)
        except KeyError as e:
            missing_param = str(e).strip("'")
            raise ValueError(
                f"Missing template parameter '{missing_param}' in CustomSQL. "
                f"Available parameters: {list(self.parameters.keys())}"
            )

    @property
    def sql_col(self) -> str:
        """Column name for SQL alias."""
        # Include first 8 chars of expression for readability
        expr_preview = self.sql_expression[:20].replace(" ", "_")
        return f"{self._prefix}_custom_sql_{expr_preview}"

    @property
    def symbol(self) -> str:
        """Symbol for display."""
        return f"custom_sql({self.sql_expression[:30]}...)"
```

**Test First** (`tests/test_custom_sql.py`):
```python
"""Test CustomSQL operation."""
import pytest
from dqx.ops import CustomSQL


def test_custom_sql_simple():
    """CustomSQL should work with simple expressions."""
    op = CustomSQL("COUNT(DISTINCT user_id)")
    assert op.sql_expression == "COUNT(DISTINCT user_id)"
    assert op.render_sql() == "COUNT(DISTINCT user_id)"
    assert "custom_sql_COUNT(DISTINCT_user_" in op.sql_col
    assert op.parameters == {}


def test_custom_sql_with_template_parameters():
    """CustomSQL should substitute template parameters."""
    op = CustomSQL(
        "SUM(CASE WHEN amount > $threshold THEN amount ELSE 0 END)",
        parameters={"threshold": 1000}
    )

    rendered = op.render_sql()
    assert rendered == "SUM(CASE WHEN amount > 1000 THEN amount ELSE 0 END)"


def test_custom_sql_multiple_parameters():
    """CustomSQL should handle multiple template parameters."""
    op = CustomSQL(
        "COUNT(CASE WHEN status = '$status' AND region = '$region' THEN 1 END)",
        parameters={"status": "active", "region": "US"}
    )

    rendered = op.render_sql()
    assert "status = 'active'" in rendered
    assert "region = 'US'" in rendered


def test_custom_sql_missing_parameter():
    """CustomSQL should raise ValueError for missing parameters."""
    op = CustomSQL(
        "SUM(amount) WHERE category = '$category'",
        parameters={"region": "US"}  # Missing 'category'
    )

    with pytest.raises(ValueError) as exc_info:
        op.render_sql()

    assert "Missing template parameter 'category'" in str(exc_info.value)
    assert "Available parameters: ['region']" in str(exc_info.value)


def test_custom_sql_escaping():
    """CustomSQL should handle $ escaping."""
    # $$ escapes to single $
    op = CustomSQL("SUM($$revenue)")
    assert op.render_sql() == "SUM($revenue)"

    # With actual parameter
    op2 = CustomSQL("SUM($metric) + $$100", parameters={"metric": "sales"})
    assert op2.render_sql() == "SUM(sales) + $100"


def test_custom_sql_symbol_truncation():
    """CustomSQL symbol should truncate long expressions."""
    long_expr = "SELECT " + "x" * 100
    op = CustomSQL(long_expr)

    symbol = op.symbol
    assert symbol.startswith("custom_sql(")
    assert symbol.endswith("...)")
    assert len(symbol) < 50  # Reasonable length
```

#### Task 3.2: Add CustomSQL to __all__
**File**: `src/dqx/ops.py`

Update `__all__` to include CustomSQL:
```python
__all__ = [
    # ... existing exports ...
    "CountValues",
    "CountableValues",
    "UniqueCount",
    "CustomSQL",  # Add this
]
```

**Verification**:
```bash
uv run mypy src/dqx/ops.py tests/test_custom_sql.py
uv run pytest tests/test_custom_sql.py -xvs
```

### Task Group 4: Dialect Parameter Grouping Support

#### Task 4.1: Add Parameter Grouping Logic to Base Dialect
**File**: `src/dqx/dialect.py`

**Add helper method to both DuckDBDialect and BigQueryDialect**:
```python
def _group_operations_by_parameters(self, ops: Sequence[SqlOp]) -> dict[tuple, list[SqlOp]]:
    """Group operations by their parameters for efficient CTE generation.

    Operations with identical parameters can share the same source CTE,
    reducing query complexity and improving performance.

    Args:
        ops: List of SQL operations to group

    Returns:
        Dictionary mapping parameter tuples to operation lists
    """
    groups: dict[tuple, list[SqlOp]] = {}

    for op in ops:
        # Create hashable key from parameters
        params_key = tuple(sorted(op.parameters.items()))

        if params_key not in groups:
            groups[params_key] = []
        groups[params_key].append(op)

    return groups
```

**Test** (`tests/test_dialect_parameter_grouping.py`):
```python
"""Test parameter grouping in dialects."""
import pytest
from dqx.ops import Average, Sum, CustomSQL
from dqx.dialect import DuckDBDialect, BigQueryDialect


@pytest.mark.parametrize("dialect_class", [DuckDBDialect, BigQueryDialect])
def test_dialect_groups_operations_by_parameters(dialect_class):
    """Dialects should group operations by parameters."""
    dialect = dialect_class()

    # Operations with different parameters
    ops = [
        Average("price", parameters={"region": "US"}),
        Sum("amount", parameters={"region": "US"}),
        Average("cost", parameters={"region": "EU"}),
        Sum("total", parameters={"region": "EU"}),
        CustomSQL("COUNT(*)", parameters={}),
    ]

    groups = dialect._group_operations_by_parameters(ops)

    # Should have 3 groups: US, EU, and empty
    assert len(groups) == 3

    # Check US group
    us_key = (("region", "US"),)
    assert us_key in groups
    assert len(groups[us_key]) == 2
    assert all(op.parameters["region"] == "US" for op in groups[us_key])

    # Check EU group
    eu_key = (("region", "EU"),)
    assert eu_key in groups
    assert len(groups[eu_key]) == 2

    # Check empty group
    empty_key = ()
    assert empty_key in groups
    assert len(groups[empty_key]) == 1


def test_parameter_grouping_with_multiple_params():
    """Test grouping with multiple parameters."""
    dialect = DuckDBDialect()

    ops = [
        Average("x", parameters={"region": "US", "category": "A"}),
        Sum("y", parameters={"region": "US", "category": "A"}),
        Average("z", parameters={"category": "A", "region": "US"}),  # Same as first
    ]

    groups = dialect._group_operations_by_parameters(ops)

    # All should be in same group (parameters are sorted)
    assert len(groups) == 1
    key = (("category", "A"), ("region", "US"))
    assert key in groups
    assert len(groups[key]) == 3
```

#### Task 4.2: Update translate_sql_op for CustomSQL
**File**: `src/dqx/dialect.py`

**Add to both DuckDBDialect and BigQueryDialect translate_sql_op methods**:

For DuckDBDialect:
```python
def translate_sql_op(self, op: ops.SqlOp) -> str:
    """Translate SqlOp to DuckDB SQL syntax."""

    match op:
        # ... existing cases ...

        case ops.CustomSQL():
            rendered_sql = op.render_sql()
            return f"CAST(({rendered_sql}) AS DOUBLE) AS '{op.sql_col}'"

        case _:
            raise ValueError(f"Unsupported SqlOp type: {type(op).__name__}")
```

For BigQueryDialect (similar but with backticks):
```python
def translate_sql_op(self, op: ops.SqlOp) -> str:
    """Translate SqlOp to BigQuery SQL syntax."""

    match op:
        # ... existing cases ...

        case ops.CustomSQL():
            rendered_sql = op.render_sql()
            return f"CAST(({rendered_sql}) AS FLOAT64) AS `{op.sql_col}`"

        case _:
            raise ValueError(f"Unsupported SqlOp type: {type(op).__name__}")
```

**Test** (`tests/test_dialect_custom_sql.py`):
```python
"""Test CustomSQL translation in dialects."""
import pytest
from dqx.ops import CustomSQL
from dqx.dialect import DuckDBDialect, BigQueryDialect


def test_duckdb_translates_custom_sql():
    """DuckDB should translate CustomSQL operations."""
    dialect = DuckDBDialect()

    # Simple expression
    op = CustomSQL("COUNT(DISTINCT user_id)")
    sql = dialect.translate_sql_op(op)

    assert "CAST((COUNT(DISTINCT user_id)) AS DOUBLE)" in sql
    assert f"AS '{op.sql_col}'" in sql

    # With parameters
    op2 = CustomSQL("SUM(amount * $factor)", parameters={"factor": 1.5})
    sql2 = dialect.translate_sql_op(op2)

    assert "CAST((SUM(amount * 1.5)) AS DOUBLE)" in sql2


def test_bigquery_translates_custom_sql():
    """BigQuery should translate CustomSQL operations."""
    dialect = BigQueryDialect()

    op = CustomSQL("APPROX_QUANTILES(value, 100)[OFFSET(50)]")
    sql = dialect.translate_sql_op(op)

    assert "CAST((APPROX_QUANTILES(value, 100)[OFFSET(50)]) AS FLOAT64)" in sql
    assert f"AS `{op.sql_col}`" in sql  # Backticks for BigQuery


def test_custom_sql_error_propagation():
    """CustomSQL errors should propagate through dialect."""
    dialect = DuckDBDialect()

    # Missing parameter
    op = CustomSQL("SUM($missing)", parameters={})

    with pytest.raises(ValueError) as exc_info:
        dialect.translate_sql_op(op)

    assert "Missing template parameter 'missing'" in str(exc_info.value)
```

**Verification**:
```bash
uv run mypy src/dqx/dialect.py tests/test_dialect_parameter_grouping.py tests/test_dialect_custom_sql.py
uv run pytest tests/test_dialect_parameter_grouping.py tests/test_dialect_custom_sql.py -xvs
```

### Task Group 5: Enhanced Batch Query Generation

#### Task 5.1: Update build_batch_cte_query with Parameter Grouping
**File**: `src/dqx/dialect.py`

**Update _build_cte_parts to handle parameter groups**:
```python
def _build_cte_parts_with_params(
    dialect: "Dialect", cte_data: list["BatchCTEData"]
) -> tuple[list[str], list[tuple[str, list[ops.SqlOp]]]]:
    """Build CTE parts with parameter-aware grouping.

    Args:
        dialect: The dialect instance to use for SQL translation
        cte_data: List of BatchCTEData objects

    Returns:
        Tuple of (cte_parts, metrics_info)
    """
    if not cte_data:
        raise ValueError("No CTE data provided")

    cte_parts = []
    metrics_info: list[tuple[str, list[ops.SqlOp]]] = []

    for i, data in enumerate(cte_data):
        date_suffix = data.key.yyyy_mm_dd.strftime("%Y_%m_%d")

        # Group operations by parameters
        ops_by_params = dialect._group_operations_by_parameters(data.ops)

        # Create CTE for each parameter group
        for j, (params_key, grouped_ops) in enumerate(ops_by_params.items()):
            params_dict = dict(params_key)

            # Source CTE with parameters
            source_cte = f"source_{date_suffix}_{i}_{j}"
            # Pass parameters to data source CTE generation
            parameterized_cte_sql = data.cte_sql  # Already includes base CTE
            cte_parts.append(f"{source_cte} AS ({parameterized_cte_sql})")

            # Metrics CTE for this parameter group
            if grouped_ops:
                metrics_cte = f"metrics_{date_suffix}_{i}_{j}"
                expressions = [dialect.translate_sql_op(op) for op in grouped_ops]
                metrics_select = ", ".join(expressions)
                cte_parts.append(f"{metrics_cte} AS (SELECT {metrics_select} FROM {source_cte})")

                # Store metrics info
                metrics_info.append((metrics_cte, list(grouped_ops)))

    return cte_parts, metrics_info
```

**Update build_batch_cte_query for both dialects**:
```python
def build_batch_cte_query(self, cte_data: list["BatchCTEData"]) -> str:
    """Build batch CTE query using parameter-aware grouping."""

    # Use enhanced CTE parts builder
    cte_parts, metrics_info = _build_cte_parts_with_params(self, cte_data)

    if not metrics_info:
        raise ValueError("No metrics to compute")

    # Build value selects with array format
    value_selects = []
    for data, (metrics_cte, data_ops) in zip(cte_data, metrics_info):
        date_str = data.key.yyyy_mm_dd.isoformat()
        values_expr = self._format_array_values(data_ops)  # Existing method
        value_selects.append(f"SELECT '{date_str}' as date, {values_expr} as values FROM {metrics_cte}")

    # Combine CTEs and selects
    cte_clause = "WITH\n  " + ",\n  ".join(cte_parts)
    union_clause = "\n".join(
        f"{'UNION ALL' if i > 0 else ''}\n{select}"
        for i, select in enumerate(value_selects)
    )

    return f"{cte_clause}\n{union_clause}"
```

**Test** (`tests/test_dialect_batch_parameters.py`):
```python
"""Test batch query generation with parameters."""
import datetime
import pytest
from dqx.common import ResultKey
from dqx.ops import Average, Sum, CustomSQL
from dqx.dialect import DuckDBDialect, BigQueryDialect, BatchCTEData


@pytest.mark.parametrize("dialect_class", [DuckDBDialect, BigQueryDialect])
def test_batch_query_groups_by_parameters(dialect_class):
    """Batch queries should group operations by parameters."""
    dialect = dialect_class()

    # Create test data with different parameter groups
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags=None)

    cte_data = [
        BatchCTEData(
            key=key,
            cte_sql="SELECT * FROM sales WHERE date = '2024-01-01'",
            ops=[
                Average("price", parameters={"region": "US"}),
                Sum("amount", parameters={"region": "US"}),
                Average("price", parameters={"region": "EU"}),
                CustomSQL("COUNT(DISTINCT user_id)"),
            ]
        )
    ]

    sql = dialect.build_batch_cte_query(cte_data)

    # Should have multiple source CTEs for different parameter groups
    assert "source_2024_01_01_0_0" in sql  # First parameter group
    assert "source_2024_01_01_0_1" in sql  # Second parameter group
    assert "source_2024_01_01_0_2" in sql  # Third parameter group (empty params)

    # Should have corresponding metrics CTEs
    assert "metrics_2024_01_01_0_0" in sql
    assert "metrics_2024_01_01_0_1" in sql
    assert "metrics_2024_01_01_0_2" in sql


def test_parameter_aware_batch_sql_generation():
    """Test full SQL generation with parameter grouping."""
    dialect = DuckDBDialect()

    # Multiple dates with different operations
    cte_data = [
        BatchCTEData(
            key=ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags=None),
            cte_sql="SELECT * FROM sales WHERE date = '2024-01-01'",
            ops=[
                Average("revenue", parameters={"segment": "enterprise"}),
                Sum("revenue", parameters={"segment": "enterprise"}),
                Average("revenue", parameters={"segment": "smb"}),
            ]
        ),
        BatchCTEData(
            key=ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 2), tags=None),
            cte_sql="SELECT * FROM sales WHERE date = '2024-01-02'",
            ops=[
                CustomSQL("PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)"),
            ]
        )
    ]

    sql = dialect.build_batch_cte_query(cte_data)

    # Verify structure
    assert sql.startswith("WITH")
    assert "UNION ALL" in sql
    assert "'2024-01-01' as date" in sql
    assert "'2024-01-02' as date" in sql

    # Verify array format for values
    assert "[{" in sql  # DuckDB array syntax
    assert "'key':" in sql
    assert "'value':" in sql
```

### Task Group 6: Update Analyzer Integration

#### Task 6.1: Update analyze_sql_ops for Parameter Support
**File**: `src/dqx/analyzer.py`

**Current analyze_sql_ops signature** (no changes needed - parameters flow through ops):
```python
def analyze_sql_ops(
    ds: SqlDataSource,
    ops_by_key: dict[ResultKey, list[SqlOp[Any]]]
) -> None:
    """Analyze SQL operations and assign values.

    The operations already contain parameters, which will be used
    for grouping during SQL generation.
    """
    # Build CTE data
    cte_data = []
    for key, ops in ops_by_key.items():
        # CTE generation uses first parameter set found
        # This is fine since dialect will group by parameters
        cte_sql = ds.cte(key.yyyy_mm_dd)
        cte_data.append(BatchCTEData(key=key, cte_sql=cte_sql, ops=ops))

    # Get dialect and generate SQL
    dialect = get_dialect(ds.dialect)
    batch_sql = dialect.build_batch_cte_query(cte_data)

    # Execute and process results...
```

**Test** (`tests/test_analyzer_parameters.py`):
```python
"""Test analyzer with parameter support."""
import datetime
import pytest
from dqx.common import ResultKey
from dqx.ops import Average, Sum, CustomSQL
from dqx.analyzer import analyze_sql_ops


def test_analyzer_processes_operations_with_parameters(mock_data_source):
    """Analyzer should handle operations with parameters."""

    # Operations with different parameters
    ops_by_key = {
        ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags=None): [
            Average("price", parameters={"region": "US"}),
            Sum("quantity", parameters={"region": "US"}),
            Average("price", parameters={"region": "EU"}),
            CustomSQL("MAX(order_id)"),
        ]
    }

    # Analyzer should process without errors
    analyze_sql_ops(mock_data_source, ops_by_key)

    # Verify operations have values assigned
    for ops in ops_by_key.values():
        for op in ops:
            assert op.value is not None


def test_custom_sql_through_analyzer(mock_data_source):
    """CustomSQL operations should work through analyzer."""

    ops_by_key = {
        ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags=None): [
            CustomSQL(
                "SUM(CASE WHEN amount > $threshold THEN 1 ELSE 0 END)",
                parameters={"threshold": 1000}
            ),
            CustomSQL("COUNT(DISTINCT customer_id)"),
        ]
    }

    # Should process successfully
    analyze_sql_ops(mock_data_source, ops_by_key)

    # Values should be assigned
    for ops in ops_by_key.values():
        for op in ops:
            assert op.value is not None
```

### Task Group 7: Update Test Data Sources

#### Task 7.1: Update Test Fixtures
**File**: `tests/fixtures/datasources.py`

**Update existing test data sources**:
```python
class TestDataSource:
    """Test data source with parameter support."""

    table_name = "test_table"
    dialect = "duckdb"

    def __init__(self, data: dict[datetime.date, list[dict[str, Any]]]):
        self.data = data

    def cte(self, nominal_date: datetime.date, parameters: Parameters | None = None) -> str:
        """Generate CTE with optional parameter filtering."""
        base_filter = f"date = '{nominal_date}'"

        if parameters:
            # Apply parameter-based filtering
            for key, value in parameters.items():
                if key == "region":
                    base_filter += f" AND region = '{value}'"
                elif key == "segment":
                    base_filter += f" AND segment = '{value}'"
                elif key == "min_amount":
                    base_filter += f" AND amount >= {value}"

        return f"SELECT * FROM {self.table_name} WHERE {base_filter}"
```

**Test** (`tests/test_fixtures_parameters.py`):
```python
"""Test fixtures with parameter support."""
import datetime
from tests.fixtures.datasources import TestDataSource


def test_test_datasource_supports_parameters():
    """Test data source should support parameters."""

    ds = TestDataSource({})

    # Without parameters
    cte1 = ds.cte(datetime.date(2024, 1, 1))
    assert cte1 == "SELECT * FROM test_table WHERE date = '2024-01-01'"

    # With single parameter
    cte2 = ds.cte(datetime.date(2024, 1, 1), {"region": "US"})
    assert "AND region = 'US'" in cte2

    # With multiple parameters
    cte3 = ds.cte(
        datetime.date(2024, 1, 1),
        {"region": "EU", "min_amount": 100}
    )
    assert "AND region = 'EU'" in cte3
    assert "AND amount >= 100" in cte3
```

### Task Group 8: Integration Tests

#### Task 8.1: End-to-End Integration Tests
**File**: `tests/test_custom_sql_integration.py`

```python
"""Integration tests for CustomSQL and parameters."""
import datetime
import pytest
from dqx.api import VerificationSuite
from dqx.ops import Average, Sum, CustomSQL
from tests.fixtures.datasources import create_test_datasource


def test_custom_sql_in_verification_suite():
    """CustomSQL should work in verification suite."""

    # Create test data
    data = {
        datetime.date(2024, 1, 1): [
            {"amount": 100, "status": "active", "region": "US"},
            {"amount": 200, "status": "active", "region": "US"},
            {"amount": 300, "status": "pending", "region": "EU"},
        ]
    }

    ds = create_test_datasource(data)

    # Create suite with custom SQL
    suite = VerificationSuite("test_suite") \
        .add(Sum("amount")) \
        .add(CustomSQL("COUNT(DISTINCT status)")) \
        .add(CustomSQL(
            "SUM(CASE WHEN status = '$status' THEN amount ELSE 0 END)",
            parameters={"status": "active"}
        ))

    # Run verification
    result = suite.verify(ds, datetime.date(2024, 1, 1))

    # Check results
    assert result.metrics[0].value == 600  # Sum of all amounts
    assert result.metrics[1].value == 2    # Two distinct statuses
    assert result.metrics[2].value == 300  # Sum of active amounts


def test_parameter_grouping_efficiency():
    """Operations with same parameters should share CTEs."""

    data = {
        datetime.date(2024, 1, 1): [
            {"price": 10, "quantity": 5, "region": "US"},
            {"price": 20, "quantity": 3, "region": "US"},
            {"price": 15, "quantity": 4, "region": "EU"},
        ]
    }

    ds = create_test_datasource(data)

    # Multiple operations with same parameters
    suite = VerificationSuite("test_suite") \
        .add(Average("price", parameters={"region": "US"})) \
        .add(Sum("quantity", parameters={"region": "US"})) \
        .add(Average("price", parameters={"region": "EU"})) \
        .add(Sum("quantity", parameters={"region": "EU"}))

    result = suite.verify(ds, datetime.date(2024, 1, 1))

    # US metrics
    assert result.metrics[0].value == 15  # Average price US
    assert result.metrics[1].value == 8   # Sum quantity US

    # EU metrics
    assert result.metrics[2].value == 15  # Average price EU
    assert result.metrics[3].value == 4   # Sum quantity EU


def test_mixed_standard_and_custom_operations():
    """Standard and custom operations should work together."""

    data = {
        datetime.date(2024, 1, 1): [
            {"revenue": 1000, "cost": 600, "customer_id": 1},
            {"revenue": 1500, "cost": 900, "customer_id": 2},
            {"revenue": 800, "cost": 500, "customer_id": 1},
        ]
    }

    ds = create_test_datasource(data)

    suite = VerificationSuite("profit_analysis") \
        .add(Sum("revenue")) \
        .add(Sum("cost")) \
        .add(CustomSQL("SUM(revenue - cost)")) \
        .add(CustomSQL("COUNT(DISTINCT customer_id)")) \
        .add(CustomSQL(
            "AVG(CASE WHEN revenue > $min_revenue THEN revenue - cost ELSE NULL END)",
            parameters={"min_revenue": 1000}
        ))

    result = suite.verify(ds, datetime.date(2024, 1, 1))

    assert result.metrics[0].value == 3300  # Total revenue
    assert result.metrics[1].value == 2000  # Total cost
    assert result.metrics[2].value == 1300  # Total profit
    assert result.metrics[3].value == 2     # Unique customers
    assert result.metrics[4].value == 500   # Avg profit for high revenue
```

### Task Group 9: Documentation

#### Task 9.1: Update API Documentation
**File**: `docs/api/custom_sql.md`

```markdown
# CustomSQL Operation

The `CustomSQL` operation allows you to write custom SQL expressions for metrics that aren't covered by the standard operations.

## Basic Usage

```python
from dqx.ops import CustomSQL

# Simple custom metric
distinct_users = CustomSQL("COUNT(DISTINCT user_id)")

# Complex calculation
high_value_orders = CustomSQL("""
    COUNT(CASE WHEN order_total > 1000 THEN 1 END)
""")
```

## Template Parameters

CustomSQL supports template parameters using Python's string.Template syntax:

```python
# Using template parameters
threshold_metric = CustomSQL(
    "SUM(CASE WHEN amount > $threshold THEN amount ELSE 0 END)",
    parameters={"threshold": 1000}
)

# Multiple parameters
filtered_average = CustomSQL(
    """
    AVG(CASE
        WHEN status = '$status' AND region = '$region'
        THEN amount
        ELSE NULL
    END)
    """,
    parameters={"status": "active", "region": "US"}
)
```

## Parameter Rules

1. Template variables use `$variable` syntax
2. To include a literal `$`, use `$$`
3. Missing parameters will raise `ValueError`
4. Parameters are also used for CTE optimization

## Integration with VerificationSuite

```python
from dqx.api import VerificationSuite

suite = VerificationSuite("custom_metrics")
    .add(CustomSQL("PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)"))
    .add(CustomSQL(
        "SUM(amount) FILTER (WHERE category = '$category')",
        parameters={"category": "electronics"}
    ))
```

## Best Practices

1. **Keep expressions simple**: Complex logic should be in the data source CTE
2. **Use parameters for flexibility**: Avoid hardcoding values
3. **Test thoroughly**: Custom SQL bypasses DQX's type safety
4. **Document your metrics**: Explain what each custom metric calculates
```

#### Task 9.2: Update README with Examples
**File**: Update relevant sections in README.md to mention CustomSQL capability

### Final Verification Step

**Run all tests and checks**:
```bash
# Type checking
uv run mypy src/dqx tests/

# Linting
uv run ruff check src/ tests/

# All tests
uv run pytest tests/ -xvs

# Coverage
uv run pytest tests/ --cov=dqx --cov-report=term-missing
```

## Implementation Summary

This plan adds CustomSQL and universal parameter support to DQX with:

1. **Backward Compatibility**: All existing code continues to work
2. **Minimal Changes**: Only essential modifications to protocols
3. **Parameter Grouping**: Efficient SQL generation
4. **Template Support**: Flexible custom SQL expressions
5. **Comprehensive Tests**: TDD approach throughout

The implementation follows KISS/YAGNI principles, adding complexity only where necessary while providing a foundation for future enhancements.
