"""DQL interpreter - executes parsed DQL against DQX runtime."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, overload

import sympy as sp

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey, SqlDataSource
from dqx.dql.ast import (
    Assertion,
    Check,
    Collection,
    Expr,
    Suite,
    Tunable,
)
from dqx.dql.errors import DQLError
from dqx.dql.parser import parse, parse_file
from dqx.functions import Coalesce
from dqx.orm.repositories import MetricDB
from dqx.tunables import TunableFloat, TunableInt, TunablePercent


@dataclass(frozen=True)
class AssertionResult:
    """Result from executing a single DQL assertion."""

    check_name: str
    assertion_name: str | None
    passed: bool
    metric_value: float | None
    threshold: float | None
    condition: str
    severity: str
    reason: str | None  # Failure reason if not passed


@dataclass(frozen=True)
class SuiteResults:
    """Results from executing a DQL suite."""

    suite_name: str
    assertions: tuple[AssertionResult, ...]
    execution_date: date

    def all_passed(self) -> bool:
        """Check if all assertions passed."""
        return all(a.passed for a in self.assertions)

    @property
    def failures(self) -> tuple[AssertionResult, ...]:
        """Get failed assertions."""
        return tuple(a for a in self.assertions if not a.passed)

    @property
    def passes(self) -> tuple[AssertionResult, ...]:
        """Get passed assertions."""
        return tuple(a for a in self.assertions if a.passed)


class Interpreter:
    """Execute DQL files against DQX runtime.

    The interpreter translates DQL AST into DQX VerificationSuite API calls.
    All execution context (datasources, metric storage) is provided by the caller.

    Example:
        >>> db = InMemoryMetricDB()
        >>> datasources = {"orders": DuckDBDataSource(...)}
        >>> interp = Interpreter(db=db)
        >>> results = interp.run(Path("suite.dql"), datasources, date.today())
        >>> print(f"Passed: {len(results.passes)}, Failed: {len(results.failures)}")
    """

    def __init__(self, db: MetricDB):
        """Initialize interpreter with metric storage.

        Args:
            db: Metric database for storing/retrieving historical metrics
        """
        self.db = db
        self.tunables: dict[str, float] = {}
        self.current_check_name: str | None = None
        self.datasources: dict[str, SqlDataSource] = {}

    @overload
    def run(
        self,
        source: Path,
        datasources: Mapping[str, SqlDataSource],
        date: date,
        tags: dict[str, str] | None = None,
    ) -> SuiteResults:
        """Parse and execute a DQL file.

        Args:
            source: Path to .dql file
            datasources: Dataset name -> datasource mapping
            date: Execution date for ResultKey
            tags: Optional tags for ResultKey

        Returns:
            SuiteResults with all assertion outcomes

        Raises:
            DQLError: Parse error, validation error, execution error, or missing datasources
        """
        ...

    @overload
    def run(
        self,
        source: str,
        datasources: Mapping[str, SqlDataSource],
        date: date,
        tags: dict[str, str] | None = None,
        *,
        filename: str | None = None,
    ) -> SuiteResults:
        """Parse and execute DQL source code.

        Args:
            source: DQL source code string
            datasources: Dataset name -> datasource mapping
            date: Execution date for ResultKey
            tags: Optional tags for ResultKey
            filename: Optional filename for error messages

        Returns:
            SuiteResults with all assertion outcomes

        Raises:
            DQLError: Parse error, validation error, execution error, or missing datasources
        """
        ...

    def run(
        self,
        source: str | Path,
        datasources: Mapping[str, SqlDataSource],
        date: date,
        tags: dict[str, str] | None = None,
        *,
        filename: str | None = None,
    ) -> SuiteResults:
        """Parse and execute DQL from file or string.

        Dispatches to appropriate parser based on source type:
        - Path objects are parsed as DQL files
        - Strings are parsed as DQL source code

        Args:
            source: Path to .dql file, or DQL source code string
            datasources: Dataset name -> datasource mapping
            date: Execution date for ResultKey
            tags: Optional tags for ResultKey
            filename: Optional filename for error messages (only used when source is str)

        Returns:
            SuiteResults with all assertion outcomes

        Raises:
            DQLError: Parse error, validation error, execution error, or missing datasources
        """
        if isinstance(source, Path):
            suite_ast = parse_file(source)
        else:
            suite_ast = parse(source, filename=filename)

        return self._execute(suite_ast, datasources, date, tags)

    def _execute(
        self,
        suite_ast: Suite,
        datasources: Mapping[str, SqlDataSource],
        execution_date: date,
        tags: dict[str, str] | None,
    ) -> SuiteResults:
        """Execute a parsed suite AST."""
        # Store datasources for access in helpers
        self.datasources = dict(datasources)  # Convert to dict for internal use

        # Validate datasources match what DQL expects (fail fast)
        self._validate_datasources(suite_ast, datasources)

        # Build DQX VerificationSuite from AST
        suite = self._build_suite(suite_ast)

        # Execute the suite
        tags_dict = tags if tags is not None else {}
        key = ResultKey(execution_date, tags_dict)
        suite.run(list(datasources.values()), key)

        # Collect and return results
        return self._collect_results(suite, suite_ast.name, execution_date)

    def _validate_datasources(self, suite_ast: Suite, datasources: Mapping[str, SqlDataSource]) -> None:
        """Ensure all datasets referenced in DQL are provided. Fail fast on missing datasets."""
        required_datasets: set[str] = set()
        for check_item in suite_ast.checks:
            required_datasets.update(check_item.datasets)

        missing = required_datasets - datasources.keys()
        if missing:
            raise DQLError(
                f"Missing datasources: {sorted(missing)}. "
                f"DQL requires {sorted(required_datasets)}, "
                f"but only {sorted(datasources.keys())} were provided.",
                loc=suite_ast.loc,
            )

    def _build_suite(self, suite_ast: Suite) -> VerificationSuite:
        """Convert DQL Suite AST into DQX VerificationSuite."""
        # Build tunables (creates Tunable objects that will be auto-discovered)
        tunables = self._build_tunables(suite_ast.tunables)

        # Build checks (tunables are used in expressions, will be auto-discovered)
        checks = [self._build_check(c, tunables) for c in suite_ast.checks]

        # Create suite - tunables will be auto-discovered from check expressions
        suite = VerificationSuite(
            checks=checks,
            db=self.db,
            name=suite_ast.name,
            data_av_threshold=suite_ast.availability_threshold or 0.9,
        )

        return suite

    def _build_tunables(self, tunables_ast: tuple[Tunable, ...]) -> list[TunableFloat | TunablePercent | TunableInt]:
        """Convert DQL tunables to DQX Tunable objects."""
        result: list[TunableFloat | TunablePercent | TunableInt] = []
        for t in tunables_ast:
            # Evaluate expressions to get numeric values
            min_val = self._eval_simple_expr(t.bounds[0])
            max_val = self._eval_simple_expr(t.bounds[1])
            value = self._eval_simple_expr(t.value)

            # Store for substitution
            self.tunables[t.name] = value

            # Determine tunable type based on value
            if 0 <= value <= 1 and 0 <= min_val <= 1 and 0 <= max_val <= 1:
                # Percentage tunable
                result.append(TunablePercent(name=t.name, value=value, bounds=(min_val, max_val)))
            elif isinstance(value, int) and isinstance(min_val, int) and isinstance(max_val, int):
                # Integer tunable
                result.append(TunableInt(name=t.name, value=value, bounds=(min_val, max_val)))
            else:
                # Float tunable
                result.append(TunableFloat(name=t.name, value=value, bounds=(min_val, max_val)))

        return result

    def _eval_simple_expr(self, expr: Expr) -> float:
        """Evaluate simple numeric expressions (for tunable bounds and values)."""
        # Substitute tunables first
        text = self._substitute_tunables(expr.text.strip())

        # Handle percentages (already converted by parser, but keep for safety)
        if text.endswith("%"):  # pragma: no cover
            # Parser converts percentages to decimals
            return float(text[:-1]) / 100

        # Handle numeric literals - preserve int vs float
        try:
            # Try int first
            if "." not in text:
                return int(text)
            return float(text)
        except ValueError:  # pragma: no cover
            # Parser validates numeric literals
            raise DQLError(f"Cannot evaluate expression: {text}", loc=expr.loc) from None

    def _build_check(self, check_ast: Check, _tunables: list[TunableFloat | TunablePercent | TunableInt]) -> Callable:
        """Convert DQL Check AST to Python check function."""
        check_name = check_ast.name
        assertions = check_ast.assertions

        # Create the check function
        @check(name=check_name, datasets=list(check_ast.datasets))
        def dynamic_check(mp: MetricProvider, ctx: Context) -> None:
            """Generated check function from DQL."""
            # Store current check name for context
            self.current_check_name = check_name

            # Execute each assertion
            for assertion_ast in assertions:
                self._build_statement(assertion_ast, mp, ctx)

        return dynamic_check

    def _build_statement(self, statement: Assertion | Collection, mp: MetricProvider, ctx: Context) -> None:
        """Convert DQL Assertion or Collection to ctx.assert_that() call."""
        # Dispatch based on type
        if isinstance(statement, Collection):
            self._build_collection(statement, mp, ctx)
        else:
            self._build_assertion(statement, mp, ctx)

    def _setup_assertion_ready(self, statement: Assertion | Collection, mp: MetricProvider, ctx: Context) -> Any | None:
        """Common setup for assertions and collections.

        Returns AssertionReady object or None if statement is disabled.

        Args:
            statement: Assertion or Collection AST node
            mp: MetricProvider for evaluating metric expressions
            ctx: Context for creating assertions

        Returns:
            AssertionReady object if statement should be processed, None if disabled
        """
        # Evaluate metric expression
        metric_value = self._eval_metric_expr(statement.expr, mp)

        # Use severity from DQL statement
        severity = statement.severity

        # Validate name is present
        if statement.name is None:  # pragma: no cover
            # Parser should have caught this
            raise DQLError("Statement must have a name", loc=statement.loc)

        # Extract cost annotation
        cost_annotation = self._get_cost_annotation(statement)
        cost_dict = {k: float(v) for k, v in cost_annotation.items()} if cost_annotation else None

        # Build and return assertion ready object
        return ctx.assert_that(metric_value).where(
            name=statement.name,
            severity=severity.value,
            tags=set(statement.tags),
            experimental=self._has_annotation(statement, "experimental"),
            required=self._has_annotation(statement, "required"),
            cost=cost_dict,
        )

    def _build_assertion(self, assertion_ast: Assertion, mp: MetricProvider, ctx: Context) -> None:
        """Convert DQL Assertion to ctx.assert_that() call."""
        ready = self._setup_assertion_ready(assertion_ast, mp, ctx)
        if ready is None:  # pragma: no cover
            return

        # Apply condition
        self._apply_condition(ready, assertion_ast)

    def _build_collection(self, collection_ast: Collection, mp: MetricProvider, ctx: Context) -> None:
        """Convert DQL Collection to ctx.assert_that(...).noop() call."""
        ready = self._setup_assertion_ready(collection_ast, mp, ctx)
        if ready is None:  # pragma: no cover
            return

        # Call noop() instead of applying a condition
        ready.noop()

    def _eval_metric_expr(self, expr: Expr, mp: MetricProvider) -> Any:
        """Parse metric expression using sympy, with special handling for extensions."""
        # Check if expression contains extension functions with named params
        expr_text = self._substitute_tunables(expr.text)

        # Handle stddev specially since it has named params that sympy doesn't understand
        if "stddev(" in expr_text and (", n=" in expr_text or ", offset=" in expr_text):
            return self._handle_stddev_extension(expr_text, mp)

        # Build namespace with metric functions
        namespace = self._build_metric_namespace(mp)

        # Parse with sympy
        try:
            return sp.sympify(expr_text, locals=namespace, evaluate=False)
        except (sp.SympifyError, TypeError, ValueError) as e:  # pragma: no cover
            # Sympy parsing error (very rare with valid DQL)
            raise DQLError(
                f"Failed to parse metric expression: {expr.text}\n{e}",
                loc=expr.loc,
            ) from e

    def _handle_stddev_extension(self, expr_text: str, mp: MetricProvider) -> Any:
        """Handle stddev extension function with named parameters.

        Parses stddev calls with optional offset and required n parameters in any order:
        - stddev(expr, n=7)
        - stddev(expr, offset=1, n=7)
        - stddev(expr, n=7, offset=1)

        Args:
            expr_text: Expression string containing stddev call
            mp: MetricProvider for accessing extension functions

        Returns:
            Result of mp.ext.stddev() call via sp.sympify()
        """
        # Find the matching closing paren for stddev( by counting parentheses
        # This properly handles nested function calls like stddev(day_over_day(avg(x)), n=7)
        stddev_start = expr_text.find("stddev(")
        if stddev_start == -1:  # pragma: no cover
            # Should not happen - caller checks for stddev( first
            namespace = self._build_metric_namespace(mp)
            return sp.sympify(expr_text, locals=namespace, evaluate=False)

        # Start after "stddev("
        pos = stddev_start + 7
        paren_count = 1
        inner_start = pos

        # Find the matching closing paren
        while pos < len(expr_text) and paren_count > 0:
            if expr_text[pos] == "(":
                paren_count += 1
            elif expr_text[pos] == ")":
                paren_count -= 1
            pos += 1

        if paren_count != 0:  # pragma: no cover - defensive fallback
            # Malformed expression - fallback to normal parsing
            namespace = self._build_metric_namespace(mp)
            return sp.sympify(expr_text, locals=namespace, evaluate=False)

        # Extract everything inside stddev(...)
        inner_content = expr_text[inner_start : pos - 1]

        # Split inner content by commas, being careful about nested functions
        # Look for the first comma that's at the top level (not inside nested parens)
        parts = []
        current_part = []
        paren_depth = 0

        for char in inner_content:
            if char == "(":
                paren_depth += 1
                current_part.append(char)
            elif char == ")":
                paren_depth -= 1
                current_part.append(char)
            elif char == "," and paren_depth == 0:
                parts.append("".join(current_part).strip())
                current_part = []
            else:
                current_part.append(char)

        # Don't forget the last part
        if current_part:
            parts.append("".join(current_part).strip())

        # First part is the inner expression
        inner_expr_text = parts[0] if parts else ""

        # Extract offset and n parameters from remaining parts
        offset = 0  # Default offset
        n = None  # Will be required if params exist

        for part in parts[1:]:
            # Match offset=N or n=N with optional whitespace
            offset_match = re.search(r"offset\s*=\s*(\d+)", part)
            n_match = re.search(r"n\s*=\s*(\d+)", part)

            if offset_match:
                offset = int(offset_match.group(1))
            if n_match:
                n = int(n_match.group(1))

        # If params exist but n is not found, this shouldn't happen with valid DQL
        # The grammar should enforce n parameter when using stddev with params
        if len(parts) > 1 and n is None:  # pragma: no cover
            raise ValueError(f"stddev requires 'n' parameter: {expr_text}")

        # Parse the inner expression via _build_metric_namespace and sp.sympify
        # Don't recursively call _eval_metric_expr to avoid infinite loop
        namespace = self._build_metric_namespace(mp)
        inner_metric = sp.sympify(inner_expr_text, locals=namespace, evaluate=False)

        # Call mp.ext.stddev with parsed parameters
        # Note: stddev without params is handled by normal sympy evaluation
        if n is not None:
            result = mp.ext.stddev(inner_metric, offset=offset, n=n)
        else:  # pragma: no cover - stddev without params shouldn't reach here
            # This path shouldn't be reached - stddev without params won't match
            # the condition in _eval_metric_expr that calls this method
            namespace = self._build_metric_namespace(mp)
            return sp.sympify(expr_text, locals=namespace, evaluate=False)

        return result

    def _build_metric_namespace(self, mp: MetricProvider) -> dict[str, Any]:
        """Build sympy namespace with all metric and math functions."""

        def _to_str(arg: Any) -> str:
            """Convert sympy Symbol to string."""
            return str(arg) if isinstance(arg, sp.Symbol) else arg

        def _convert_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
            """Convert sympy types in kwargs to Python primitives.

            Handles:
            - sp.Integer -> int (lag=1, n=7, offset=1)
            - sp.Float -> float
            - sp.Symbol -> str (dataset=ds1)
            - Other types pass through unchanged
            """
            result: dict[str, Any] = {}
            for key, value in kw.items():
                if isinstance(value, sp.Basic):
                    # Convert sympy numbers to Python int/float
                    if value.is_Integer:
                        result[key] = int(value)  # type: ignore[arg-type]
                    elif value.is_Float or value.is_Rational:  # pragma: no cover - rare sympy output
                        result[key] = float(value)  # type: ignore[arg-type]  # pragma: no cover
                    elif isinstance(value, sp.Symbol):
                        # For symbols (like dataset names), convert to string
                        result[key] = str(value)
                    else:  # pragma: no cover - defensive fallback for unknown sympy types
                        # For other sympy types, try to extract value
                        try:  # pragma: no cover
                            result[key] = float(value)  # type: ignore[arg-type]  # pragma: no cover
                        except (TypeError, AttributeError):  # pragma: no cover
                            result[key] = str(value)  # pragma: no cover
                else:
                    result[key] = value  # pragma: no cover - passthrough of non-sympy values
            return result

        def _convert_list_arg(cols: Any) -> list[str]:
            """Convert list of Symbols/tokens to list of strings.

            Handles:
            - [Symbol('name')] -> ['name']
            - [Symbol('id'), Symbol('date')] -> ['id', 'date']
            - Symbol('name') -> ['name'] (single column case)
            """
            if isinstance(cols, list):
                return [_to_str(item) for item in cols]
            elif isinstance(cols, tuple):  # pragma: no cover - tuples not produced by parser
                return [_to_str(item) for item in cols]  # pragma: no cover
            else:  # pragma: no cover - single column fallback
                # Single column passed without list brackets
                return [_to_str(cols)]  # pragma: no cover

        def _convert_value(val: Any) -> int | str | bool:
            """Convert value argument to proper Python type for count_values.

            Handles:
            - sp.Integer/Zero/One -> int
            - sp.Float -> int (rounded)
            - sp.Symbol -> str
            - bool/int/str -> unchanged
            - float -> int (rounded)
            """
            if isinstance(val, sp.Basic):
                # Convert sympy types
                if val.is_Integer:
                    return int(val)  # type: ignore[arg-type]
                elif val.is_Float or val.is_Rational:  # pragma: no cover - rare sympy output
                    # Convert float to int for count_values
                    return int(float(val))  # type: ignore[arg-type]  # pragma: no cover
                elif isinstance(val, sp.Symbol):  # pragma: no cover - symbols handled elsewhere
                    return str(val)  # pragma: no cover
                else:  # pragma: no cover - defensive fallback for unknown sympy types
                    # Try to extract numeric value
                    try:  # pragma: no cover
                        return int(val)  # type: ignore[arg-type]  # pragma: no cover
                    except (TypeError, AttributeError):  # pragma: no cover
                        return str(val)  # pragma: no cover
            elif isinstance(val, float):
                # Convert float to int for count_values
                return int(val)  # pragma: no cover - float values converted at parse time
            elif isinstance(val, (int, str, bool)):
                return val
            else:  # pragma: no cover - defensive fallback for unknown types
                return str(val)  # pragma: no cover

        namespace = {
            # Math functions
            "abs": sp.Abs,
            "sqrt": sp.sqrt,
            "log": sp.log,
            "exp": sp.exp,
            "min": sp.Min,
            "max": sp.Max,
            # Base metrics - convert Symbol args to strings and kwargs to Python types
            "num_rows": lambda **kw: mp.num_rows(**_convert_kwargs(kw)),
            "null_count": lambda col, **kw: mp.null_count(_to_str(col), **_convert_kwargs(kw)),
            "average": lambda col, **kw: mp.average(_to_str(col), **_convert_kwargs(kw)),
            "sum": lambda col, **kw: mp.sum(_to_str(col), **_convert_kwargs(kw)),
            "minimum": lambda col, **kw: mp.minimum(_to_str(col), **_convert_kwargs(kw)),
            "maximum": lambda col, **kw: mp.maximum(_to_str(col), **_convert_kwargs(kw)),
            "variance": lambda col, **kw: mp.variance(_to_str(col), **_convert_kwargs(kw)),
            "unique_count": lambda col, **kw: mp.unique_count(_to_str(col), **_convert_kwargs(kw)),
            "duplicate_count": lambda cols, **kw: mp.duplicate_count(_convert_list_arg(cols), **_convert_kwargs(kw)),
            "count_values": lambda col, val, **kw: mp.count_values(
                _to_str(col), _convert_value(val), **_convert_kwargs(kw)
            ),
            "first": lambda col, **kw: mp.first(_to_str(col), **_convert_kwargs(kw)),
            # Utility functions
            "coalesce": lambda *args: Coalesce(*args),
            # Extension functions
            "day_over_day": lambda metric, **kw: mp.ext.day_over_day(metric, **_convert_kwargs(kw)),
            "week_over_week": lambda metric, **kw: mp.ext.week_over_week(metric, **_convert_kwargs(kw)),
            # Note: stddev with n parameter is handled specially in _handle_stddev_extension
            # RAW SQL ESCAPE HATCH
            "custom_sql": lambda expr: mp.custom_sql(_to_str(expr)),
        }
        return namespace

    def _substitute_tunables(self, expr_text: str) -> str:
        """Replace tunable names with their values in expression.

        Uses word-boundary regex to avoid corrupting identifiers when one tunable
        name is a prefix of another (e.g., MAX and MAX_VALUE).
        """
        result = expr_text
        # Sort by length descending to handle longest matches first
        for name in sorted(self.tunables.keys(), key=len, reverse=True):
            value = self.tunables[name]
            # Use word boundaries to match only complete identifiers
            pattern = r"\b" + re.escape(name) + r"\b"
            result = re.sub(pattern, str(value), result)
        return result

    def _apply_condition(self, ready: Any, assertion_ast: Assertion) -> None:
        """Apply the assertion condition to AssertionReady."""
        cond = assertion_ast.condition

        # Validate threshold is present for conditions that require it
        requires_threshold = cond in (">", ">=", "<", "<=", "==", "!=", "between")
        if requires_threshold and assertion_ast.threshold is None:  # pragma: no cover
            # Parser ensures threshold is present for these conditions
            raise DQLError(f"Condition '{cond}' requires a threshold", assertion_ast.loc)

        if cond == ">":
            threshold = self._eval_simple_expr(assertion_ast.threshold)  # type: ignore[arg-type]
            ready.is_gt(threshold)
        elif cond == ">=":
            threshold = self._eval_simple_expr(assertion_ast.threshold)  # type: ignore[arg-type]
            ready.is_geq(threshold)
        elif cond == "<":
            threshold = self._eval_simple_expr(assertion_ast.threshold)  # type: ignore[arg-type]
            ready.is_lt(threshold)
        elif cond == "<=":
            threshold = self._eval_simple_expr(assertion_ast.threshold)  # type: ignore[arg-type]
            ready.is_leq(threshold)
        elif cond == "==":
            threshold = self._eval_simple_expr(assertion_ast.threshold)  # type: ignore[arg-type]
            if assertion_ast.tolerance:
                ready.is_eq(threshold, tol=assertion_ast.tolerance)
            else:
                ready.is_eq(threshold)
        elif cond == "!=":
            threshold = self._eval_simple_expr(assertion_ast.threshold)  # type: ignore[arg-type]
            ready.is_neq(threshold)
        elif cond == "between":
            if assertion_ast.threshold_upper is None:  # pragma: no cover
                # Parser ensures upper threshold is present for between
                raise DQLError("Condition 'between' requires upper threshold", assertion_ast.loc)
            lower = self._eval_simple_expr(assertion_ast.threshold)  # type: ignore[arg-type]
            upper = self._eval_simple_expr(assertion_ast.threshold_upper)
            ready.is_between(lower, upper)
        elif cond == "is":
            if assertion_ast.keyword == "positive":
                ready.is_positive()
            elif assertion_ast.keyword == "negative":
                ready.is_negative()
            else:  # pragma: no cover
                # Parser ensures only 'positive' or 'negative' keywords
                raise DQLError(f"Unknown keyword: {assertion_ast.keyword}", assertion_ast.loc)
        else:  # pragma: no cover
            # Parser validates all conditions
            raise DQLError(f"Unknown condition: {cond}", assertion_ast.loc)

    # === Annotation Helpers ===

    def _has_annotation(self, statement_ast: Assertion | Collection, name: str) -> bool:
        """Check if assertion or collection has a specific annotation."""
        return any(ann.name == name for ann in statement_ast.annotations)

    def _get_cost_annotation(self, statement_ast: Assertion | Collection) -> dict[str, int] | None:
        """Extract cost annotation args if present."""
        for ann in statement_ast.annotations:
            if ann.name == "cost":
                # Convert args to expected format
                return {
                    "fp": int(ann.args.get("false_positive", 1)),
                    "fn": int(ann.args.get("false_negative", 1)),
                }
        return None

    # === Results Collection ===

    def _collect_results(self, suite: VerificationSuite, suite_name: str, execution_date: date) -> SuiteResults:
        """Collect results from executed suite."""
        from returns.result import Failure, Success

        # Get results from DQX suite
        dqx_results = suite.collect_results()

        # Convert to our result format
        assertions = []
        for result in dqx_results:
            # Extract metric value if available
            metric_value = None
            reason = None

            match result.metric:
                case Success(value):
                    metric_value = value
                case Failure(failures):  # pragma: no cover
                    # Metric evaluation failure (rare - data issues)
                    reason = "; ".join(str(f) for f in failures)

            assertions.append(
                AssertionResult(
                    check_name=result.check,
                    assertion_name=result.assertion,
                    passed=(result.status == "PASSED"),
                    metric_value=metric_value,
                    threshold=None,  # Not easily available from DQX result
                    condition=result.expression or "unknown",
                    severity=result.severity,
                    reason=reason,
                )
            )

        return SuiteResults(
            suite_name=suite_name,
            assertions=tuple(assertions),
            execution_date=execution_date,
        )
