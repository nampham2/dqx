"""DQL interpreter - executes parsed DQL against DQX runtime."""

from __future__ import annotations

import re
from calendar import monthrange
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import sympy as sp

from dqx.api import Context, MetricProvider, VerificationSuite, check
from dqx.common import ResultKey, SqlDataSource
from dqx.dql.ast import (
    Assertion,
    Check,
    DateExpr,
    DisableRule,
    Expr,
    Profile,
    Rule,
    ScaleRule,
    SetSeverityRule,
    Severity,
    Suite,
    Tunable,
)
from dqx.dql.errors import DQLError
from dqx.dql.parser import parse, parse_file
from dqx.functions import Coalesce
from dqx.orm.repositories import MetricDB
from dqx.tunables import TunableFloat, TunableInt, TunablePercent

# Month name to number mapping
MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# Weekday name to number mapping (0=Monday, 6=Sunday)
WEEKDAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass
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


@dataclass
class SuiteResults:
    """Results from executing a DQL suite."""

    suite_name: str
    assertions: list[AssertionResult]
    execution_date: date

    def all_passed(self) -> bool:
        """Check if all assertions passed."""
        return all(a.passed for a in self.assertions)

    @property
    def failures(self) -> list[AssertionResult]:
        """Get failed assertions."""
        return [a for a in self.assertions if not a.passed]

    @property
    def passes(self) -> list[AssertionResult]:
        """Get passed assertions."""
        return [a for a in self.assertions if a.passed]


class Interpreter:
    """Execute DQL files against DQX runtime.

    The interpreter translates DQL AST into DQX VerificationSuite API calls.
    All execution context (datasources, metric storage) is provided by the caller.

    Example:
        >>> db = InMemoryMetricDB()
        >>> datasources = {"orders": DuckDBDataSource(...)}
        >>> interp = Interpreter(db=db)
        >>> results = interp.run_file("suite.dql", datasources, date.today())
        >>> print(f"Passed: {len(results.passes)}, Failed: {len(results.failures)}")
    """

    def __init__(self, db: MetricDB):
        """Initialize interpreter with metric storage.

        Args:
            db: Metric database for storing/retrieving historical metrics
        """
        self.db = db
        self.tunables: dict[str, float] = {}
        self.active_profiles: list[Profile] = []
        self.current_check_name: str | None = None
        self.datasources: dict[str, SqlDataSource] = {}

    def run_file(
        self,
        path: str | Path,
        datasources: Mapping[str, SqlDataSource],
        date: date,
        tags: set[str] | None = None,
    ) -> SuiteResults:
        """Parse and execute a DQL file.

        Args:
            path: Path to .dql file
            datasources: Dataset name -> datasource mapping
            date: Execution date for ResultKey
            tags: Optional tags for ResultKey

        Returns:
            SuiteResults with all assertion outcomes

        Raises:
            DQLError: Parse error, validation error, or execution error
            KeyError: Dataset referenced in DQL not provided in datasources
        """
        suite_ast = parse_file(path)
        return self._execute(suite_ast, datasources, date, tags)

    def run_string(
        self,
        source: str,
        datasources: Mapping[str, SqlDataSource],
        date: date,
        tags: set[str] | None = None,
        filename: str | None = None,
    ) -> SuiteResults:
        """Parse and execute DQL source code.

        Same as run_file but accepts source string directly.
        """
        suite_ast = parse(source, filename=filename)
        return self._execute(suite_ast, datasources, date, tags)

    def _execute(
        self,
        suite_ast: Suite,
        datasources: Mapping[str, SqlDataSource],
        execution_date: date,
        tags: set[str] | None,
    ) -> SuiteResults:
        """Execute a parsed suite AST."""
        # Store datasources for access in helpers
        self.datasources = dict(datasources)  # Convert to dict for internal use

        # Validate datasources match what DQL expects (fail fast)
        self._validate_datasources(suite_ast, datasources)

        # Activate profiles for this execution date
        self._activate_profiles(suite_ast.profiles, execution_date)

        # Build DQX VerificationSuite from AST
        suite = self._build_suite(suite_ast)

        # Execute the suite
        # Convert set tags to dict tags (DQX expects dict[str, str])
        tags_dict: dict[str, str] = {tag: "" for tag in tags} if tags else {}
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
        # Build tunables
        tunables = self._build_tunables(suite_ast.tunables)

        # Build checks
        checks = [self._build_check(c, tunables) for c in suite_ast.checks]

        # Create suite with availability threshold if specified
        suite = VerificationSuite(
            checks=checks,
            db=self.db,
            name=suite_ast.name,
            tunables=tunables,
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
            raise DQLError(f"Cannot evaluate expression: {text}", loc=expr.loc)

    def _build_check(self, check_ast: Check, tunables: list[TunableFloat | TunablePercent | TunableInt]) -> Callable:
        """Convert DQL Check AST to Python check function."""
        check_name = check_ast.name
        assertions = check_ast.assertions

        # Create the check function
        @check(name=check_name)
        def dynamic_check(mp: MetricProvider, ctx: Context) -> None:
            """Generated check function from DQL."""
            # Store current check name for profile matching
            self.current_check_name = check_name

            # Execute each assertion
            for assertion_ast in assertions:
                self._build_assertion(assertion_ast, mp, ctx)

        return dynamic_check

    def _build_assertion(self, assertion_ast: Assertion, mp: MetricProvider, ctx: Context) -> None:
        """Convert DQL Assertion to ctx.assert_that() call."""
        # Check if assertion is disabled by active profiles
        if self._is_disabled(assertion_ast):
            return  # Skip this assertion

        # Evaluate metric expression
        metric_value = self._eval_metric_expr(assertion_ast.expr, mp)

        # Apply profile scaling if active
        metric_value = self._apply_profile_scaling(metric_value, assertion_ast)

        # Resolve severity (profile may override)
        severity = self._resolve_severity(assertion_ast)

        # Build assertion ready
        assert assertion_ast.name is not None, "Assertion must have a name"

        cost_annotation = self._get_cost_annotation(assertion_ast)
        cost_dict = {k: float(v) for k, v in cost_annotation.items()} if cost_annotation else None

        ready = ctx.assert_that(metric_value).where(
            name=assertion_ast.name,
            severity=severity.value,
            tags=set(assertion_ast.tags),
            experimental=self._has_annotation(assertion_ast, "experimental"),
            required=self._has_annotation(assertion_ast, "required"),
            cost=cost_dict,
        )

        # Apply condition
        self._apply_condition(ready, assertion_ast)

    def _eval_metric_expr(self, expr: Expr, mp: MetricProvider) -> Any:
        """Parse metric expression using sympy."""
        # Build namespace with metric functions
        namespace = self._build_metric_namespace(mp)

        # Substitute tunables
        expr_text = self._substitute_tunables(expr.text)

        # Parse with sympy
        try:
            return sp.sympify(expr_text, locals=namespace, evaluate=False)
        except Exception as e:  # pragma: no cover
            # Sympy parsing error (very rare with valid DQL)
            raise DQLError(
                f"Failed to parse metric expression: {expr.text}\n{e}",
                loc=expr.loc,
            )

    def _build_metric_namespace(self, mp: MetricProvider) -> dict[str, Any]:
        """Build sympy namespace with all metric and math functions."""

        def _to_str(arg: Any) -> str:
            """Convert sympy Symbol to string."""
            return str(arg) if isinstance(arg, sp.Symbol) else arg

        namespace = {
            # Math functions
            "abs": sp.Abs,
            "sqrt": sp.sqrt,
            "log": sp.log,
            "exp": sp.exp,
            "min": sp.Min,
            "max": sp.Max,
            # Base metrics - convert Symbol args to strings
            "num_rows": lambda **kw: mp.num_rows(**kw),
            "null_count": lambda col, **kw: mp.null_count(_to_str(col), **kw),
            "average": lambda col, **kw: mp.average(_to_str(col), **kw),
            "sum": lambda col, **kw: mp.sum(_to_str(col), **kw),
            "minimum": lambda col, **kw: mp.minimum(_to_str(col), **kw),
            "maximum": lambda col, **kw: mp.maximum(_to_str(col), **kw),
            "variance": lambda col, **kw: mp.variance(_to_str(col), **kw),
            "unique_count": lambda col, **kw: mp.unique_count(_to_str(col), **kw),
            "duplicate_count": lambda cols, **kw: mp.duplicate_count(cols, **kw),  # cols is a list
            "count_values": lambda col, val, **kw: mp.count_values(_to_str(col), val, **kw),
            "first": lambda col, **kw: mp.first(_to_str(col), **kw),
            # Utility functions
            "coalesce": lambda *args: Coalesce(*args),
            # RAW SQL ESCAPE HATCH
            "custom_sql": lambda expr: mp.custom_sql(_to_str(expr)),
        }
        return namespace

    def _substitute_tunables(self, expr_text: str) -> str:
        """Replace tunable names with their values in expression."""
        result = expr_text
        for name, value in self.tunables.items():
            # Simple string replacement (tunables are identifiers, safe)
            result = result.replace(name, str(value))
        return result

    def _apply_condition(self, ready: Any, assertion_ast: Assertion) -> None:
        """Apply the assertion condition to AssertionReady."""
        cond = assertion_ast.condition

        if cond == ">":
            assert assertion_ast.threshold is not None
            threshold = self._eval_simple_expr(assertion_ast.threshold)
            ready.is_gt(threshold)
        elif cond == ">=":
            assert assertion_ast.threshold is not None
            threshold = self._eval_simple_expr(assertion_ast.threshold)
            ready.is_geq(threshold)
        elif cond == "<":
            assert assertion_ast.threshold is not None
            threshold = self._eval_simple_expr(assertion_ast.threshold)
            ready.is_lt(threshold)
        elif cond == "<=":
            assert assertion_ast.threshold is not None
            threshold = self._eval_simple_expr(assertion_ast.threshold)
            ready.is_leq(threshold)
        elif cond == "==":
            assert assertion_ast.threshold is not None
            threshold = self._eval_simple_expr(assertion_ast.threshold)
            if assertion_ast.tolerance:
                ready.is_eq(threshold, tol=assertion_ast.tolerance)
            else:
                ready.is_eq(threshold)
        elif cond == "!=":
            assert assertion_ast.threshold is not None
            threshold = self._eval_simple_expr(assertion_ast.threshold)
            ready.is_neq(threshold)
        elif cond == "between":
            assert assertion_ast.threshold is not None
            assert assertion_ast.threshold_upper is not None
            lower = self._eval_simple_expr(assertion_ast.threshold)
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

    # === Profile Logic ===

    def _activate_profiles(self, profiles_ast: tuple[Profile, ...], execution_date: date) -> None:
        """Determine which profiles are active for execution date."""
        self.active_profiles = []

        for profile_ast in profiles_ast:
            # Resolve from/to dates
            from_date = self._resolve_date_expr(profile_ast.from_date, execution_date)
            to_date = self._resolve_date_expr(profile_ast.to_date, execution_date)

            # Check if execution date falls in range
            if from_date <= execution_date <= to_date:
                self.active_profiles.append(profile_ast)

    def _resolve_date_expr(self, date_expr: DateExpr, execution_date: date) -> date:
        """Resolve DQL date expression to concrete date."""
        if isinstance(date_expr.value, date):
            result = date_expr.value  # Literal like 2024-12-25
        else:
            result = self._eval_date_function(date_expr.value, execution_date)

        # Apply day offset
        if date_expr.offset != 0:
            result = result + timedelta(days=date_expr.offset)

        return result

    def _eval_date_function(self, func_str: str, execution_date: date) -> date:
        """Evaluate date function: nth_weekday(...), last_day_of_month(), etc."""
        # Parse: "function_name(arg1, arg2, ...)"
        match = re.match(r"(\w+)\((.*)\)", func_str)
        if not match:  # pragma: no cover
            # Parser ensures valid date function format
            raise DQLError(f"Invalid date function: {func_str}")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # Dispatch to implementation
        if func_name == "nth_weekday":
            return self._nth_weekday(args_str, execution_date)
        elif func_name == "last_day_of_month":
            return self._last_day_of_month(execution_date)
        elif func_name in MONTH_NAMES:  # january, february, etc.
            return self._month_day(func_name, args_str, execution_date)
        else:  # pragma: no cover
            # Parser validates date function names
            raise DQLError(f"Unknown date function: {func_name}")

    def _nth_weekday(self, args_str: str, execution_date: date) -> date:
        """Implement nth_weekday(month, day, n).

        Examples:
            nth_weekday(november, thursday, 4)  # 4th Thursday in November
            nth_weekday(december, monday, 1)    # 1st Monday in December
        """
        # Parse args: month name, weekday name, occurrence number
        args = [a.strip() for a in args_str.split(",")]
        month_name = args[0]  # e.g., "november"
        weekday_name = args[1]  # e.g., "thursday"
        n = int(args[2])  # e.g., 4

        # Convert month name to number (1-12)
        month_num = MONTH_NAMES.get(month_name.lower())
        if month_num is None:
            raise DQLError(f"Unknown month name: {month_name}")

        # Convert weekday name to number (0=Monday, 6=Sunday)
        weekday_num = WEEKDAY_NAMES.get(weekday_name.lower())
        if weekday_num is None:
            raise DQLError(f"Unknown weekday name: {weekday_name}")

        # Find nth occurrence in execution_date's year
        year = execution_date.year
        first_day = date(year, month_num, 1)

        # Find first occurrence of the weekday
        days_ahead = (weekday_num - first_day.weekday()) % 7
        first_occurrence = first_day + timedelta(days=days_ahead)

        # Add weeks to get nth occurrence
        nth_occurrence = first_occurrence + timedelta(weeks=n - 1)

        # Verify it's still in the same month
        if nth_occurrence.month != month_num:
            raise DQLError(f"No {n}th {weekday_name} in {month_name}")

        return nth_occurrence

    def _last_day_of_month(self, execution_date: date) -> date:
        """Implement last_day_of_month()."""
        year = execution_date.year
        month = execution_date.month

        # Get last day of month
        _, last_day = monthrange(year, month)
        return date(year, month, last_day)

    def _month_day(self, month_name: str, args_str: str, execution_date: date) -> date:
        """Implement month(day) functions like december(25)."""
        day = int(args_str.strip())
        month_num = MONTH_NAMES[month_name.lower()]
        year = execution_date.year

        return date(year, month_num, day)

    def _apply_profile_scaling(self, metric_value: Any, assertion_ast: Assertion) -> Any:
        """Apply scale multipliers from active profiles."""
        multiplier = 1.0

        for profile in self.active_profiles:
            for rule in profile.rules:
                if isinstance(rule, ScaleRule):
                    if self._rule_matches_assertion(rule, assertion_ast):
                        multiplier *= rule.multiplier

        if multiplier != 1.0:
            return metric_value * multiplier
        return metric_value

    def _is_disabled(self, assertion_ast: Assertion) -> bool:
        """Check if assertion disabled by any active profile."""
        for profile in self.active_profiles:
            for rule in profile.rules:
                if isinstance(rule, DisableRule):
                    if self._rule_matches_assertion(rule, assertion_ast):
                        return True
        return False

    def _resolve_severity(self, assertion_ast: Assertion) -> Severity:
        """Resolve severity with profile overrides."""
        severity = assertion_ast.severity

        # Check for severity overrides in active profiles
        for profile in self.active_profiles:
            for rule in profile.rules:
                if isinstance(rule, SetSeverityRule):
                    if self._rule_matches_assertion(rule, assertion_ast):
                        severity = rule.severity

        return severity

    def _rule_matches_assertion(self, rule: Rule, assertion_ast: Assertion) -> bool:
        """Check if a profile rule applies to an assertion."""
        # DisableRule uses target_type/target_name
        if isinstance(rule, DisableRule):
            if rule.target_type == "check":
                return rule.target_name == self.current_check_name
            elif rule.target_type == "assertion":
                # Matches if assertion name matches and optionally check name matches
                if assertion_ast.name == rule.target_name:
                    if rule.in_check:
                        return self.current_check_name == rule.in_check
                    return True  # pragma: no cover - Parser requires 'in' clause
                return False
            return False  # pragma: no cover - Parser validates target_type

        # ScaleRule and SetSeverityRule use selector_type/selector_name
        if hasattr(rule, "selector_type"):
            if rule.selector_type == "tag":
                return rule.selector_name in assertion_ast.tags
            elif rule.selector_type == "check":
                return rule.selector_name == self.current_check_name

        return False  # pragma: no cover - Parser validates rule types

    # === Annotation Helpers ===

    def _has_annotation(self, assertion_ast: Assertion, name: str) -> bool:
        """Check if assertion has a specific annotation."""
        return any(ann.name == name for ann in assertion_ast.annotations)

    def _get_cost_annotation(self, assertion_ast: Assertion) -> dict[str, int] | None:
        """Extract cost annotation args if present."""
        for ann in assertion_ast.annotations:
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
            assertions=assertions,
            execution_date=execution_date,
        )
