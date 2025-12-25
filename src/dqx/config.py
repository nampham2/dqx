"""YAML/JSON configuration parser for DQX verification suites.

This module provides functions to load verification suites from YAML or JSON
configuration files, enabling declarative definition of data quality checks.

Example:
    suite = load_suite("suite.yaml", db=my_db)
    suite.run([datasource], key)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema
import sympy as sp
import yaml

from dqx import functions
from dqx.common import DQXError, SeverityLevel, SymbolicValidator
from dqx.profiles import (
    AssertionSelector,
    HolidayProfile,
    Profile,
    Rule,
    TagSelector,
)

if TYPE_CHECKING:
    from dqx.provider import MetricProvider


# =============================================================================
# JSON Schema Validation
# =============================================================================

# Default schema path relative to this module
_SCHEMA_PATH = Path(__file__).parent / "suite.schema.json"


@lru_cache(maxsize=1)
def _load_schema_cached(schema_path: Path) -> dict[str, Any]:
    """Load and cache the JSON schema (internal, use get_schema instead).

    Args:
        schema_path: Path to schema file.

    Returns:
        Parsed JSON schema dictionary

    Raises:
        DQXError: If schema file cannot be loaded
    """
    path = schema_path

    if not path.exists():  # pragma: no cover
        raise DQXError(f"JSON schema not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        raise DQXError(f"Failed to load JSON schema: {e}") from e


def _load_schema(schema_path: Path | None = None) -> dict[str, Any]:
    """Load the JSON schema, normalizing None to default path for caching.

    Args:
        schema_path: Path to schema file. Uses default if None.

    Returns:
        Parsed JSON schema dictionary
    """
    return _load_schema_cached(schema_path or _SCHEMA_PATH)


def validate_config_schema(
    path_or_content: str | Path,
    schema_path: Path | None = None,
) -> list[str]:
    """Validate a configuration file or string against the JSON schema.

    Args:
        path_or_content: Path to YAML file or YAML content string
        schema_path: Optional custom schema path

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> errors = validate_config_schema("suite.yaml")
        >>> if errors:
        ...     for e in errors:
        ...         print(f"Schema error: {e}")
    """
    # Load the config content
    # Determine if input is a file path or YAML content string
    # A Path object is always treated as a file path
    # A string is treated as a file path only if it exists and doesn't look like YAML
    is_file_path = isinstance(path_or_content, Path)
    if not is_file_path and isinstance(path_or_content, str):
        # Short strings without newlines that exist as files are treated as paths
        # Strings with newlines or YAML-like content are treated as YAML strings
        path = Path(path_or_content)
        is_file_path = path.exists() and "\n" not in path_or_content

    if is_file_path:
        path = Path(path_or_content) if not isinstance(path_or_content, Path) else path_or_content
        try:
            with open(path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        except FileNotFoundError:
            return [f"Configuration file not found: {path}"]
        except yaml.YAMLError as e:
            return [f"Failed to parse YAML: {e}"]
    else:
        # Assume it's a YAML string
        try:
            config_dict = yaml.safe_load(str(path_or_content))
        except yaml.YAMLError as e:
            return [f"Failed to parse YAML: {e}"]

    return validate_dict_schema(config_dict, schema_path)


def validate_dict_schema(
    config_dict: dict[str, Any],
    schema_path: Path | None = None,
) -> list[str]:
    """Validate a configuration dictionary against the JSON schema.

    Args:
        config_dict: Configuration dictionary to validate
        schema_path: Optional custom schema path

    Returns:
        List of validation error messages (empty if valid)
    """
    if not config_dict:
        return ["Configuration cannot be empty"]

    try:
        schema = _load_schema(schema_path)
    except DQXError as e:  # pragma: no cover
        return [str(e)]

    validator = jsonschema.Draft202012Validator(schema)
    errors = []

    for error in validator.iter_errors(config_dict):
        # Format error message with path
        path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
        errors.append(f"{path}: {error.message}")

    return errors


# =============================================================================
# Metric Expression Parser
# =============================================================================

# Mapping from YAML function names to MetricProvider methods
METRIC_FUNCTIONS = {
    "num_rows": "num_rows",
    "first": "first",
    "average": "average",
    "minimum": "minimum",
    "maximum": "maximum",
    "sum": "sum",
    "null_count": "null_count",
    "variance": "variance",
    "duplicate_count": "duplicate_count",
    "count_values": "count_values",
    "unique_count": "unique_count",
    "custom_sql": "custom_sql",
}

# Extended metric functions
EXTENDED_METRIC_FUNCTIONS = {
    "day_over_day": "day_over_day",
    "week_over_week": "week_over_week",
    "stddev": "stddev",
}

# Known metric parameters (not custom parameters)
KNOWN_PARAMS = {"dataset", "lag", "column", "columns", "values", "sql_expression", "offset", "n"}

# Whitelisted math functions for sympy
MATH_FUNCTIONS = {
    "abs": sp.Abs,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "exp": sp.exp,
    "min": sp.Min,
    "max": sp.Max,
}


class MetricExpressionParser:
    """Parses metric expression strings into symbolic expressions.

    The parser uses a custom sympy namespace to handle metric function calls
    and arithmetic operations.
    """

    def __init__(self, provider: MetricProvider) -> None:
        self._provider = provider
        self._symbol_map: dict[str, sp.Symbol] = {}

    def parse(self, expr_str: str) -> sp.Expr:
        """Parse a metric expression string into a sympy expression.

        Args:
            expr_str: Metric expression like "average(price)" or
                     "null_count(email) / num_rows()"

        Returns:
            Sympy expression that can be evaluated

        Raises:
            DQXError: If parsing fails
        """
        # Build namespace for sympify
        namespace = self._build_namespace()

        try:
            # Use sympify to parse the expression
            expr = sp.sympify(expr_str, locals=namespace, evaluate=False)
            return expr
        except Exception as e:
            raise DQXError(f"Failed to parse metric expression '{expr_str}': {e}") from e

    def _build_namespace(self) -> dict[str, Any]:
        """Build the namespace for sympify with metric functions."""
        namespace: dict[str, Any] = {}

        # Add math functions
        namespace.update(MATH_FUNCTIONS)

        # Add metric functions that return symbols
        for name in METRIC_FUNCTIONS:
            namespace[name] = self._create_metric_wrapper(name)

        # Add extended metric functions
        for name in EXTENDED_METRIC_FUNCTIONS:
            namespace[name] = self._create_extended_metric_wrapper(name)

        return namespace

    def _create_metric_wrapper(self, func_name: str) -> Any:
        """Create a wrapper function for a metric that returns a symbol."""

        def wrapper(*args: Any, **kwargs: Any) -> sp.Symbol:
            return self._call_metric(func_name, args, kwargs)

        return wrapper

    def _create_extended_metric_wrapper(self, func_name: str) -> Any:
        """Create a wrapper function for an extended metric."""

        def wrapper(base_metric: sp.Symbol, **kwargs: Any) -> sp.Symbol:
            return self._call_extended_metric(func_name, base_metric, kwargs)

        return wrapper

    def _call_metric(self, func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> sp.Symbol:
        """Call a metric function on the provider and return the symbol."""

        # Convert Symbol objects to strings (sympify converts unquoted names to Symbols)
        # Convert sympy Integer/Float to Python int/float
        def convert_arg(a: Any) -> Any:
            if isinstance(a, sp.Symbol):
                return str(a)
            elif isinstance(a, sp.Integer):  # pragma: no cover
                return int(a)
            elif isinstance(a, sp.Float):
                return float(a)
            return a

        args = tuple(convert_arg(a) for a in args)
        kwargs = {k: convert_arg(v) for k, v in kwargs.items()}

        # Extract known parameters
        dataset = kwargs.pop("dataset", None)
        lag = int(kwargs.pop("lag", 0))  # Ensure lag is always a Python int

        # Remaining kwargs become parameters
        parameters = kwargs if kwargs else None

        # Get the method from provider
        method = getattr(self._provider, func_name)

        # Handle different metric signatures
        if func_name == "num_rows":
            return method(lag=lag, dataset=dataset, parameters=parameters)
        elif func_name == "duplicate_count":  # pragma: no cover
            # First arg is columns list - pop from kwargs so it doesn't end up in parameters
            columns = list(args[0]) if args else kwargs.pop("columns", [])
            # Convert column names to strings
            columns = [str(c) if isinstance(c, sp.Symbol) else c for c in columns]
            # Update parameters after popping columns
            parameters = kwargs if kwargs else None
            return method(columns, lag=lag, dataset=dataset, parameters=parameters)
        elif func_name == "count_values":  # pragma: no cover
            # First arg is column, second is value(s) - pop from kwargs
            column = args[0] if args else kwargs.pop("column", None)
            values = args[1] if len(args) > 1 else kwargs.pop("values", None)
            # Update parameters after popping
            parameters = kwargs if kwargs else None
            return method(column, values, lag=lag, dataset=dataset, parameters=parameters)
        elif func_name == "custom_sql":
            # First arg is SQL expression - pop from kwargs
            # Security note: SQL is passed to provider without sanitization.
            # Only load configurations from trusted sources.
            sql_expr = args[0] if args else kwargs.pop("sql_expression", None)
            # Update parameters after popping
            parameters = kwargs if kwargs else None
            return method(sql_expr, lag=lag, dataset=dataset, parameters=parameters)
        else:
            # Standard column-based metrics - pop column from kwargs
            column = args[0] if args else kwargs.pop("column", None)
            # Update parameters after popping
            parameters = kwargs if kwargs else None
            return method(column, lag=lag, dataset=dataset, parameters=parameters)

    def _call_extended_metric(self, func_name: str, base_metric: sp.Symbol, kwargs: dict[str, Any]) -> sp.Symbol:
        """Call an extended metric function on the provider."""
        ext = self._provider.ext

        # Convert sympy types to Python types
        def to_int(v: Any, default: int = 0) -> int:
            if v is None:
                return default
            if isinstance(v, (sp.Integer, sp.Float)):  # pragma: no cover
                return int(v)
            return int(v)  # pragma: no cover

        def to_str(v: Any) -> str | None:
            if v is None:
                return None
            if isinstance(v, sp.Symbol):  # pragma: no cover
                return str(v)
            return str(v) if v else None  # pragma: no cover

        if func_name == "day_over_day":
            lag = to_int(kwargs.get("lag"), 0)
            dataset = to_str(kwargs.get("dataset"))
            return ext.day_over_day(base_metric, lag=lag, dataset=dataset)
        elif func_name == "week_over_week":
            lag = to_int(kwargs.get("lag"), 0)
            dataset = to_str(kwargs.get("dataset"))
            return ext.week_over_week(base_metric, lag=lag, dataset=dataset)
        elif func_name == "stddev":
            offset = to_int(kwargs.get("offset"), 0)
            n = to_int(kwargs.get("n"), 7)
            dataset = to_str(kwargs.get("dataset"))
            return ext.stddev(base_metric, offset=offset, n=n, dataset=dataset)
        else:  # pragma: no cover
            raise DQXError(f"Unknown extended metric: {func_name}")


# =============================================================================
# Validator/Expect Parser
# =============================================================================

# Regex patterns for expect syntax
# Note: Order matters! Put simple patterns (no groups) first to avoid false matches
EXPECT_PATTERNS = [
    (r"^positive$", "positive"),
    (r"^negative$", "negative"),
    (r"^collect$", "noop"),
    (r"^>=\s*(.+)$", "geq"),  # >= N (must be before > N)
    (r"^<=\s*(.+)$", "leq"),  # <= N (must be before < N)
    (r"^>\s*(.+)$", "gt"),  # > N
    (r"^<\s*(.+)$", "lt"),  # < N
    (r"^=\s*(.+)$", "eq"),  # = N
    (r"^between\s+(.+)\s+and\s+(.+)$", "between"),  # between A and B
]


def _make_gt_validator(value: float, tol: float) -> SymbolicValidator:
    return SymbolicValidator(f"> {value}", lambda x: functions.is_gt(x, value, tol))


def _make_geq_validator(value: float, tol: float) -> SymbolicValidator:
    return SymbolicValidator(f"≥ {value}", lambda x: functions.is_geq(x, value, tol))


def _make_lt_validator(value: float, tol: float) -> SymbolicValidator:
    return SymbolicValidator(f"< {value}", lambda x: functions.is_lt(x, value, tol))


def _make_leq_validator(value: float, tol: float) -> SymbolicValidator:
    return SymbolicValidator(f"≤ {value}", lambda x: functions.is_leq(x, value, tol))


def _make_eq_validator(value: float, tol: float) -> SymbolicValidator:
    return SymbolicValidator(f"= {value}", lambda x: functions.is_eq(x, value, tol))


def _make_between_validator(low: float, high: float, tol: float) -> SymbolicValidator:
    return SymbolicValidator(f"∈ [{low}, {high}]", lambda x: functions.is_between(x, low, high, tol))


def _make_positive_validator(tol: float) -> SymbolicValidator:
    return SymbolicValidator("positive", lambda x: functions.is_positive(x, tol))


def _make_negative_validator(tol: float) -> SymbolicValidator:
    return SymbolicValidator("negative", lambda x: functions.is_negative(x, tol))


def parse_expect(expect_str: str, tolerance: float | None = None) -> SymbolicValidator:
    """Parse an expect string into a SymbolicValidator.

    Args:
        expect_str: Validation condition like "> 0", "between 0.8 and 1.2"
        tolerance: Optional tolerance for numeric comparisons

    Returns:
        SymbolicValidator with name and validation function

    Raises:
        DQXError: If expect string format is invalid
    """
    expect_str = expect_str.strip()
    tol = tolerance if tolerance is not None else functions.EPSILON

    for pattern, op_type in EXPECT_PATTERNS:
        match = re.match(pattern, expect_str, re.IGNORECASE)
        if match:
            try:
                if op_type == "gt":
                    value = float(match.group(1))
                    return _make_gt_validator(value, tol)
                elif op_type == "geq":
                    value = float(match.group(1))
                    return _make_geq_validator(value, tol)
                elif op_type == "lt":
                    value = float(match.group(1))
                    return _make_lt_validator(value, tol)
                elif op_type == "leq":
                    value = float(match.group(1))
                    return _make_leq_validator(value, tol)
                elif op_type == "eq":
                    value = float(match.group(1))
                    return _make_eq_validator(value, tol)
                elif op_type == "between":
                    low = float(match.group(1))
                    high = float(match.group(2))
                    return _make_between_validator(low, high, tol)
                elif op_type == "positive":
                    return _make_positive_validator(tol)
                elif op_type == "negative":
                    return _make_negative_validator(tol)
                elif op_type == "noop":
                    return SymbolicValidator("noop", functions.noop)
            except ValueError as e:
                raise DQXError(f"Invalid numeric value in expect '{expect_str}': {e}") from e

    raise DQXError(f"Invalid expect format: '{expect_str}'")


# =============================================================================
# Profile Parser
# =============================================================================


def parse_profile(profile_dict: dict[str, Any]) -> Profile:
    """Parse a profile dictionary into a Profile object.

    Args:
        profile_dict: Dictionary with profile configuration

    Returns:
        Profile instance

    Raises:
        DQXError: If profile format is invalid
    """
    name = profile_dict.get("name")
    profile_type = profile_dict.get("type")

    if not name:
        raise DQXError("Profile must have a 'name' field")
    if not profile_type:
        raise DQXError(f"Profile '{name}' must have a 'type' field")

    if profile_type == "holiday":
        return _parse_holiday_profile(profile_dict)
    else:
        raise DQXError(f"Unknown profile type: '{profile_type}'")


def _parse_holiday_profile(profile_dict: dict[str, Any]) -> HolidayProfile:
    """Parse a holiday profile dictionary."""
    name = profile_dict["name"]

    start_date_str = profile_dict.get("start_date")
    end_date_str = profile_dict.get("end_date")

    if not start_date_str or not end_date_str:
        raise DQXError(f"Holiday profile '{name}' requires 'start_date' and 'end_date'")

    try:
        start_date = date.fromisoformat(start_date_str)
        end_date = date.fromisoformat(end_date_str)
    except ValueError as e:
        raise DQXError(f"Invalid date format in profile '{name}': {e}") from e

    rules = []
    for rule_dict in profile_dict.get("rules", []):
        rules.append(_parse_rule(rule_dict))

    return HolidayProfile(name=name, start_date=start_date, end_date=end_date, rules=rules)


def _parse_rule(rule_dict: dict[str, Any]) -> Rule:
    """Parse a rule dictionary into a Rule object."""
    # Determine selector type
    check_name = rule_dict.get("check")
    assertion_name = rule_dict.get("assertion")
    tag_name = rule_dict.get("tag")

    selector: AssertionSelector | TagSelector
    if check_name:
        selector = AssertionSelector(check=check_name, assertion=assertion_name)
    elif tag_name:
        selector = TagSelector(tag=tag_name)
    else:
        raise DQXError("Rule must have either 'check' or 'tag' field")

    # Parse action and modifiers
    disabled = rule_dict.get("action") == "disable"
    metric_multiplier = rule_dict.get("metric_multiplier", 1.0)
    severity = rule_dict.get("severity")

    if severity is not None and severity not in ("P0", "P1", "P2", "P3"):
        raise DQXError(f"Invalid severity '{severity}' in rule")

    return Rule(
        selector=selector,
        disabled=disabled,
        metric_multiplier=metric_multiplier,
        severity=severity,
    )


# =============================================================================
# Configuration Loader
# =============================================================================


@dataclass
class CheckConfig:
    """Configuration for a single check."""

    name: str
    datasets: list[str] = field(default_factory=list)
    assertions: list["AssertionConfig"] = field(default_factory=list)


@dataclass
class AssertionConfig:
    """Configuration for a single assertion."""

    name: str
    metric: str
    expect: str
    severity: SeverityLevel = "P1"
    tolerance: float | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SuiteConfig:
    """Configuration for a verification suite."""

    name: str
    data_av_threshold: float = 0.9
    checks: list[CheckConfig] = field(default_factory=list)
    profiles: list[Profile] = field(default_factory=list)


def load_config(path: str | Path, *, validate_schema: bool = True) -> SuiteConfig:
    """Load and parse a suite configuration file.

    Args:
        path: Path to YAML or JSON configuration file
        validate_schema: If True, validate against JSON schema before parsing

    Returns:
        Parsed SuiteConfig object

    Raises:
        DQXError: If file cannot be loaded, parsed, or fails schema validation
    """
    path = Path(path)

    if not path.exists():
        raise DQXError(f"Configuration file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise DQXError(f"Failed to parse configuration file: {e}") from e

    if validate_schema:
        errors = validate_dict_schema(config_dict)
        if errors:
            raise DQXError("Schema validation failed:\n  " + "\n  ".join(errors))

    return parse_config(config_dict)


def load_config_string(content: str, *, validate_schema: bool = True) -> SuiteConfig:
    """Load and parse a suite configuration from a string.

    Args:
        content: YAML or JSON configuration string
        validate_schema: If True, validate against JSON schema before parsing

    Returns:
        Parsed SuiteConfig object

    Raises:
        DQXError: If content cannot be parsed or fails schema validation
    """
    try:
        config_dict = yaml.safe_load(content)
    except Exception as e:
        raise DQXError(f"Failed to parse configuration: {e}") from e

    if validate_schema:
        errors = validate_dict_schema(config_dict)
        if errors:
            raise DQXError("Schema validation failed:\n  " + "\n  ".join(errors))

    return parse_config(config_dict)


def parse_config(config_dict: dict[str, Any]) -> SuiteConfig:
    """Parse a configuration dictionary into a SuiteConfig.

    Args:
        config_dict: Dictionary with suite configuration

    Returns:
        Parsed SuiteConfig object

    Raises:
        DQXError: If configuration is invalid
    """
    if not config_dict:
        raise DQXError("Configuration cannot be empty")

    name = config_dict.get("name")
    if not name:
        raise DQXError("Configuration must have a 'name' field")

    data_av_threshold = config_dict.get("data_av_threshold", 0.9)

    # Parse checks
    checks_list = config_dict.get("checks", [])
    if not checks_list:
        raise DQXError("Configuration must have at least one check")

    checks = []
    for check_dict in checks_list:
        checks.append(_parse_check(check_dict))

    # Parse profiles
    profiles = []
    for profile_dict in config_dict.get("profiles", []):
        profiles.append(parse_profile(profile_dict))

    return SuiteConfig(
        name=name,
        data_av_threshold=data_av_threshold,
        checks=checks,
        profiles=profiles,
    )


def _parse_check(check_dict: dict[str, Any]) -> CheckConfig:
    """Parse a check dictionary into a CheckConfig."""
    name = check_dict.get("name")
    if not name:
        raise DQXError("Check must have a 'name' field")

    datasets = check_dict.get("datasets", [])

    assertions_list = check_dict.get("assertions", [])
    if not assertions_list:
        raise DQXError(f"Check '{name}' must have at least one assertion")

    assertions = []
    for assertion_dict in assertions_list:
        assertions.append(_parse_assertion(assertion_dict, name))

    return CheckConfig(name=name, datasets=datasets, assertions=assertions)


def _parse_assertion(assertion_dict: dict[str, Any], check_name: str) -> AssertionConfig:
    """Parse an assertion dictionary into an AssertionConfig."""
    name = assertion_dict.get("name")
    if not name:
        raise DQXError(f"Assertion in check '{check_name}' must have a 'name' field")

    metric = assertion_dict.get("metric")
    if not metric:
        raise DQXError(f"Assertion '{name}' must have a 'metric' field")

    expect = assertion_dict.get("expect")
    if not expect:
        raise DQXError(f"Assertion '{name}' must have an 'expect' field")

    severity = assertion_dict.get("severity", "P1")
    tolerance = assertion_dict.get("tolerance")
    tags = assertion_dict.get("tags", [])

    return AssertionConfig(
        name=name,
        metric=metric,
        expect=expect,
        severity=severity,
        tolerance=tolerance,
        tags=tags,
    )


# =============================================================================
# Validation
# =============================================================================


def validate_config(path: str | Path) -> list[str]:
    """Validate a configuration file with full semantic checks.

    This function fully loads and parses the configuration to perform
    semantic validation (duplicate names, expect format, etc.).
    For lightweight schema-only validation, use validate_config_schema().

    Args:
        path: Path to configuration file

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    try:
        config = load_config(path)
    except DQXError as e:
        errors.append(str(e))
        return errors

    # Validate checks
    check_names = set()
    for check in config.checks:
        if check.name in check_names:
            errors.append(f"Duplicate check name: '{check.name}'")
        check_names.add(check.name)

        # Validate assertions
        assertion_names = set()
        for assertion in check.assertions:
            if assertion.name in assertion_names:
                errors.append(f"Duplicate assertion name in check '{check.name}': '{assertion.name}'")
            assertion_names.add(assertion.name)

            # Validate severity
            if assertion.severity not in ("P0", "P1", "P2", "P3"):  # pragma: no cover
                errors.append(f"Invalid severity '{assertion.severity}' in assertion '{assertion.name}'")

            # Validate expect format (basic check)
            try:
                parse_expect(assertion.expect, assertion.tolerance)
            except DQXError as e:  # pragma: no cover
                errors.append(f"Invalid expect in assertion '{assertion.name}': {e}")

    # Validate profile names
    profile_names = set()
    for profile in config.profiles:
        if profile.name in profile_names:
            errors.append(f"Duplicate profile name: '{profile.name}'")
        profile_names.add(profile.name)

    return errors


# =============================================================================
# Serialization
# =============================================================================


def suite_config_to_dict(config: SuiteConfig) -> dict[str, Any]:
    """Convert a SuiteConfig to a dictionary for YAML serialization.

    Args:
        config: SuiteConfig to convert

    Returns:
        Dictionary suitable for YAML serialization
    """
    result: dict[str, Any] = {
        "name": config.name,
    }

    # Always include data_av_threshold for round-trip safety
    result["data_av_threshold"] = config.data_av_threshold

    # Serialize checks
    result["checks"] = [_check_to_dict(check) for check in config.checks]

    # Serialize profiles
    if config.profiles:
        result["profiles"] = [_profile_to_dict(profile) for profile in config.profiles]

    return result


def _check_to_dict(check: CheckConfig) -> dict[str, Any]:
    """Convert a CheckConfig to a dictionary."""
    result: dict[str, Any] = {"name": check.name}

    if check.datasets:
        result["datasets"] = check.datasets

    result["assertions"] = [_assertion_to_dict(a) for a in check.assertions]

    return result


def _assertion_to_dict(assertion: AssertionConfig) -> dict[str, Any]:
    """Convert an AssertionConfig to a dictionary."""
    result: dict[str, Any] = {
        "name": assertion.name,
        "metric": assertion.metric,
        "expect": assertion.expect,
    }

    if assertion.severity != "P1":
        result["severity"] = assertion.severity

    if assertion.tolerance is not None:
        result["tolerance"] = assertion.tolerance

    if assertion.tags:
        result["tags"] = assertion.tags

    return result


def _profile_to_dict(profile: Profile) -> dict[str, Any]:
    """Convert a Profile to a dictionary."""
    if isinstance(profile, HolidayProfile):
        result: dict[str, Any] = {
            "name": profile.name,
            "type": "holiday",
            "start_date": profile.start_date.isoformat(),
            "end_date": profile.end_date.isoformat(),
        }

        if profile.rules:
            result["rules"] = [_rule_to_dict(rule) for rule in profile.rules]

        return result
    else:  # pragma: no cover
        raise DQXError(f"Unsupported profile type for serialization: {type(profile)}")


def _rule_to_dict(rule: Rule) -> dict[str, Any]:
    """Convert a Rule to a dictionary."""
    result: dict[str, Any] = {}

    # Serialize selector
    if isinstance(rule.selector, AssertionSelector):
        result["check"] = rule.selector.check
        if rule.selector.assertion:
            result["assertion"] = rule.selector.assertion
    elif isinstance(rule.selector, TagSelector):
        result["tag"] = rule.selector.tag
    else:  # pragma: no cover
        raise DQXError(f"Unsupported selector type for serialization: {type(rule.selector)}")

    # Serialize action/modifiers
    if rule.disabled:
        result["action"] = "disable"

    if rule.severity is not None:
        result["severity"] = rule.severity

    # Include metric_multiplier if non-default OR if it's the only modifier
    # (schema requires at least one of: action, metric_multiplier, severity)
    has_other_modifier = rule.disabled or rule.severity is not None
    if rule.metric_multiplier != 1.0 or not has_other_modifier:
        result["metric_multiplier"] = rule.metric_multiplier

    return result


def suite_config_to_yaml(config: SuiteConfig) -> str:
    """Serialize a SuiteConfig to YAML string.

    Args:
        config: SuiteConfig to serialize

    Returns:
        YAML string representation
    """
    config_dict = suite_config_to_dict(config)
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False, allow_unicode=True)
