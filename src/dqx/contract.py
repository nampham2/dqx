"""ODCS Contract parsing for DQX.

This module provides :class:`Contract`, a dataclass that represents one schema
object from an ODCS (Open Data Contract Standard) YAML document.

Usage::

    from pathlib import Path
    from dqx.contract import Contract

    contracts = Contract.from_odcs(Path("orders.odcs.yaml"))
    # list[Contract] — one per schema[] entry in the ODCS file

The module validates incoming YAML documents against the vendored ODCS JSON
Schema (v3.1.0) before any DQX-specific processing.  Structural errors surface
as :class:`ContractValidationError` before a single quality rule is read.
"""

from __future__ import annotations

import importlib.resources
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

import sympy as sp
import yaml
from jsonschema import Draft201909Validator
from jsonschema import ValidationError as _JsonSchemaValidationError

from dqx.api import Context
from dqx.common import SeverityLevel
from dqx.provider import MetricProvider

if TYPE_CHECKING:
    from dqx.api import AssertionReady, DecoratedCheck


# ---------------------------------------------------------------------------
# Public error types
# ---------------------------------------------------------------------------


class ContractValidationError(Exception):
    """Raised when an ODCS document fails JSON Schema or DQX structural validation.

    This error is raised during :meth:`Contract.from_odcs` before any check
    executes.  It indicates that the contract document itself is invalid and
    cannot be processed.
    """


class SchemaValidationError(Exception):
    """Raised at runtime when actual data does not match the declared schema.

    This error is raised during ``to_checks()`` execution against a live
    datasource — for example, when a column declared as ``logicalType:
    integer`` contains ``pa.string()`` values, or when a column with
    ``required: true`` contains null values.
    """


class ContractWarning(UserWarning):
    """Emitted for valid but non-executable contract constructs.

    Processing continues after each warning; the offending rule or SLA entry
    is skipped.  Examples:

    - An ``slaProperties`` latency entry whose timestamp column cannot be
      resolved (neither ``element:`` nor a partitioned property found).
    - A quality rule with ``type: sql`` (not executable in DQX).
    - A quality rule with ``type: custom`` and an unsupported engine.
    """


# ---------------------------------------------------------------------------
# JSON Schema — loaded once at module import
# ---------------------------------------------------------------------------


def _load_odcs_schema() -> dict[str, Any]:
    """Load the vendored ODCS JSON Schema from the package's _schemas directory.

    Returns:
        Parsed JSON Schema dict.
    """
    schema_text = (
        importlib.resources.files("dqx._schemas").joinpath("odcs-json-schema-v3.1.0.json").read_text(encoding="utf-8")
    )
    return json.loads(schema_text)  # type: ignore[no-any-return]


_ODCS_SCHEMA: dict[str, Any] = _load_odcs_schema()
_ODCS_VALIDATOR = Draft201909Validator(_ODCS_SCHEMA)

# ---------------------------------------------------------------------------
# SLA unit conversion
# ---------------------------------------------------------------------------

_HOURS_UNITS: frozenset[str] = frozenset({"h", "hr", "hour", "hours"})
_DAYS_UNITS: frozenset[str] = frozenset({"d", "day", "days"})
_YEARS_UNITS: frozenset[str] = frozenset({"y", "yr", "year", "years"})
_LATENCY_SYNONYMS: frozenset[str] = frozenset({"latency", "ly"})

# ODCS operator field names — used by both library and custom rule execution.
_ODCS_OPERATOR_FIELDS: frozenset[str] = frozenset(
    {
        "mustBe",
        "mustNotBe",
        "mustBeGreaterThan",
        "mustBeGreaterOrEqualTo",
        "mustBeLessThan",
        "mustBeLessOrEqualTo",
        "mustBeBetween",
        "mustNotBeBetween",
    }
)

# Accepted column_type values for min_length / max_length.
_LENGTH_COLUMN_TYPES: frozenset[str] = frozenset({"string", "list", "map"})


def _unit_to_hours(value: float, unit: str) -> float:
    """Convert a (value, unit) pair to hours.

    Args:
        value: Numeric SLA value.
        unit: Time unit string (h/hr/hour/hours, d/day/days, y/yr/year/years).

    Returns:
        Equivalent number of hours as a float.

    Raises:
        ContractValidationError: If the unit is not recognised.
    """
    if unit in _HOURS_UNITS:
        return float(value)
    if unit in _DAYS_UNITS:
        return float(value) * 24.0
    if unit in _YEARS_UNITS:
        return float(value) * 8760.0
    raise ContractValidationError(f"SLA latency: unrecognised unit '{unit}'")


# ---------------------------------------------------------------------------
# Severity and operator translation helpers
# ---------------------------------------------------------------------------


def _severity_from_odcs(severity: str | None) -> SeverityLevel:
    """Map an ODCS severity string to a DQX SeverityLevel.

    Args:
        severity: ODCS ``severity`` field value, or ``None`` when absent.

    Returns:
        ``"P0"`` for ``"error"``; ``"P1"`` for everything else (including
        ``None`` and unrecognised strings).
    """
    if severity == "error":
        return "P0"
    return "P1"


def _apply_odcs_operators(ready: AssertionReady, rule: dict[str, Any]) -> None:
    """Apply exactly one ODCS operator from *rule* to an :class:`AssertionReady`.

    Reads the first recognised ODCS operator field (``mustBe``,
    ``mustNotBe``, ``mustBeGreaterThan``, ``mustBeGreaterOrEqualTo``,
    ``mustBeLessThan``, ``mustBeLessOrEqualTo``, ``mustBeBetween``,
    ``mustNotBeBetween``) and calls the corresponding
    :class:`~dqx.api.AssertionReady` method.  When no operator is present
    the assertion is recorded as a no-op (metric observed, never fails).

    Args:
        ready: A configured :class:`~dqx.api.AssertionReady` instance.
        rule: Raw ODCS quality rule dict.

    Raises:
        ContractValidationError: If ``mustBeBetween`` or ``mustNotBeBetween``
            is not a two-element list.
    """
    if "mustBe" in rule:
        ready.is_eq(float(rule["mustBe"]))
    elif "mustNotBe" in rule:
        ready.is_neq(float(rule["mustNotBe"]))
    elif "mustBeGreaterThan" in rule:
        ready.is_gt(float(rule["mustBeGreaterThan"]))
    elif "mustBeGreaterOrEqualTo" in rule:
        ready.is_geq(float(rule["mustBeGreaterOrEqualTo"]))
    elif "mustBeLessThan" in rule:
        ready.is_lt(float(rule["mustBeLessThan"]))
    elif "mustBeLessOrEqualTo" in rule:
        ready.is_leq(float(rule["mustBeLessOrEqualTo"]))
    elif "mustBeBetween" in rule:
        bounds = rule["mustBeBetween"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ContractValidationError(f"mustBeBetween must be a list of [lower, upper], got: {bounds!r}")
        ready.is_between(float(bounds[0]), float(bounds[1]))
    elif "mustNotBeBetween" in rule:
        bounds = rule["mustNotBeBetween"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ContractValidationError(f"mustNotBeBetween must be a list of [lower, upper], got: {bounds!r}")
        ready.is_not_between(float(bounds[0]), float(bounds[1]))
    else:
        ready.noop()


def _require_column(column: str | None, rule_name: str, check_type: str) -> str:
    """Return *column* or raise :class:`ContractValidationError` when it is ``None``.

    Args:
        column: Column name from the rule, or ``None`` when absent.
        rule_name: Human-readable rule name (used in the error message).
        check_type: Check type string (used in the error message).

    Returns:
        The validated column name.

    Raises:
        ContractValidationError: If *column* is ``None``.
    """
    if column is None:
        raise ContractValidationError(
            f"Quality rule '{rule_name}': check '{check_type}' requires a 'column' field but none was provided"
        )
    return column


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlaLatency:
    """Resolved SLA latency constraint.

    Attributes:
        max_age_hours: Maximum allowed data age expressed in hours.
        timestamp_column: Name of the column used to measure data freshness.
    """

    max_age_hours: float
    timestamp_column: str


@dataclass(frozen=True)
class ContractProperty:
    """A single property (column) declared in an ODCS schema object.

    Attributes:
        name: Property name as declared in the ODCS ``name`` field.
        logical_type: ODCS ``logicalType`` value, or ``None`` if absent.
        physical_type: ODCS ``physicalType`` value, or ``None`` if absent.
        required: ``True`` if the column must not contain nulls.  Defaults to
            ``False`` (nullable) when the ODCS ``required`` field is absent.
        partitioned: ``True`` if the column is a partition key.
        partition_key_position: Position in the composite partition key; ``-1``
            when not partitioned.
        logical_type_options: Free-form ``logicalTypeOptions`` dict, or ``None``.
        quality: Raw quality rules attached to this property (excluding ``type:
            text`` rules, which go to :attr:`text_rules`).
    """

    name: str
    logical_type: str | None
    physical_type: str | None
    required: bool
    partitioned: bool
    partition_key_position: int
    logical_type_options: dict[str, Any] | None
    quality: tuple[dict[str, Any], ...]

    @property
    def text_rules(self) -> tuple[dict[str, Any], ...]:
        """Convenience accessor — text rules are not stored on ContractProperty.

        Returns:
            Always an empty tuple; text-rule separation happens at the schema
            (table) level, not the property level.
        """
        return ()  # pragma: no cover


@dataclass(frozen=True)
class ContractSchema:
    """A single schema object from an ODCS ``schema[]`` array.

    Attributes:
        name: Logical name of the schema object (ODCS ``name``).
        physical_name: Physical table name (ODCS ``physicalName``), or ``None``.
        properties: Tuple of parsed :class:`ContractProperty` instances.
        quality: Tuple of raw quality rule dicts (excludes ``type: text``).
        text_rules: Tuple of raw ``type: text`` quality rule dicts (documentation
            only; no assertions generated from these).
    """

    name: str
    physical_name: str | None
    properties: tuple[ContractProperty, ...]
    quality: tuple[dict[str, Any], ...]
    text_rules: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Contract:
    """A DQX contract derived from one schema object in an ODCS document.

    One :class:`Contract` instance corresponds to one entry in the ODCS
    ``schema[]`` list.  Use :meth:`from_odcs` to construct contracts from a
    file — it always returns a ``list[Contract]``, even for single-schema files.

    Attributes:
        name: Top-level ODCS contract ``name``, or ``None`` if absent.
        version: ODCS ``version`` field.
        api_version: ODCS ``apiVersion`` field.
        contract_id: ODCS ``id`` field.
        status: ODCS ``status`` field.
        dataset: Physical or logical dataset name for this schema object.
            Prefers ``physicalName`` over ``name`` when both are present.
        schema_def: Parsed schema definition for this contract.
        sla_latency: Resolved latency constraint, or ``None`` when no latency
            ``slaProperties`` entry can be resolved.
        sla_metadata: Immutable mapping of non-latency SLA properties keyed by
            property name.  Values are the full SLA entry dicts (excluding the
            ``property`` key itself).  Use ``dict(contract.sla_metadata)`` if
            a mutable copy is needed.
    """

    name: str | None
    version: str
    api_version: str
    contract_id: str
    status: str
    dataset: str
    schema_def: ContractSchema
    sla_latency: SlaLatency | None
    sla_metadata: Mapping[str, Any]

    @classmethod
    def from_odcs(cls, path: Path) -> list[Contract]:
        """Parse an ODCS YAML file and return one :class:`Contract` per schema object.

        The method performs these steps in order:

        1. Read the YAML file (raises :class:`FileNotFoundError` if absent,
           :class:`yaml.YAMLError` if malformed).
        2. Validate the parsed dict against the vendored ODCS JSON Schema
           v3.1.0 (raises :class:`ContractValidationError` on failure).
        3. Parse each entry in ``schema[]`` into a :class:`ContractSchema`
           and a :class:`Contract`.
        4. Parse ``slaProperties``, resolving latency entries and storing all
           others in ``sla_metadata``.

        Args:
            path: Path to the ODCS YAML file.

        Returns:
            List of :class:`Contract` instances — one per ``schema[]`` entry.
            Returns an empty list if the document has no ``schema`` key or
            an empty ``schema`` array.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
            ContractValidationError: If the document fails JSON Schema
                validation (missing required fields, wrong enum values, etc.).
        """
        raw = _read_yaml(path)
        _validate_against_schema(raw)

        # Parse SLA properties — shared across all schema objects in this file
        sla_latency, sla_metadata = _parse_sla_properties(
            raw.get("slaProperties", []),
            schema_objects=raw.get("schema", []),
        )

        contracts: list[Contract] = []
        for schema_obj in raw.get("schema") or []:
            schema_def = _parse_schema_object(schema_obj)
            dataset = schema_obj.get("physicalName") or schema_obj["name"]
            contracts.append(
                cls(
                    name=raw.get("name"),
                    version=str(raw["version"]),
                    api_version=str(raw["apiVersion"]),
                    contract_id=str(raw["id"]),
                    status=str(raw["status"]),
                    dataset=dataset,
                    schema_def=schema_def,
                    sla_latency=sla_latency,
                    sla_metadata=MappingProxyType(sla_metadata),
                )
            )
        return contracts

    def to_checks(self) -> list[DecoratedCheck]:
        """Translate this contract into a list containing one :class:`DecoratedCheck`.

        All quality rules declared in this contract's schema object — both
        table-level and property-level — are consolidated into a single
        ``DecoratedCheck`` node named ``"Contract: {name or dataset}"``.

        The returned list can be composed freely with hand-coded checks::

            suite = VerificationSuite(
                checks=contract.to_checks() + [custom_check],
                db=db,
                name="my-suite",
            )

        Returns:
            A list with exactly one :class:`~dqx.api.DecoratedCheck`.

        Raises:
            NotImplementedError: Immediately when ``sla_latency`` is set —
                the SLA freshness check requires ``MetricProvider.freshness()``
                which is not yet implemented.
            NotImplementedError: When any quality rule requires a
                ``MetricProvider`` method that is not yet implemented
                (e.g., ``check: avg_length``, ``check: percentile``).
            ContractValidationError: When ``mustBeBetween`` / ``mustNotBeBetween``
                is not a two-element list, an unknown DQX check type is used, a
                column-level metric is applied at the table level without a column
                name, or an invalid ``column_type`` is provided for
                ``min_length`` / ``max_length``.
        """
        if self.sla_latency is not None:
            raise NotImplementedError(
                "SLA freshness check requires MetricProvider.freshness() which is not yet implemented"
            )

        from dqx.api import check as _dqx_check

        check_name = f"Contract: {self.name or self.dataset}"
        schema = self.schema_def
        dataset = self.dataset

        def _run_checks(mp: MetricProvider, ctx: Context) -> None:
            for rule in schema.quality:
                _dispatch_rule(rule, None, mp, ctx)
            for prop in schema.properties:
                for rule in prop.quality:
                    _dispatch_rule(rule, prop.name, mp, ctx)

        _run_checks.__name__ = check_name
        decorated: DecoratedCheck = _dqx_check(name=check_name, datasets=[dataset])(_run_checks)
        return [decorated]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    if not path.exists():
        raise FileNotFoundError(f"ODCS contract file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        result = yaml.safe_load(fh)
    return result if isinstance(result, dict) else {}


def _validate_against_schema(raw: dict[str, Any]) -> None:
    """Validate a parsed ODCS document against the vendored JSON Schema.

    Args:
        raw: Parsed YAML dict to validate.

    Raises:
        ContractValidationError: If the document fails validation.
    """
    try:
        _ODCS_VALIDATOR.validate(raw)
    except _JsonSchemaValidationError as exc:
        raise ContractValidationError(str(exc.message)) from exc


def _parse_schema_object(obj: dict[str, Any]) -> ContractSchema:
    """Parse a single ODCS ``schema[]`` object into a :class:`ContractSchema`.

    Args:
        obj: Raw dict from the ODCS ``schema[]`` array.

    Returns:
        Parsed :class:`ContractSchema` instance.
    """
    properties = tuple(_parse_property(p) for p in (obj.get("properties") or []))

    quality_rules: list[dict[str, Any]] = []
    text_rules: list[dict[str, Any]] = []
    for rule in obj.get("quality") or []:
        if rule.get("type") == "text":
            text_rules.append(rule)
        else:
            quality_rules.append(rule)

    return ContractSchema(
        name=obj["name"],
        physical_name=obj.get("physicalName"),
        properties=properties,
        quality=tuple(quality_rules),
        text_rules=tuple(text_rules),
    )


def _parse_property(prop: dict[str, Any]) -> ContractProperty:
    """Parse a single ODCS property dict into a :class:`ContractProperty`.

    Args:
        prop: Raw property dict from the ODCS ``schema[].properties[]`` array.

    Returns:
        Parsed :class:`ContractProperty` instance.
    """
    quality_rules: list[dict[str, Any]] = []
    for rule in prop.get("quality") or []:
        if rule.get("type") != "text":
            quality_rules.append(rule)

    return ContractProperty(
        name=prop["name"],
        logical_type=prop.get("logicalType"),
        physical_type=prop.get("physicalType"),
        required=bool(prop.get("required", False)),
        partitioned=bool(prop.get("partitioned", False)),
        partition_key_position=int(prop.get("partitionKeyPosition", -1)),
        logical_type_options=prop.get("logicalTypeOptions"),
        quality=tuple(quality_rules),
    )


def _resolve_timestamp_column(
    sla_entry: dict[str, Any],
    schema_objects: list[dict[str, Any]],
) -> str | None:
    """Resolve the timestamp column name for a latency SLA entry.

    Resolution order:
    1. ``element:`` field in ``"table.column"`` notation — use the part after
       the last ``.``.
    2. Search ``schema[].properties[]`` for the first property with both
       ``partitioned: true`` and ``partitionKeyPosition: 1``.

    Args:
        sla_entry: A single ``slaProperties`` dict with ``property: latency``.
        schema_objects: The full ``schema[]`` list from the ODCS document.

    Returns:
        Column name string, or ``None`` if neither resolution method succeeds.
    """
    element: str | None = sla_entry.get("element")
    if element:
        return element.rsplit(".", 1)[-1]

    for schema_obj in schema_objects:
        for prop in schema_obj.get("properties") or []:
            if prop.get("partitioned") and prop.get("partitionKeyPosition") == 1:
                return str(prop["name"])

    return None


def _parse_sla_properties(
    sla_list: list[dict[str, Any]],
    schema_objects: list[dict[str, Any]],
) -> tuple[SlaLatency | None, dict[str, Any]]:
    """Parse the ``slaProperties`` list into resolved latency + metadata dict.

    Latency entries (``property: latency`` or ``property: ly``) are resolved
    to a :class:`SlaLatency` instance.  If the timestamp column cannot be
    determined, a :class:`ContractWarning` is emitted and ``sla_latency`` is
    set to ``None``.

    All other SLA properties are collected into ``sla_metadata`` keyed by
    their ``property`` name.

    Args:
        sla_list: Raw ``slaProperties`` list from the ODCS document.
        schema_objects: Full ``schema[]`` list (used for partition fallback).

    Returns:
        Tuple of ``(sla_latency, sla_metadata)``.
    """
    sla_latency: SlaLatency | None = None
    sla_metadata: dict[str, Any] = {}

    for entry in sla_list:
        prop_name: str = str(entry.get("property", ""))
        if prop_name in _LATENCY_SYNONYMS:
            raw_value = entry.get("value")
            if raw_value is None:
                warnings.warn(
                    "SLA latency entry is missing the 'value' field; skipping",
                    ContractWarning,
                    stacklevel=4,
                )
                continue
            value = float(raw_value)
            unit = str(entry.get("unit", "h"))
            max_age_hours = _unit_to_hours(value, unit)
            timestamp_col = _resolve_timestamp_column(entry, schema_objects)
            if timestamp_col is None:
                warnings.warn(
                    "SLA latency rule: no timestamp column could be resolved; freshness check skipped",
                    ContractWarning,
                    stacklevel=4,
                )
            else:
                sla_latency = SlaLatency(
                    max_age_hours=max_age_hours,
                    timestamp_column=timestamp_col,
                )
        else:
            entry_without_property = {k: v for k, v in entry.items() if k != "property"}
            sla_metadata[prop_name] = entry_without_property

    return sla_latency, sla_metadata


# ---------------------------------------------------------------------------
# Library metric rule execution
# ---------------------------------------------------------------------------


def _execute_library_rule(
    rule: dict[str, Any],
    property_name: str | None,
    mp: MetricProvider,
    ctx: Context,
) -> None:
    """Execute one ODCS library metric rule as a DQX assertion.

    Dispatches on ``rule["metric"]`` to the appropriate :class:`MetricProvider`
    call.  The resulting metric expression is passed through
    :func:`_apply_odcs_operators` to produce the assertion.

    Args:
        rule: Raw ODCS quality rule dict with ``metric`` field present.
        property_name: Column name when the rule is property-level;
            ``None`` for table-level rules.
        mp: Active :class:`~dqx.provider.MetricProvider`.
        ctx: Active :class:`~dqx.api.Context`.

    Raises:
        ContractValidationError: If ``mustBeBetween`` or ``mustNotBeBetween``
            is malformed (propagated from :func:`_apply_odcs_operators`).
    """
    metric_name: str = str(rule.get("metric", ""))
    rule_name: str = str(rule.get("name", metric_name))
    severity = _severity_from_odcs(rule.get("severity"))
    unit: str = str(rule.get("unit", ""))
    arguments: dict[str, Any] = rule.get("arguments") or {}

    if metric_name == "rowCount":
        metric_expr = mp.num_rows()

    elif metric_name in ("nullValues", "missingValues"):
        if metric_name == "missingValues" and arguments.get("missingValues") is not None:
            warnings.warn(
                f"Quality rule '{rule_name}': metric 'missingValues' with missingValues "
                "argument is not executable in DQX; skipping",
                ContractWarning,
                stacklevel=4,
            )
            return
        if property_name is None:
            raise ContractValidationError(
                f"Quality rule '{rule_name}': metric '{metric_name}' requires a property-level column; "
                "found at table level with no column name"
            )
        col = property_name
        metric_expr = mp.null_count(col)
        if unit == "percent":
            metric_expr = metric_expr / mp.num_rows()

    elif metric_name == "invalidValues":
        if arguments.get("pattern") is not None:
            warnings.warn(
                f"Quality rule '{rule_name}': metric 'invalidValues' with pattern "
                "argument is not executable in DQX; skipping",
                ContractWarning,
                stacklevel=4,
            )
            return
        if property_name is None:
            raise ContractValidationError(
                f"Quality rule '{rule_name}': metric 'invalidValues' requires a property-level column; "
                "found at table level with no column name"
            )
        col = property_name
        valid_values: list[Any] = rule.get("validValues") or []
        metric_expr = mp.num_rows() - mp.count_values(col, valid_values)
        if unit == "percent":
            metric_expr = metric_expr / mp.num_rows()

    elif metric_name == "duplicateValues":
        if property_name is not None:
            # Property-level: single column
            metric_expr = mp.duplicate_count([str(property_name)])
        else:
            # Table-level: composite key from arguments.properties
            props: list[str] = [str(p) for p in (arguments.get("properties") or [])]
            metric_expr = mp.duplicate_count(props)
        if unit == "percent":
            metric_expr = metric_expr / mp.num_rows()

    else:
        # Unknown metric — skip silently (no assertion produced)
        return

    ready = ctx.assert_that(metric_expr).config(name=rule_name, severity=severity)
    _apply_odcs_operators(ready, rule)


# ---------------------------------------------------------------------------
# Custom DQX rule execution (type: custom, engine: dqx)
# ---------------------------------------------------------------------------


def _execute_custom_dqx_rule(
    rule: dict[str, Any],
    mp: MetricProvider,
    ctx: Context,
) -> None:
    """Execute one ODCS ``type: custom, engine: dqx`` rule.

    Parses the ``implementation:`` field as YAML and dispatches on the
    ``check:`` key to the appropriate :class:`MetricProvider` call.

    Args:
        rule: Raw ODCS quality rule dict with ``type: custom`` and
            ``engine: dqx``.
        mp: Active :class:`~dqx.provider.MetricProvider`.
        ctx: Active :class:`~dqx.api.Context`.

    Raises:
        NotImplementedError: For ``check: avg_length``, ``check: percentile``,
            and ``check: pattern`` which lack a backing MetricProvider method.
        ContractValidationError: For unknown ``check:`` values, missing
            ``column:`` field for column-level checks, or invalid
            ``column_type:`` value for ``min_length`` / ``max_length``.
    """
    impl: dict[str, Any] = yaml.safe_load(str(rule.get("implementation", ""))) or {}
    check_type: str = str(impl.get("check", ""))
    column: str | None = impl.get("column")
    return_pct: bool = impl.get("return") == "pct"
    rule_name: str = str(rule.get("name", check_type))
    severity = _severity_from_odcs(rule.get("severity"))

    if check_type == "num_rows":
        metric_expr = mp.num_rows()

    elif check_type == "missing":
        col = _require_column(column, rule_name, check_type)
        metric_expr = mp.null_count(col)
        if return_pct:
            metric_expr = metric_expr / mp.num_rows()

    elif check_type == "duplicates":
        col = _require_column(column, rule_name, check_type)
        metric_expr = mp.duplicate_count([col])
        if return_pct:
            metric_expr = metric_expr / mp.num_rows()

    elif check_type == "whitelist":
        col = _require_column(column, rule_name, check_type)
        values: list[Any] = impl.get("values") or []
        metric_expr = mp.count_values(col, values)
        if return_pct:
            metric_expr = metric_expr / mp.num_rows()

    elif check_type == "blacklist":
        col = _require_column(column, rule_name, check_type)
        values = impl.get("values") or []
        metric_expr = mp.num_rows() - mp.count_values(col, values)
        if return_pct:
            metric_expr = metric_expr / mp.num_rows()

    elif check_type == "cardinality":
        metric_expr = mp.unique_count(_require_column(column, rule_name, check_type))

    elif check_type == "min":
        metric_expr = mp.minimum(_require_column(column, rule_name, check_type))

    elif check_type == "max":
        metric_expr = mp.maximum(_require_column(column, rule_name, check_type))

    elif check_type == "mean":
        metric_expr = mp.average(_require_column(column, rule_name, check_type))

    elif check_type == "sum":
        metric_expr = mp.sum(_require_column(column, rule_name, check_type))

    elif check_type == "count":
        col = _require_column(column, rule_name, check_type)
        metric_expr = mp.num_rows() - mp.null_count(col)

    elif check_type == "variance":
        metric_expr = mp.variance(_require_column(column, rule_name, check_type))

    elif check_type == "stddev":
        metric_expr = sp.sqrt(mp.variance(_require_column(column, rule_name, check_type)))

    elif check_type == "min_length":
        col_type_raw: str = str(impl.get("column_type", "string"))
        if col_type_raw not in _LENGTH_COLUMN_TYPES:
            raise ContractValidationError(
                f"Quality rule '{rule_name}': check 'min_length' column_type must be one of "
                f"{sorted(_LENGTH_COLUMN_TYPES)!r}, got '{col_type_raw}'"
            )
        col_type: Literal["string", "list", "map"] = col_type_raw  # type: ignore[assignment]
        metric_expr = mp.min_length(_require_column(column, rule_name, check_type), col_type)

    elif check_type == "max_length":
        col_type_raw = str(impl.get("column_type", "string"))
        if col_type_raw not in _LENGTH_COLUMN_TYPES:
            raise ContractValidationError(
                f"Quality rule '{rule_name}': check 'max_length' column_type must be one of "
                f"{sorted(_LENGTH_COLUMN_TYPES)!r}, got '{col_type_raw}'"
            )
        col_type = col_type_raw  # type: ignore[assignment]
        metric_expr = mp.max_length(_require_column(column, rule_name, check_type), col_type)

    elif check_type == "avg_length":
        raise NotImplementedError("check: avg_length requires MetricProvider.avg_length() which is not yet implemented")

    elif check_type == "percentile":
        raise NotImplementedError("check: percentile requires MetricProvider.percentile() which is not yet implemented")

    elif check_type == "pattern":
        raise NotImplementedError("check: pattern requires MetricProvider.pattern_match() which is not yet implemented")

    else:
        raise ContractValidationError(f"Unknown DQX check type '{check_type}' in custom rule '{rule_name}'")

    ready = ctx.assert_that(metric_expr).config(name=rule_name, severity=severity)
    # Operators may appear at the rule level (non-ODCS-standard but accepted)
    # or inside the implementation: block (ODCS-valid placement for custom rules).
    # Rule-level operators take precedence; fall back to implementation-level.
    operator_source = rule if any(k in rule for k in _ODCS_OPERATOR_FIELDS) else impl
    _apply_odcs_operators(ready, operator_source)


# ---------------------------------------------------------------------------
# Rule dispatcher and Contract.to_checks()
# ---------------------------------------------------------------------------


def _dispatch_rule(
    rule: dict[str, Any],
    property_name: str | None,
    mp: MetricProvider,
    ctx: Context,
) -> None:
    """Route one ODCS quality rule to the appropriate executor.

    Args:
        rule: Raw ODCS quality rule dict.
        property_name: Column name when property-level; ``None`` for
            table-level rules.
        mp: Active :class:`~dqx.provider.MetricProvider`.
        ctx: Active :class:`~dqx.api.Context`.
    """
    rule_type: str = str(rule.get("type", "library"))
    rule_name: str = str(rule.get("name", ""))

    if rule_type == "text":
        return

    if rule_type == "sql":
        warnings.warn(
            f"Quality rule '{rule_name}': type 'sql' is not executable in DQX; skipping",
            ContractWarning,
            stacklevel=5,
        )
        return

    if rule_type == "custom":
        engine: str = str(rule.get("engine", ""))
        if engine != "dqx":
            warnings.warn(
                f"Quality rule '{rule_name}': type 'custom' with engine '{engine}' is not executable in DQX; skipping",
                ContractWarning,
                stacklevel=5,
            )
            return
        _execute_custom_dqx_rule(rule, mp, ctx)
        return

    # Default: library rule (metric: field present, or type: library)
    _execute_library_rule(rule, property_name, mp, ctx)
