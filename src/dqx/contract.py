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

Note:
    ``to_checks()`` is **not yet implemented** and will be added in a later
    phase once quality-rule parsing and ``MetricProvider`` mapping are in place.
"""

from __future__ import annotations

import importlib.resources
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft201909Validator
from jsonschema import ValidationError as _JsonSchemaValidationError


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

    Note:
        This class is a placeholder for Phase 2 (``to_checks()`` implementation).
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
        sla_metadata: Dict of non-latency SLA properties keyed by property
            name.  Values are the full SLA entry dicts (excluding the
            ``property`` key itself).
    """

    name: str | None
    version: str
    api_version: str
    contract_id: str
    status: str
    dataset: str
    schema_def: ContractSchema
    sla_latency: SlaLatency | None
    sla_metadata: dict[str, Any]

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
                    sla_metadata=sla_metadata,
                )
            )
        return contracts


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
            value = float(entry.get("value", 0))
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
