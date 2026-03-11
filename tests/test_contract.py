"""Tests for the ODCS Contract class and from_odcs() parsing.

This module covers:
- JSON Schema validation (ContractValidationError on structural errors)
- Parsing single and multi-schema ODCS files
- Property parsing (logicalType, required, partitioned, etc.)
- Quality rule categorisation (text vs executable)
- SLA latency resolution (element:, partitioning, warnings)
- SLA metadata storage for non-latency properties
- Error conditions (missing file, invalid YAML)
- Immutability of all dataclasses
- Severity and operator translation helpers
"""

from __future__ import annotations

import datetime
import textwrap
import warnings
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from dqx.api import VerificationSuite
from dqx.common import AssertionResult, ResultKey
from dqx.contract import (
    Contract,
    ContractProperty,
    ContractSchema,
    ContractValidationError,
    ContractWarning,
    SlaLatency,
    _apply_odcs_operators,
    _execute_custom_dqx_rule,
    _execute_library_rule,
    _severity_from_odcs,
)
import dqx
from dqx.datasource import DuckRelationDataSource
from dqx.orm.repositories import InMemoryMetricDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    """Write a YAML string to a temp file and return the path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


_MINIMAL_SCHEMA = """\
    apiVersion: v3.1.0
    kind: DataContract
    id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
    version: "1.0.0"
    status: active
    schema:
      - name: orders
"""

_MINIMAL_NO_SCHEMA = """\
    apiVersion: v3.1.0
    kind: DataContract
    id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
    version: "1.0.0"
    status: active
"""


# ---------------------------------------------------------------------------
# ContractValidationError — JSON Schema enforcement
# ---------------------------------------------------------------------------


class TestContractValidationError:
    """Contract.from_odcs() raises ContractValidationError for invalid docs."""

    def test_missing_version(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            status: active
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        with pytest.raises(ContractValidationError, match="version"):
            Contract.from_odcs(path)

    def test_missing_api_version(self, tmp_path: Path) -> None:
        content = """\
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        with pytest.raises(ContractValidationError, match="apiVersion"):
            Contract.from_odcs(path)

    def test_missing_kind(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        with pytest.raises(ContractValidationError, match="kind"):
            Contract.from_odcs(path)

    def test_missing_id(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            version: "1.0.0"
            status: active
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        with pytest.raises(ContractValidationError, match="id"):
            Contract.from_odcs(path)

    def test_missing_status(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        with pytest.raises(ContractValidationError, match="status"):
            Contract.from_odcs(path)

    def test_wrong_kind_enum(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: WrongKind
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        with pytest.raises(ContractValidationError):
            Contract.from_odcs(path)

    def test_wrong_api_version_enum(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v99.0.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        with pytest.raises(ContractValidationError):
            Contract.from_odcs(path)


# ---------------------------------------------------------------------------
# File-level errors
# ---------------------------------------------------------------------------


class TestContractFileErrors:
    """Errors raised before YAML or JSON Schema processing."""

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Contract.from_odcs(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("key: [unclosed bracket\n")
        with pytest.raises(yaml.YAMLError):
            Contract.from_odcs(p)


# ---------------------------------------------------------------------------
# Parsing — single and multi-schema
# ---------------------------------------------------------------------------


class TestContractFromOdcsParsing:
    """Contract.from_odcs() returns the correct number of Contract instances."""

    def test_single_schema_returns_list_of_one(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        contracts = Contract.from_odcs(path)
        assert isinstance(contracts, list)
        assert len(contracts) == 1

    def test_no_schema_key_returns_empty_list(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_NO_SCHEMA)
        contracts = Contract.from_odcs(path)
        assert contracts == []

    def test_multi_schema_returns_one_per_object(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
              - name: customers
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        contracts = Contract.from_odcs(path)
        assert len(contracts) == 2

    def test_dataset_prefers_physical_name(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
                physicalName: orders_tbl
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        contracts = Contract.from_odcs(path)
        assert contracts[0].dataset == "orders_tbl"

    def test_dataset_falls_back_to_name(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        contracts = Contract.from_odcs(path)
        assert contracts[0].dataset == "orders"

    def test_top_level_name_stored_on_contract(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            name: orders_contract
            version: "1.0.0"
            status: active
            schema:
              - name: orders
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        contracts = Contract.from_odcs(path)
        assert contracts[0].name == "orders_contract"

    def test_top_level_name_absent_is_none(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        contracts = Contract.from_odcs(path)
        assert contracts[0].name is None

    def test_version_and_api_version_stored(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        c = Contract.from_odcs(path)[0]
        assert c.version == "1.0.0"
        assert c.api_version == "v3.1.0"

    def test_contract_id_stored(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        c = Contract.from_odcs(path)[0]
        assert c.contract_id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def test_status_stored(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        c = Contract.from_odcs(path)[0]
        assert c.status == "active"


# ---------------------------------------------------------------------------
# Property parsing
# ---------------------------------------------------------------------------


class TestContractProperties:
    """ContractProperty fields are parsed correctly from ODCS properties[]."""

    def _contract_with_props(self, tmp_path: Path, props_yaml: str) -> Contract:
        content = f"""\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
                properties:
{textwrap.indent(textwrap.dedent(props_yaml), "                  ")}
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        return Contract.from_odcs(path)[0]

    def test_logical_type_parsed(self, tmp_path: Path) -> None:
        c = self._contract_with_props(tmp_path, "- name: order_id\n  logicalType: integer\n")
        prop = c.schema_def.properties[0]
        assert prop.logical_type == "integer"

    def test_physical_type_parsed(self, tmp_path: Path) -> None:
        c = self._contract_with_props(tmp_path, "- name: order_id\n  physicalType: BIGINT\n")
        prop = c.schema_def.properties[0]
        assert prop.physical_type == "BIGINT"

    def test_required_true_parsed(self, tmp_path: Path) -> None:
        c = self._contract_with_props(tmp_path, "- name: order_id\n  required: true\n")
        prop = c.schema_def.properties[0]
        assert prop.required is True

    def test_required_defaults_to_false(self, tmp_path: Path) -> None:
        c = self._contract_with_props(tmp_path, "- name: order_id\n")
        prop = c.schema_def.properties[0]
        assert prop.required is False

    def test_partitioned_true_parsed(self, tmp_path: Path) -> None:
        c = self._contract_with_props(tmp_path, "- name: order_date\n  partitioned: true\n  partitionKeyPosition: 1\n")
        prop = c.schema_def.properties[0]
        assert prop.partitioned is True
        assert prop.partition_key_position == 1

    def test_partitioned_defaults_to_false(self, tmp_path: Path) -> None:
        c = self._contract_with_props(tmp_path, "- name: order_id\n")
        prop = c.schema_def.properties[0]
        assert prop.partitioned is False
        assert prop.partition_key_position == -1

    def test_logical_type_absent_is_none(self, tmp_path: Path) -> None:
        c = self._contract_with_props(tmp_path, "- name: order_id\n")
        prop = c.schema_def.properties[0]
        assert prop.logical_type is None

    def test_physical_type_absent_is_none(self, tmp_path: Path) -> None:
        c = self._contract_with_props(tmp_path, "- name: order_id\n")
        prop = c.schema_def.properties[0]
        assert prop.physical_type is None

    def test_multiple_properties_parsed(self, tmp_path: Path) -> None:
        c = self._contract_with_props(
            tmp_path,
            "- name: order_id\n  logicalType: integer\n- name: status\n  logicalType: string\n",
        )
        names = [p.name for p in c.schema_def.properties]
        assert names == ["order_id", "status"]

    def test_logical_type_options_parsed(self, tmp_path: Path) -> None:
        props_yaml = """\
            - name: created_at
              logicalType: timestamp
              logicalTypeOptions:
                timezone: true
                defaultTimezone: UTC
        """
        c = self._contract_with_props(tmp_path, props_yaml)
        prop = c.schema_def.properties[0]
        assert prop.logical_type_options == {"timezone": True, "defaultTimezone": "UTC"}

    def test_logical_type_options_absent_is_none(self, tmp_path: Path) -> None:
        c = self._contract_with_props(tmp_path, "- name: order_id\n")
        prop = c.schema_def.properties[0]
        assert prop.logical_type_options is None

    def test_no_properties_gives_empty_tuple(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        c = Contract.from_odcs(path)[0]
        assert c.schema_def.properties == ()

    def test_property_quality_rules_stored_on_property(self, tmp_path: Path) -> None:
        props_yaml = """\
            - name: order_id
              logicalType: integer
              quality:
                - name: not_null
                  type: library
                  metric: nullValues
                  mustBe: 0
        """
        c = self._contract_with_props(tmp_path, props_yaml)
        prop = c.schema_def.properties[0]
        assert len(prop.quality) == 1
        assert prop.quality[0]["name"] == "not_null"


# ---------------------------------------------------------------------------
# Quality rule categorisation
# ---------------------------------------------------------------------------


class TestContractQualityRules:
    """type: text rules go to text_rules; all others stay in quality."""

    def _contract_with_quality(self, tmp_path: Path, quality_yaml: str) -> Contract:
        content = f"""\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
                quality:
{textwrap.indent(textwrap.dedent(quality_yaml), "                  ")}
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        return Contract.from_odcs(path)[0]

    def test_text_rule_goes_to_text_rules(self, tmp_path: Path) -> None:
        quality_yaml = """\
            - name: doc_rule
              type: text
              description: "This is documentation"
        """
        c = self._contract_with_quality(tmp_path, quality_yaml)
        assert len(c.schema_def.text_rules) == 1
        assert c.schema_def.quality == ()

    def test_library_rule_goes_to_quality(self, tmp_path: Path) -> None:
        quality_yaml = """\
            - name: row_count_check
              type: library
              metric: rowCount
              mustBeGreaterOrEqualTo: 100
        """
        c = self._contract_with_quality(tmp_path, quality_yaml)
        assert len(c.schema_def.quality) == 1
        assert c.schema_def.text_rules == ()

    def test_sql_rule_goes_to_quality(self, tmp_path: Path) -> None:
        quality_yaml = """\
            - name: sql_check
              type: sql
              query: "SELECT COUNT(*) FROM orders WHERE id IS NULL"
              mustBe: 0
        """
        c = self._contract_with_quality(tmp_path, quality_yaml)
        assert len(c.schema_def.quality) == 1

    def test_custom_dqx_rule_goes_to_quality(self, tmp_path: Path) -> None:
        quality_yaml = """\
            - name: num_rows_check
              type: custom
              engine: dqx
              implementation: |
                check: num_rows
        """
        c = self._contract_with_quality(tmp_path, quality_yaml)
        assert len(c.schema_def.quality) == 1

    def test_mixed_rules_split_correctly(self, tmp_path: Path) -> None:
        quality_yaml = """\
            - name: doc_note
              type: text
              description: "Documentation note"
            - name: row_count_check
              type: library
              metric: rowCount
              mustBeGreaterOrEqualTo: 100
            - name: another_doc
              type: text
              description: "Another note"
        """
        c = self._contract_with_quality(tmp_path, quality_yaml)
        assert len(c.schema_def.text_rules) == 2
        assert len(c.schema_def.quality) == 1

    def test_no_quality_gives_empty_tuples(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        c = Contract.from_odcs(path)[0]
        assert c.schema_def.quality == ()
        assert c.schema_def.text_rules == ()


# ---------------------------------------------------------------------------
# SLA latency resolution
# ---------------------------------------------------------------------------


class TestContractSlaLatency:
    """SLA latency is resolved to SlaLatency(max_age_hours, timestamp_column)."""

    def _contract_with_sla(self, tmp_path: Path, sla_yaml: str, props_yaml: str = "") -> Contract:
        props_lines = ""
        if props_yaml:
            indented = textwrap.indent(textwrap.dedent(props_yaml).strip(), "                ")
            props_lines = f"\n                properties:\n{indented}"
        content = textwrap.dedent(f"""\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            slaProperties:
{textwrap.indent(textwrap.dedent(sla_yaml).strip(), "              ")}
            schema:
              - name: orders
                physicalName: orders_tbl{props_lines}
        """)
        path = _write_yaml(tmp_path, "c.yaml", content)
        return Contract.from_odcs(path)[0]

    def test_element_notation_resolves_timestamp_column(self, tmp_path: Path) -> None:
        sla_yaml = """\
            - property: latency
              value: 24
              unit: h
              element: orders_tbl.order_date
        """
        c = self._contract_with_sla(tmp_path, sla_yaml)
        assert c.sla_latency is not None
        assert c.sla_latency.timestamp_column == "order_date"
        assert c.sla_latency.max_age_hours == 24.0

    def test_partitioned_fallback_resolves_timestamp_column(self, tmp_path: Path) -> None:
        sla_yaml = """\
            - property: latency
              value: 12
              unit: h
        """
        props_yaml = """\
            - name: order_date
              logicalType: date
              partitioned: true
              partitionKeyPosition: 1
        """
        c = self._contract_with_sla(tmp_path, sla_yaml, props_yaml)
        assert c.sla_latency is not None
        assert c.sla_latency.timestamp_column == "order_date"

    def test_no_element_no_partition_emits_warning_and_none(self, tmp_path: Path) -> None:
        sla_yaml = """\
            - property: latency
              value: 24
              unit: h
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            c = self._contract_with_sla(tmp_path, sla_yaml)
        assert c.sla_latency is None
        assert any(issubclass(w.category, ContractWarning) for w in caught)

    def test_unit_hours_conversion(self, tmp_path: Path) -> None:
        for unit in ("h", "hr", "hour", "hours"):
            sla_yaml = f"""\
                - property: latency
                  value: 6
                  unit: {unit}
                  element: orders_tbl.order_date
            """
            c = self._contract_with_sla(tmp_path, sla_yaml)
            assert c.sla_latency is not None
            assert c.sla_latency.max_age_hours == 6.0, f"Failed for unit={unit}"

    def test_unit_days_conversion(self, tmp_path: Path) -> None:
        for unit in ("d", "day", "days"):
            sla_yaml = f"""\
                - property: latency
                  value: 2
                  unit: {unit}
                  element: orders_tbl.order_date
            """
            c = self._contract_with_sla(tmp_path, sla_yaml)
            assert c.sla_latency is not None
            assert c.sla_latency.max_age_hours == 48.0, f"Failed for unit={unit}"

    def test_unit_years_conversion(self, tmp_path: Path) -> None:
        for unit in ("y", "yr", "year", "years"):
            sla_yaml = f"""\
                - property: latency
                  value: 1
                  unit: {unit}
                  element: orders_tbl.order_date
            """
            c = self._contract_with_sla(tmp_path, sla_yaml)
            assert c.sla_latency is not None
            assert c.sla_latency.max_age_hours == 8760.0, f"Failed for unit={unit}"

    def test_ly_synonym_for_latency(self, tmp_path: Path) -> None:
        sla_yaml = """\
            - property: ly
              value: 24
              unit: h
              element: orders_tbl.order_date
        """
        c = self._contract_with_sla(tmp_path, sla_yaml)
        assert c.sla_latency is not None
        assert c.sla_latency.max_age_hours == 24.0

    def test_no_sla_properties_gives_none_latency(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        c = Contract.from_odcs(path)[0]
        assert c.sla_latency is None

    def test_element_extracts_last_part_after_dot(self, tmp_path: Path) -> None:
        sla_yaml = """\
            - property: latency
              value: 24
              unit: h
              element: orders_tbl.created_at
        """
        c = self._contract_with_sla(tmp_path, sla_yaml)
        assert c.sla_latency is not None
        assert c.sla_latency.timestamp_column == "created_at"

    def test_unknown_unit_raises_contract_validation_error(self, tmp_path: Path) -> None:
        sla_yaml = """\
            - property: latency
              value: 24
              unit: minutes
              element: orders_tbl.order_date
        """
        with pytest.raises(ContractValidationError, match="unrecognised unit"):
            self._contract_with_sla(tmp_path, sla_yaml)

    def test_missing_value_emits_warning_and_skips(self, tmp_path: Path) -> None:
        """A latency entry with a null 'value' field emits ContractWarning and is skipped."""
        sla_yaml = """\
            - property: latency
              value: null
              unit: h
              element: orders_tbl.order_date
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            c = self._contract_with_sla(tmp_path, sla_yaml)
        assert c.sla_latency is None
        assert any(issubclass(w.category, ContractWarning) for w in caught)
        assert any("missing" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# SLA metadata for non-latency properties
# ---------------------------------------------------------------------------


class TestContractSlaMetadata:
    """Non-latency SLA properties are stored in sla_metadata, not executed."""

    def test_retention_stored_in_sla_metadata(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            slaProperties:
              - property: retention
                value: 3
                unit: y
            schema:
              - name: orders
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        c = Contract.from_odcs(path)[0]
        assert "retention" in c.sla_metadata
        assert c.sla_metadata["retention"]["value"] == 3
        assert c.sla_metadata["retention"]["unit"] == "y"

    def test_multiple_non_latency_properties_all_stored(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            slaProperties:
              - property: retention
                value: 3
                unit: y
              - property: availability
                value: 99.9
                unit: percent
              - property: frequency
                value: 1
                unit: d
            schema:
              - name: orders
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        c = Contract.from_odcs(path)[0]
        assert set(c.sla_metadata.keys()) == {"retention", "availability", "frequency"}

    def test_latency_not_in_sla_metadata(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            slaProperties:
              - property: latency
                value: 24
                unit: h
                element: orders.order_date
              - property: retention
                value: 3
                unit: y
            schema:
              - name: orders
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        c = Contract.from_odcs(path)[0]
        assert "latency" not in c.sla_metadata
        assert "retention" in c.sla_metadata

    def test_no_sla_gives_empty_metadata(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        c = Contract.from_odcs(path)[0]
        assert c.sla_metadata == {}


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestContractImmutability:
    """All dataclasses are frozen (immutable after construction)."""

    def test_contract_is_frozen(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "c.yaml", _MINIMAL_SCHEMA)
        c = Contract.from_odcs(path)[0]
        with pytest.raises((AttributeError, TypeError)):
            c.name = "new_name"  # type: ignore[misc]

    def test_sla_latency_is_frozen(self) -> None:
        sla = SlaLatency(max_age_hours=24.0, timestamp_column="order_date")
        with pytest.raises((AttributeError, TypeError)):
            sla.max_age_hours = 48.0  # type: ignore[misc]

    def test_contract_property_is_frozen(self) -> None:
        prop = ContractProperty(
            name="order_id",
            logical_type="integer",
            physical_type=None,
            required=False,
            partitioned=False,
            partition_key_position=-1,
            logical_type_options=None,
            quality=(),
        )
        with pytest.raises((AttributeError, TypeError)):
            prop.name = "new_name"  # type: ignore[misc]

    def test_contract_schema_is_frozen(self) -> None:
        schema = ContractSchema(
            name="orders",
            physical_name=None,
            properties=(),
            quality=(),
            text_rules=(),
        )
        with pytest.raises((AttributeError, TypeError)):
            schema.name = "new_name"  # type: ignore[misc]

    def test_sla_metadata_is_immutable(self, tmp_path: Path) -> None:
        """sla_metadata is a read-only MappingProxyType; callers cannot mutate it."""
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            slaProperties:
              - property: retention
                value: 3
                unit: y
            schema:
              - name: orders
        """
        path = _write_yaml(tmp_path, "c.yaml", content)
        c = Contract.from_odcs(path)[0]
        with pytest.raises(TypeError):
            c.sla_metadata["new_key"] = "new_value"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Integration test — official ODCS full example
# https://bitol-io.github.io/open-data-contract-standard/latest/examples/all/full-example.odcs.yaml
# ---------------------------------------------------------------------------

_FULL_EXAMPLE = Path(__file__).parent / "fixtures" / "contracts" / "full-example.odcs.yaml"


class TestFullOdcsExample:
    """Load the official ODCS full example and assert parsed values.

    The fixture file is a verbatim copy of the canonical ODCS example:
    https://bitol-io.github.io/open-data-contract-standard/latest/examples/all/full-example.odcs.yaml

    Tests are organised by concern so that a single failing assertion pinpoints
    exactly which field or schema object is mis-parsed.
    """

    @pytest.fixture
    def contracts(self) -> list[Contract]:
        return Contract.from_odcs(_FULL_EXAMPLE)

    @pytest.fixture
    def tbl(self, contracts: list[Contract]) -> Contract:
        """The first schema object: tbl / tbl_1."""
        return contracts[0]

    @pytest.fixture
    def receivers(self, contracts: list[Contract]) -> Contract:
        """The second schema object: receivers / receivers_master."""
        return contracts[1]

    # ------------------------------------------------------------------
    # Top-level document fields
    # ------------------------------------------------------------------

    def test_parses_without_error(self, contracts: list[Contract]) -> None:
        """Full example passes JSON Schema validation and parses cleanly."""
        assert contracts is not None

    def test_returns_two_contracts(self, contracts: list[Contract]) -> None:
        """Two schema[] objects → two Contract instances."""
        assert len(contracts) == 2

    def test_top_level_name_is_absent(self, tbl: Contract) -> None:
        """The full example has no top-level 'name' field."""
        assert tbl.name is None

    def test_version(self, tbl: Contract) -> None:
        assert tbl.version == "1.1.0"

    def test_api_version(self, tbl: Contract) -> None:
        assert tbl.api_version == "v3.1.0"

    def test_contract_id(self, tbl: Contract) -> None:
        assert tbl.contract_id == "53581432-6c55-4ba2-a65f-72344a91553a"

    def test_status(self, tbl: Contract) -> None:
        assert tbl.status == "active"

    # ------------------------------------------------------------------
    # Schema 0: tbl / tbl_1
    # ------------------------------------------------------------------

    def test_tbl_dataset_uses_physical_name(self, tbl: Contract) -> None:
        """physicalName 'tbl_1' is preferred over logical name 'tbl'."""
        assert tbl.dataset == "tbl_1"

    def test_tbl_schema_logical_name(self, tbl: Contract) -> None:
        assert tbl.schema_def.name == "tbl"

    def test_tbl_schema_physical_name(self, tbl: Contract) -> None:
        assert tbl.schema_def.physical_name == "tbl_1"

    def test_tbl_has_three_properties(self, tbl: Contract) -> None:
        assert len(tbl.schema_def.properties) == 3

    def test_tbl_property_names_in_order(self, tbl: Contract) -> None:
        names = [p.name for p in tbl.schema_def.properties]
        assert names == ["transaction_reference_date", "rcvr_id", "rcvr_cntry_code"]

    def test_tbl_has_one_table_level_quality_rule(self, tbl: Contract) -> None:
        """The rowCount library rule is stored in schema_def.quality."""
        assert len(tbl.schema_def.quality) == 1

    def test_tbl_table_quality_rule_metric(self, tbl: Contract) -> None:
        assert tbl.schema_def.quality[0]["metric"] == "rowCount"

    def test_tbl_no_text_rules(self, tbl: Contract) -> None:
        assert tbl.schema_def.text_rules == ()

    # transaction_reference_date property
    def test_txn_ref_dt_logical_type(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[0]
        assert prop.logical_type == "date"

    def test_txn_ref_dt_physical_type(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[0]
        assert prop.physical_type == "date"

    def test_txn_ref_dt_not_required(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[0]
        assert prop.required is False

    def test_txn_ref_dt_is_partitioned(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[0]
        assert prop.partitioned is True

    def test_txn_ref_dt_partition_key_position(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[0]
        assert prop.partition_key_position == 1

    def test_txn_ref_dt_no_property_quality_rules(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[0]
        assert prop.quality == ()

    # rcvr_id property
    def test_rcvr_id_logical_type(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[1]
        assert prop.logical_type == "string"

    def test_rcvr_id_physical_type(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[1]
        assert prop.physical_type == "varchar(18)"

    def test_rcvr_id_not_required(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[1]
        assert prop.required is False

    def test_rcvr_id_not_partitioned(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[1]
        assert prop.partitioned is False

    # rcvr_cntry_code property
    def test_rcvr_cntry_code_logical_type(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[2]
        assert prop.logical_type == "string"

    def test_rcvr_cntry_code_has_one_quality_rule(self, tbl: Contract) -> None:
        """nullValues library rule is stored on the property."""
        prop = tbl.schema_def.properties[2]
        assert len(prop.quality) == 1

    def test_rcvr_cntry_code_quality_rule_metric(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[2]
        assert prop.quality[0]["metric"] == "nullValues"

    def test_rcvr_cntry_code_quality_rule_severity(self, tbl: Contract) -> None:
        prop = tbl.schema_def.properties[2]
        assert prop.quality[0]["severity"] == "error"

    # ------------------------------------------------------------------
    # Schema 1: receivers / receivers_master
    # ------------------------------------------------------------------

    def test_receivers_dataset_uses_physical_name(self, receivers: Contract) -> None:
        assert receivers.dataset == "receivers_master"

    def test_receivers_schema_logical_name(self, receivers: Contract) -> None:
        assert receivers.schema_def.name == "receivers"

    def test_receivers_has_four_properties(self, receivers: Contract) -> None:
        assert len(receivers.schema_def.properties) == 4

    def test_receivers_property_names_in_order(self, receivers: Contract) -> None:
        names = [p.name for p in receivers.schema_def.properties]
        assert names == ["id", "country_code", "receiver_name", "receiver_type"]

    def test_receivers_no_table_level_quality_rules(self, receivers: Contract) -> None:
        assert receivers.schema_def.quality == ()

    def test_receivers_id_required(self, receivers: Contract) -> None:
        assert receivers.schema_def.properties[0].required is True

    def test_receivers_country_code_required(self, receivers: Contract) -> None:
        assert receivers.schema_def.properties[1].required is True

    def test_receivers_receiver_name_required(self, receivers: Contract) -> None:
        assert receivers.schema_def.properties[2].required is True

    def test_receivers_receiver_type_not_required(self, receivers: Contract) -> None:
        assert receivers.schema_def.properties[3].required is False

    def test_receivers_all_string_logical_types(self, receivers: Contract) -> None:
        types = [p.logical_type for p in receivers.schema_def.properties]
        assert all(t == "string" for t in types)

    # ------------------------------------------------------------------
    # SLA
    # ------------------------------------------------------------------

    def test_sla_latency_resolved(self, tbl: Contract) -> None:
        """latency entry resolves to a SlaLatency instance (not None)."""
        assert tbl.sla_latency is not None

    def test_sla_latency_max_age_hours(self, tbl: Contract) -> None:
        """value=4, unit=d → 4 * 24 = 96 hours."""
        assert tbl.sla_latency is not None
        assert tbl.sla_latency.max_age_hours == 96.0

    def test_sla_latency_timestamp_column(self, tbl: Contract) -> None:
        """element='tab1.txn_ref_dt' → timestamp_column='txn_ref_dt'."""
        assert tbl.sla_latency is not None
        assert tbl.sla_latency.timestamp_column == "txn_ref_dt"

    def test_sla_metadata_does_not_contain_latency(self, tbl: Contract) -> None:
        assert "latency" not in tbl.sla_metadata

    def test_sla_metadata_contains_general_availability(self, tbl: Contract) -> None:
        assert "generalAvailability" in tbl.sla_metadata

    def test_sla_metadata_contains_end_of_support(self, tbl: Contract) -> None:
        assert "endOfSupport" in tbl.sla_metadata

    def test_sla_metadata_contains_end_of_life(self, tbl: Contract) -> None:
        assert "endOfLife" in tbl.sla_metadata

    def test_sla_metadata_retention_value(self, tbl: Contract) -> None:
        assert tbl.sla_metadata["retention"]["value"] == 3

    def test_sla_metadata_retention_unit(self, tbl: Contract) -> None:
        assert tbl.sla_metadata["retention"]["unit"] == "y"

    def test_sla_metadata_frequency_value(self, tbl: Contract) -> None:
        assert tbl.sla_metadata["frequency"]["value"] == 1

    def test_sla_metadata_time_of_availability_present(self, tbl: Contract) -> None:
        """The ODCS example has two timeOfAvailability entries with the same
        property name. The current dict-based sla_metadata stores one key per
        property name, so the second entry (driver: analytics) overwrites the
        first (driver: regulatory). We assert the key is present and that the
        surviving entry has the analytics driver value.
        """
        assert "timeOfAvailability" in tbl.sla_metadata
        assert tbl.sla_metadata["timeOfAvailability"]["driver"] == "analytics"

    # ------------------------------------------------------------------
    # Both contracts share the same top-level SLA (parsed once per file)
    # ------------------------------------------------------------------

    def test_receivers_sla_latency_same_as_tbl(self, tbl: Contract, receivers: Contract) -> None:
        """SLA is parsed at the file level; both contracts carry the same value."""
        assert receivers.sla_latency == tbl.sla_latency

    def test_receivers_sla_metadata_same_as_tbl(self, tbl: Contract, receivers: Contract) -> None:
        assert receivers.sla_metadata == tbl.sla_metadata


# ---------------------------------------------------------------------------
# Helpers shared by translation-helper tests
# ---------------------------------------------------------------------------


def _run_check_fn(check_fn: object, data: pa.Table, dataset: str = "t") -> list[AssertionResult]:
    """Run a bare check function (not decorated) inside a minimal suite."""
    from dqx.api import check as dqx_check

    decorated = dqx_check(name="test-check", datasets=[dataset])(check_fn)  # type: ignore[arg-type]
    datasource = DuckRelationDataSource.from_arrow(data, dataset)
    db = InMemoryMetricDB()
    suite = VerificationSuite(checks=[decorated], db=db, name="test-suite")
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})
    suite.run([datasource], key)
    return suite.collect_results()


# ---------------------------------------------------------------------------
# _severity_from_odcs
# ---------------------------------------------------------------------------


class TestSeverityFromOdcs:
    """Unit tests for _severity_from_odcs()."""

    def test_error_maps_to_p0(self) -> None:
        assert _severity_from_odcs("error") == "P0"

    def test_warning_maps_to_p1(self) -> None:
        assert _severity_from_odcs("warning") == "P1"

    def test_none_maps_to_p1(self) -> None:
        assert _severity_from_odcs(None) == "P1"

    def test_unknown_string_maps_to_p1(self) -> None:
        """Any unrecognised severity string defaults to P1."""
        assert _severity_from_odcs("info") == "P1"


# ---------------------------------------------------------------------------
# _apply_odcs_operators
# ---------------------------------------------------------------------------


class TestApplyOdcsOperators:
    """Integration tests for _apply_odcs_operators() via a minimal suite."""

    _DATA = pa.table({"v": [10, 20, 30]})

    def _run(self, rule: dict) -> list[AssertionResult]:
        from dqx.api import Context
        from dqx.provider import MetricProvider

        def check_fn(mp: MetricProvider, ctx: Context) -> None:
            metric = mp.num_rows()
            ready = ctx.assert_that(metric).config(name="test-assertion", severity="P1")
            _apply_odcs_operators(ready, rule)

        return _run_check_fn(check_fn, self._DATA)

    # ------------------------------------------------------------------
    # noop when no operator
    # ------------------------------------------------------------------

    def test_no_operator_produces_noop(self) -> None:
        results = self._run({})
        assert len(results) == 1
        assert results[0].status == "PASSED"

    # ------------------------------------------------------------------
    # mustBe
    # ------------------------------------------------------------------

    def test_must_be_passes_when_equal(self) -> None:
        results = self._run({"mustBe": 3})
        assert results[0].status == "PASSED"

    def test_must_be_fails_when_not_equal(self) -> None:
        results = self._run({"mustBe": 99})
        assert results[0].status == "FAILED"

    # ------------------------------------------------------------------
    # mustNotBe
    # ------------------------------------------------------------------

    def test_must_not_be_passes_when_different(self) -> None:
        results = self._run({"mustNotBe": 99})
        assert results[0].status == "PASSED"

    def test_must_not_be_fails_when_equal(self) -> None:
        results = self._run({"mustNotBe": 3})
        assert results[0].status == "FAILED"

    # ------------------------------------------------------------------
    # mustBeGreaterThan
    # ------------------------------------------------------------------

    def test_must_be_greater_than_passes(self) -> None:
        results = self._run({"mustBeGreaterThan": 2})
        assert results[0].status == "PASSED"

    def test_must_be_greater_than_fails_when_equal(self) -> None:
        results = self._run({"mustBeGreaterThan": 3})
        assert results[0].status == "FAILED"

    # ------------------------------------------------------------------
    # mustBeGreaterOrEqualTo
    # ------------------------------------------------------------------

    def test_must_be_greater_or_equal_to_passes_when_equal(self) -> None:
        results = self._run({"mustBeGreaterOrEqualTo": 3})
        assert results[0].status == "PASSED"

    def test_must_be_greater_or_equal_to_passes_when_greater(self) -> None:
        results = self._run({"mustBeGreaterOrEqualTo": 1})
        assert results[0].status == "PASSED"

    def test_must_be_greater_or_equal_to_fails(self) -> None:
        results = self._run({"mustBeGreaterOrEqualTo": 4})
        assert results[0].status == "FAILED"

    # ------------------------------------------------------------------
    # mustBeLessThan
    # ------------------------------------------------------------------

    def test_must_be_less_than_passes(self) -> None:
        results = self._run({"mustBeLessThan": 4})
        assert results[0].status == "PASSED"

    def test_must_be_less_than_fails_when_equal(self) -> None:
        results = self._run({"mustBeLessThan": 3})
        assert results[0].status == "FAILED"

    # ------------------------------------------------------------------
    # mustBeLessOrEqualTo
    # ------------------------------------------------------------------

    def test_must_be_less_or_equal_to_passes_when_equal(self) -> None:
        results = self._run({"mustBeLessOrEqualTo": 3})
        assert results[0].status == "PASSED"

    def test_must_be_less_or_equal_to_fails(self) -> None:
        results = self._run({"mustBeLessOrEqualTo": 2})
        assert results[0].status == "FAILED"

    # ------------------------------------------------------------------
    # mustBeBetween
    # ------------------------------------------------------------------

    def test_must_be_between_passes(self) -> None:
        results = self._run({"mustBeBetween": [1, 5]})
        assert results[0].status == "PASSED"

    def test_must_be_between_fails_outside(self) -> None:
        results = self._run({"mustBeBetween": [10, 20]})
        assert results[0].status == "FAILED"

    def test_must_be_between_scalar_raises(self) -> None:
        with pytest.raises(ContractValidationError, match="mustBeBetween"):
            self._run({"mustBeBetween": 3})

    def test_must_be_between_wrong_length_raises(self) -> None:
        with pytest.raises(ContractValidationError, match="mustBeBetween"):
            self._run({"mustBeBetween": [1, 2, 3]})

    # ------------------------------------------------------------------
    # mustNotBeBetween
    # ------------------------------------------------------------------

    def test_must_not_be_between_passes_outside(self) -> None:
        results = self._run({"mustNotBeBetween": [10, 20]})
        assert results[0].status == "PASSED"

    def test_must_not_be_between_fails_inside(self) -> None:
        results = self._run({"mustNotBeBetween": [1, 5]})
        assert results[0].status == "FAILED"

    def test_must_not_be_between_scalar_raises(self) -> None:
        with pytest.raises(ContractValidationError, match="mustNotBeBetween"):
            self._run({"mustNotBeBetween": 3})


# ---------------------------------------------------------------------------
# Helpers shared by library-rule and custom-rule tests
# ---------------------------------------------------------------------------


def _run_library_rule(
    rule: dict,
    data: pa.Table,
    dataset: str = "t",
    property_name: str | None = None,
) -> list[AssertionResult]:
    """Execute a single library rule via _execute_library_rule inside a suite."""
    from dqx.api import Context, check as dqx_check
    from dqx.provider import MetricProvider

    def check_fn(mp: MetricProvider, ctx: Context) -> None:
        _execute_library_rule(rule, property_name, mp, ctx)

    decorated = dqx_check(name="test-check", datasets=[dataset])(check_fn)
    datasource = DuckRelationDataSource.from_arrow(data, dataset)
    db = InMemoryMetricDB()
    suite = VerificationSuite(checks=[decorated], db=db, name="test-suite")
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})
    suite.run([datasource], key)
    return suite.collect_results()


# ---------------------------------------------------------------------------
# _execute_library_rule
# ---------------------------------------------------------------------------


class TestLibraryRuleExecution:
    """Integration tests for _execute_library_rule()."""

    # ------------------------------------------------------------------
    # rowCount
    # ------------------------------------------------------------------

    def test_row_count_passes_when_geq(self) -> None:
        data = pa.table({"v": [1, 2, 3]})
        rule = {"name": "rc", "metric": "rowCount", "mustBeGreaterOrEqualTo": 3}
        results = _run_library_rule(rule, data)
        assert len(results) == 1
        assert results[0].status == "PASSED"

    def test_row_count_fails_when_below(self) -> None:
        data = pa.table({"v": [1, 2, 3]})
        rule = {"name": "rc", "metric": "rowCount", "mustBeGreaterOrEqualTo": 10}
        results = _run_library_rule(rule, data)
        assert results[0].status == "FAILED"

    def test_row_count_assertion_name_uses_rule_name(self) -> None:
        data = pa.table({"v": [1]})
        rule = {"name": "my_row_count_check", "metric": "rowCount", "mustBeGreaterOrEqualTo": 1}
        results = _run_library_rule(rule, data)
        assert results[0].assertion == "my_row_count_check"

    def test_row_count_severity_error_maps_to_p0(self) -> None:
        data = pa.table({"v": [1]})
        rule = {"name": "rc", "metric": "rowCount", "mustBeGreaterOrEqualTo": 1, "severity": "error"}
        results = _run_library_rule(rule, data)
        assert results[0].severity == "P0"

    def test_row_count_noop_when_no_operator(self) -> None:
        data = pa.table({"v": [1, 2]})
        rule = {"name": "rc", "metric": "rowCount"}
        results = _run_library_rule(rule, data)
        assert results[0].status == "PASSED"

    # ------------------------------------------------------------------
    # nullValues
    # ------------------------------------------------------------------

    def test_null_values_passes_when_zero_nulls(self) -> None:
        data = pa.table({"amount": pa.array([1.0, 2.0, 3.0])})
        rule = {"name": "no_nulls", "metric": "nullValues", "mustBe": 0}
        results = _run_library_rule(rule, data, property_name="amount")
        assert results[0].status == "PASSED"

    def test_null_values_fails_when_nulls_present(self) -> None:
        data = pa.table({"amount": pa.array([1.0, None, 3.0])})
        rule = {"name": "no_nulls", "metric": "nullValues", "mustBe": 0}
        results = _run_library_rule(rule, data, property_name="amount")
        assert results[0].status == "FAILED"

    def test_null_values_percent_mode(self) -> None:
        # 1 out of 4 nulls = 0.25; mustBeLessOrEqualTo 0.5 → passes
        data = pa.table({"amount": pa.array([1.0, None, 3.0, 4.0])})
        rule = {"name": "low_nulls", "metric": "nullValues", "unit": "percent", "mustBeLessOrEqualTo": 0.5}
        results = _run_library_rule(rule, data, property_name="amount")
        assert results[0].status == "PASSED"

    def test_null_values_percent_mode_fails(self) -> None:
        # 2 out of 4 nulls = 0.5; mustBeLessOrEqualTo 0.1 → fails
        data = pa.table({"amount": pa.array([1.0, None, None, 4.0])})
        rule = {"name": "low_nulls", "metric": "nullValues", "unit": "percent", "mustBeLessOrEqualTo": 0.1}
        results = _run_library_rule(rule, data, property_name="amount")
        assert results[0].status == "FAILED"

    # ------------------------------------------------------------------
    # missingValues
    # ------------------------------------------------------------------

    def test_missing_values_behaves_like_null_values(self) -> None:
        data = pa.table({"col": pa.array([1, None, 3])})
        rule = {"name": "no_missing", "metric": "missingValues", "mustBe": 0}
        results = _run_library_rule(rule, data, property_name="col")
        assert results[0].status == "FAILED"

    def test_missing_values_with_sentinel_list_warns_and_skips(self) -> None:
        data = pa.table({"col": pa.array(["a", "b"])})
        rule = {
            "name": "sentinel",
            "metric": "missingValues",
            "mustBe": 0,
            "arguments": {"missingValues": ["N/A", ""]},
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = _run_library_rule(rule, data, property_name="col")
        assert len(results) == 0
        assert any("missingValues" in str(w.message) for w in caught)

    # ------------------------------------------------------------------
    # invalidValues
    # ------------------------------------------------------------------

    def test_invalid_values_passes_when_all_valid(self) -> None:
        data = pa.table({"status": pa.array(["active", "inactive"])})
        rule = {
            "name": "valid_status",
            "metric": "invalidValues",
            "mustBe": 0,
            "validValues": ["active", "inactive", "pending"],
        }
        results = _run_library_rule(rule, data, property_name="status")
        assert results[0].status == "PASSED"

    def test_invalid_values_fails_when_invalid_present(self) -> None:
        data = pa.table({"status": pa.array(["active", "UNKNOWN"])})
        rule = {
            "name": "valid_status",
            "metric": "invalidValues",
            "mustBe": 0,
            "validValues": ["active", "inactive"],
        }
        results = _run_library_rule(rule, data, property_name="status")
        assert results[0].status == "FAILED"

    def test_invalid_values_with_pattern_warns_and_skips(self) -> None:
        data = pa.table({"iban": pa.array(["GB29NWBK60161331926819"])})
        rule = {
            "name": "iban_pattern",
            "metric": "invalidValues",
            "mustBe": 0,
            "arguments": {"pattern": "^[A-Z]{2}[0-9]{2}"},
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = _run_library_rule(rule, data, property_name="iban")
        assert len(results) == 0
        assert any("invalidValues" in str(w.message) for w in caught)

    # ------------------------------------------------------------------
    # duplicateValues — property level
    # ------------------------------------------------------------------

    def test_duplicate_values_property_passes_when_unique(self) -> None:
        data = pa.table({"id": pa.array([1, 2, 3])})
        rule = {"name": "unique_id", "metric": "duplicateValues", "mustBe": 0}
        results = _run_library_rule(rule, data, property_name="id")
        assert results[0].status == "PASSED"

    def test_duplicate_values_property_fails_when_dupes(self) -> None:
        data = pa.table({"id": pa.array([1, 1, 2])})
        rule = {"name": "unique_id", "metric": "duplicateValues", "mustBe": 0}
        results = _run_library_rule(rule, data, property_name="id")
        assert results[0].status == "FAILED"

    # ------------------------------------------------------------------
    # duplicateValues — table level (arguments.properties)
    # ------------------------------------------------------------------

    def test_duplicate_values_table_level_passes(self) -> None:
        data = pa.table({"a": pa.array([1, 2, 3]), "b": pa.array([4, 5, 6])})
        rule = {
            "name": "composite_unique",
            "metric": "duplicateValues",
            "mustBe": 0,
            "arguments": {"properties": ["a", "b"]},
        }
        results = _run_library_rule(rule, data, property_name=None)
        assert results[0].status == "PASSED"

    def test_duplicate_values_table_level_fails(self) -> None:
        data = pa.table({"a": pa.array([1, 1, 3]), "b": pa.array([4, 4, 6])})
        rule = {
            "name": "composite_unique",
            "metric": "duplicateValues",
            "mustBe": 0,
            "arguments": {"properties": ["a", "b"]},
        }
        results = _run_library_rule(rule, data, property_name=None)
        assert results[0].status == "FAILED"

    def test_invalid_values_percent_mode_passes(self) -> None:
        # 0 invalid out of 3 = 0.0 percent; mustBe 0 → passes
        data = pa.table({"status": pa.array(["active", "inactive", "active"])})
        rule = {
            "name": "valid_status_pct",
            "metric": "invalidValues",
            "mustBe": 0,
            "unit": "percent",
            "validValues": ["active", "inactive"],
        }
        results = _run_library_rule(rule, data, property_name="status")
        assert results[0].status == "PASSED"

    def test_duplicate_values_percent_mode_passes(self) -> None:
        # 0 duplicates out of 3 = 0.0; mustBe 0 → passes
        data = pa.table({"id": pa.array([1, 2, 3])})
        rule = {"name": "unique_id_pct", "metric": "duplicateValues", "mustBe": 0, "unit": "percent"}
        results = _run_library_rule(rule, data, property_name="id")
        assert results[0].status == "PASSED"

    def test_unknown_metric_produces_no_assertion(self) -> None:
        data = pa.table({"v": [1, 2]})
        rule = {"name": "unknown_metric_rule", "metric": "someUnknownMetric", "mustBe": 0}
        results = _run_library_rule(rule, data)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# _execute_custom_dqx_rule
# ---------------------------------------------------------------------------


def _run_custom_rule(
    rule: dict,
    data: pa.Table,
    dataset: str = "t",
) -> list[AssertionResult]:
    """Execute a single custom DQX rule via _execute_custom_dqx_rule."""
    from dqx.api import Context, check as dqx_check
    from dqx.provider import MetricProvider

    def check_fn(mp: MetricProvider, ctx: Context) -> None:
        _execute_custom_dqx_rule(rule, mp, ctx)

    decorated = dqx_check(name="test-check", datasets=[dataset])(check_fn)
    datasource = DuckRelationDataSource.from_arrow(data, dataset)
    db = InMemoryMetricDB()
    suite = VerificationSuite(checks=[decorated], db=db, name="test-suite")
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})
    suite.run([datasource], key)
    return suite.collect_results()


def _make_custom_rule(check_impl: str, operator_key: str, operator_val: object, name: str = "r") -> dict:
    """Build a minimal type:custom engine:dqx rule dict."""
    return {
        "name": name,
        "type": "custom",
        "engine": "dqx",
        "severity": "warning",
        operator_key: operator_val,
        "implementation": check_impl,
    }


class TestCustomDqxRuleExecution:
    """Integration tests for _execute_custom_dqx_rule()."""

    # ------------------------------------------------------------------
    # num_rows (table-level, no column:)
    # ------------------------------------------------------------------

    def test_num_rows_passes(self) -> None:
        data = pa.table({"v": [1, 2, 3]})
        rule = _make_custom_rule("check: num_rows\n", "mustBeGreaterOrEqualTo", 3)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_num_rows_fails(self) -> None:
        data = pa.table({"v": [1, 2, 3]})
        rule = _make_custom_rule("check: num_rows\n", "mustBeGreaterOrEqualTo", 10)
        assert _run_custom_rule(rule, data)[0].status == "FAILED"

    # ------------------------------------------------------------------
    # missing
    # ------------------------------------------------------------------

    def test_missing_passes_when_no_nulls(self) -> None:
        data = pa.table({"col": pa.array([1, 2, 3])})
        rule = _make_custom_rule("check: missing\ncolumn: col\n", "mustBe", 0)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_missing_pct_mode(self) -> None:
        # 1 null / 4 rows = 0.25; mustBeLessOrEqualTo 0.5 → passes
        data = pa.table({"col": pa.array([1, None, 3, 4])})
        rule = _make_custom_rule("check: missing\ncolumn: col\nreturn: pct\n", "mustBeLessOrEqualTo", 0.5)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    # ------------------------------------------------------------------
    # duplicates
    # ------------------------------------------------------------------

    def test_duplicates_passes_when_unique(self) -> None:
        data = pa.table({"id": pa.array([1, 2, 3])})
        rule = _make_custom_rule("check: duplicates\ncolumn: id\n", "mustBe", 0)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_duplicates_pct_mode(self) -> None:
        data = pa.table({"id": pa.array([1, 2, 3])})
        rule = _make_custom_rule("check: duplicates\ncolumn: id\nreturn: pct\n", "mustBe", 0)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    # ------------------------------------------------------------------
    # whitelist
    # ------------------------------------------------------------------

    def test_whitelist_passes_when_all_match(self) -> None:
        data = pa.table({"status": pa.array(["active", "inactive"])})
        rule = _make_custom_rule(
            "check: whitelist\ncolumn: status\nvalues:\n  - active\n  - inactive\n",
            "mustBeGreaterOrEqualTo",
            2,
        )
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_whitelist_pct_mode(self) -> None:
        data = pa.table({"status": pa.array(["active", "inactive", "active"])})
        rule = _make_custom_rule(
            "check: whitelist\ncolumn: status\nvalues:\n  - active\n  - inactive\nreturn: pct\n",
            "mustBeGreaterOrEqualTo",
            0.9,
        )
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    # ------------------------------------------------------------------
    # blacklist
    # ------------------------------------------------------------------

    def test_blacklist_passes_when_none_forbidden(self) -> None:
        data = pa.table({"username": pa.array(["alice", "bob"])})
        rule = _make_custom_rule(
            "check: blacklist\ncolumn: username\nvalues:\n  - admin\n  - root\n",
            "mustBeGreaterOrEqualTo",
            2,
        )
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_blacklist_pct_mode(self) -> None:
        data = pa.table({"username": pa.array(["alice", "bob", "alice"])})
        rule = _make_custom_rule(
            "check: blacklist\ncolumn: username\nvalues:\n  - admin\nreturn: pct\n",
            "mustBeGreaterOrEqualTo",
            0.9,
        )
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    # ------------------------------------------------------------------
    # cardinality
    # ------------------------------------------------------------------

    def test_cardinality_passes(self) -> None:
        data = pa.table({"status": pa.array(["a", "b", "c", "a"])})
        rule = _make_custom_rule("check: cardinality\ncolumn: status\n", "mustBeLessOrEqualTo", 5)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    # ------------------------------------------------------------------
    # min / max / mean / sum / count / variance
    # ------------------------------------------------------------------

    def test_min_passes(self) -> None:
        data = pa.table({"price": pa.array([5.0, 10.0, 15.0])})
        rule = _make_custom_rule("check: min\ncolumn: price\n", "mustBeGreaterOrEqualTo", 5.0)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_max_passes(self) -> None:
        data = pa.table({"price": pa.array([5.0, 10.0, 15.0])})
        rule = _make_custom_rule("check: max\ncolumn: price\n", "mustBeLessOrEqualTo", 15.0)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_mean_passes(self) -> None:
        data = pa.table({"price": pa.array([5.0, 10.0, 15.0])})
        rule = _make_custom_rule("check: mean\ncolumn: price\n", "mustBeBetween", [9.0, 11.0])
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_sum_passes(self) -> None:
        data = pa.table({"qty": pa.array([1, 2, 3])})
        rule = _make_custom_rule("check: sum\ncolumn: qty\n", "mustBe", 6)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_count_passes(self) -> None:
        # count = non-null values = 3 - 1 null = 2
        data = pa.table({"col": pa.array([1, None, 3])})
        rule = _make_custom_rule("check: count\ncolumn: col\n", "mustBe", 2)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_variance_passes(self) -> None:
        data = pa.table({"v": pa.array([2.0, 2.0, 2.0])})
        rule = _make_custom_rule("check: variance\ncolumn: v\n", "mustBe", 0)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    # ------------------------------------------------------------------
    # stddev (sqrt of variance)
    # ------------------------------------------------------------------

    def test_stddev_passes(self) -> None:
        data = pa.table({"v": pa.array([2.0, 2.0, 2.0])})
        rule = _make_custom_rule("check: stddev\ncolumn: v\n", "mustBe", 0)
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    # ------------------------------------------------------------------
    # min_length / max_length
    # ------------------------------------------------------------------

    def test_min_length_string_passes(self) -> None:
        data = pa.table({"name": pa.array(["abc", "defg"])})
        rule = _make_custom_rule(
            "check: min_length\ncolumn: name\ncolumn_type: string\n",
            "mustBeGreaterOrEqualTo",
            3,
        )
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    def test_max_length_string_passes(self) -> None:
        data = pa.table({"name": pa.array(["abc", "defg"])})
        rule = _make_custom_rule(
            "check: max_length\ncolumn: name\ncolumn_type: string\n",
            "mustBeLessOrEqualTo",
            4,
        )
        assert _run_custom_rule(rule, data)[0].status == "PASSED"

    # ------------------------------------------------------------------
    # NotImplementedError cases
    # ------------------------------------------------------------------

    def test_avg_length_raises_not_implemented(self) -> None:
        data = pa.table({"name": pa.array(["abc"])})
        rule = _make_custom_rule(
            "check: avg_length\ncolumn: name\ncolumn_type: string\n",
            "mustBeLessOrEqualTo",
            10,
        )
        with pytest.raises(NotImplementedError, match="avg_length"):
            _run_custom_rule(rule, data)

    def test_percentile_raises_not_implemented(self) -> None:
        data = pa.table({"v": pa.array([1.0, 2.0, 3.0])})
        rule = _make_custom_rule(
            "check: percentile\ncolumn: v\npercentile: 0.95\n",
            "mustBeLessOrEqualTo",
            3.0,
        )
        with pytest.raises(NotImplementedError, match="percentile"):
            _run_custom_rule(rule, data)

    def test_pattern_raises_not_implemented(self) -> None:
        data = pa.table({"email": pa.array(["a@b.com"])})
        rule = _make_custom_rule(
            "check: pattern\ncolumn: email\npattern: '^.+@.+$'\n",
            "mustBeGreaterOrEqualTo",
            1,
        )
        with pytest.raises(NotImplementedError, match="pattern"):
            _run_custom_rule(rule, data)

    # ------------------------------------------------------------------
    # Unknown check type → ContractValidationError
    # ------------------------------------------------------------------

    def test_unknown_check_type_raises_validation_error(self) -> None:
        data = pa.table({"v": [1]})
        rule = _make_custom_rule("check: some_unknown_check\ncolumn: v\n", "mustBe", 0)
        with pytest.raises(ContractValidationError, match="some_unknown_check"):
            _run_custom_rule(rule, data)

    # ------------------------------------------------------------------
    # Assertion name uses rule name field
    # ------------------------------------------------------------------

    def test_assertion_name_uses_rule_name(self) -> None:
        data = pa.table({"v": [1, 2, 3]})
        rule = _make_custom_rule("check: num_rows\n", "mustBeGreaterOrEqualTo", 1, name="my_volume_check")
        results = _run_custom_rule(rule, data)
        assert results[0].assertion == "my_volume_check"


# ---------------------------------------------------------------------------
# Contract.to_checks() — public method
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "contracts"


def _odcs_contract(tmp_path: Path, content: str) -> list[Contract]:
    """Write a minimal valid ODCS YAML and parse it."""
    path = _write_yaml(tmp_path, "contract.odcs.yaml", content)
    return Contract.from_odcs(path)


def _run_contract(contract: Contract, data: pa.Table) -> list[AssertionResult]:
    """Run contract.to_checks() against data and collect results."""
    checks = contract.to_checks()
    datasource = DuckRelationDataSource.from_arrow(data, contract.dataset)
    db = InMemoryMetricDB()
    suite = VerificationSuite(checks=checks, db=db, name="test-suite")
    key = ResultKey(yyyy_mm_dd=datetime.date(2024, 1, 1), tags={})
    suite.run([datasource], key)
    return suite.collect_results()


class TestToChecksPublicMethod:
    """Tests for Contract.to_checks()."""

    _MINIMAL_ODCS = """\
        apiVersion: v3.1.0
        kind: DataContract
        id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
        version: "1.0.0"
        status: active
        schema:
          - name: orders
    """

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def test_returns_list_of_one_decorated_check(self, tmp_path: Path) -> None:
        contracts = _odcs_contract(tmp_path, self._MINIMAL_ODCS)
        checks = contracts[0].to_checks()
        assert isinstance(checks, list)
        assert len(checks) == 1

    def test_check_name_uses_contract_name_when_present(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            name: my_contract
            schema:
              - name: orders
        """
        contracts = _odcs_contract(tmp_path, content)
        checks = contracts[0].to_checks()
        assert checks[0].__name__ == "Contract: my_contract"

    def test_check_name_uses_dataset_when_name_absent(self, tmp_path: Path) -> None:
        contracts = _odcs_contract(tmp_path, self._MINIMAL_ODCS)
        checks = contracts[0].to_checks()
        assert checks[0].__name__ == "Contract: orders"

    def test_check_name_uses_physical_name_when_present(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
                physicalName: orders_tbl
        """
        contracts = _odcs_contract(tmp_path, content)
        checks = contracts[0].to_checks()
        assert checks[0].__name__ == "Contract: orders_tbl"

    def test_all_assertions_share_one_check_node(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
                quality:
                  - name: rc
                    metric: rowCount
                    mustBeGreaterOrEqualTo: 1
                properties:
                  - name: order_id
                    logicalType: integer
                    quality:
                      - name: no_nulls
                        metric: nullValues
                        mustBe: 0
        """
        contracts = _odcs_contract(tmp_path, content)
        data = pa.table({"order_id": pa.array([1, 2, 3])})
        results = _run_contract(contracts[0], data)
        assert len(results) == 2
        check_names = {r.check for r in results}
        assert len(check_names) == 1, "All assertions must share one check node"

    # ------------------------------------------------------------------
    # Rule dispatch
    # ------------------------------------------------------------------

    def test_text_rules_produce_no_assertions(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
                properties:
                  - name: order_id
                    quality:
                      - name: doc_rule
                        type: text
                        description: "Documentation only"
        """
        contracts = _odcs_contract(tmp_path, content)
        data = pa.table({"order_id": pa.array([1, 2])})
        results = _run_contract(contracts[0], data)
        assert len(results) == 0

    def test_sql_rule_warns_and_skips(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
                quality:
                  - name: fraud_check
                    type: sql
                    query: "SELECT 1"
                    mustBe: 0
        """
        contracts = _odcs_contract(tmp_path, content)
        data = pa.table({"v": [1]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = _run_contract(contracts[0], data)
        assert len(results) == 0
        assert any("sql" in str(w.message).lower() for w in caught)

    def test_custom_non_dqx_engine_warns_and_skips(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
                quality:
                  - name: soda_rule
                    type: custom
                    engine: soda
                    implementation: "missing_count(x) = 0"
        """
        contracts = _odcs_contract(tmp_path, content)
        data = pa.table({"v": [1]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = _run_contract(contracts[0], data)
        assert len(results) == 0
        assert any("soda" in str(w.message) for w in caught)

    def test_empty_schema_produces_no_assertions(self, tmp_path: Path) -> None:
        contracts = _odcs_contract(tmp_path, self._MINIMAL_ODCS)
        data = pa.table({"v": [1]})
        results = _run_contract(contracts[0], data)
        assert len(results) == 0

    def test_custom_dqx_rule_executes_via_to_checks(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
                quality:
                  - name: volume_floor
                    type: custom
                    engine: dqx
                    severity: warning
                    implementation: |
                      check: num_rows
                      mustBeGreaterOrEqualTo: 1
        """
        contracts = _odcs_contract(tmp_path, content)
        data = pa.table({"v": [1, 2, 3]})
        results = _run_contract(contracts[0], data)
        assert len(results) == 1
        assert results[0].status == "PASSED"

    # ------------------------------------------------------------------
    # SLA raises NotImplementedError at call time
    # ------------------------------------------------------------------

    def test_sla_latency_raises_not_implemented_at_call_time(self, tmp_path: Path) -> None:
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            slaProperties:
              - property: latency
                value: 24
                unit: h
                element: orders.created_at
            schema:
              - name: orders
        """
        contracts = _odcs_contract(tmp_path, content)
        with pytest.raises(NotImplementedError, match="freshness"):
            contracts[0].to_checks()

    def test_no_sla_latency_does_not_raise(self, tmp_path: Path) -> None:
        contracts = _odcs_contract(tmp_path, self._MINIMAL_ODCS)
        checks = contracts[0].to_checks()
        assert len(checks) == 1

    # ------------------------------------------------------------------
    # Integration: full-example.odcs.yaml fixture
    # ------------------------------------------------------------------

    def test_full_odcs_example_executes_tbl_rules(self, tmp_path: Path) -> None:
        """End-to-end: rowCount and nullValues rules execute correctly."""
        content = """\
            apiVersion: v3.1.0
            kind: DataContract
            id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
            version: "1.0.0"
            status: active
            schema:
              - name: orders
                quality:
                  - name: row_count_check
                    type: library
                    metric: rowCount
                    mustBeGreaterThan: 1000000
                    severity: error
                properties:
                  - name: rcvr_cntry_code
                    logicalType: string
                    quality:
                      - name: no_nulls
                        type: library
                        metric: nullValues
                        mustBe: 0
                        severity: error
        """
        contracts = _odcs_contract(tmp_path, content)
        data = pa.table({"rcvr_cntry_code": pa.array(["US", "GB"])})
        results = _run_contract(contracts[0], data)
        # rowCount mustBeGreaterThan 1000000 → FAILED (only 2 rows)
        # nullValues mustBe 0 → PASSED (no nulls)
        assert len(results) == 2
        by_name = {r.assertion: r.status for r in results}
        assert by_name["no_nulls"] == "PASSED"
        assert by_name["row_count_check"] == "FAILED"

    # ------------------------------------------------------------------
    # Public API export from dqx package
    # ------------------------------------------------------------------

    def test_contract_exported_from_dqx(self) -> None:
        assert hasattr(dqx, "Contract")

    def test_contract_validation_error_exported_from_dqx(self) -> None:
        assert hasattr(dqx, "ContractValidationError")

    def test_schema_validation_error_exported_from_dqx(self) -> None:
        assert hasattr(dqx, "SchemaValidationError")

    def test_contract_warning_exported_from_dqx(self) -> None:
        assert hasattr(dqx, "ContractWarning")
