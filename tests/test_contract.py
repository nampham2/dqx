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
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import pytest
import yaml

from dqx.contract import (
    Contract,
    ContractProperty,
    ContractSchema,
    ContractValidationError,
    ContractWarning,
    SlaLatency,
)


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
        """value=4, unit=d → 4 × 24 = 96 hours."""
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
