"""Validation system for DQX verification suites.

This module provides validators that check for common configuration errors
in verification suites before they run, catching issues like duplicate names
and empty checks early.

Available Validators:
    - DuplicateCheckNameValidator: Detects duplicate check names (error)
    - EmptyCheckValidator: Detects checks with no assertions (warning)
    - DuplicateAssertionNameValidator: Detects duplicate assertion names within checks (error)
    - DatasetValidator: Detects dataset mismatches between checks and symbols (error)
    - UnusedSymbolValidator: Detects symbols that are defined but never used (warning)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import sympy as sp

from dqx.graph.base import BaseNode
from dqx.graph.nodes import AssertionNode, CheckNode
from dqx.graph.traversal import Graph

if TYPE_CHECKING:
    from dqx.provider import MetricProvider


logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in the graph."""

    rule: str  # Name of the rule that found this issue
    message: str  # Human-readable description
    node_path: list[str]  # Path to the problematic node


class ValidationReport:
    """Collects validation issues and formats them for display."""

    def __init__(self) -> None:
        """Initialize empty report."""
        self._errors: list[ValidationIssue] = []
        self._warnings: list[ValidationIssue] = []

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get all error issues."""
        return self._errors

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get all warning issues."""
        return self._warnings

    def add_error(self, issue: ValidationIssue) -> None:
        """Add an error issue to the report."""
        self._errors.append(issue)

    def add_warning(self, issue: ValidationIssue) -> None:
        """Add a warning issue to the report."""
        self._warnings.append(issue)

    def has_errors(self) -> bool:
        """Check if report contains any errors."""
        return len(self._errors) > 0

    def has_warnings(self) -> bool:
        """Check if report contains any warnings."""
        return len(self._warnings) > 0

    def to_dict(self) -> dict[str, Any]:
        """Export report as structured data for programmatic consumption."""
        return {
            "errors": [
                {"rule": issue.rule, "message": issue.message, "node_path": issue.node_path} for issue in self._errors
            ],
            "warnings": [
                {"rule": issue.rule, "message": issue.message, "node_path": issue.node_path} for issue in self._warnings
            ],
            "summary": {
                "error_count": len(self._errors),
                "warning_count": len(self._warnings),
                "has_issues": self.has_errors() or self.has_warnings(),
            },
        }

    def __str__(self) -> str:
        """Format report as human-readable string."""
        if not self._errors and not self._warnings:
            return "No validation issues found."

        lines = []

        if self._errors:
            lines.append(f"\n{len(self._errors)} ERROR(S):")
            for issue in self._errors:
                lines.append(f"  [{issue.rule}] {issue.message}")
                lines.append(f"    Path: {' > '.join(issue.node_path)}")

        if self._warnings:
            lines.append(f"\n{len(self._warnings)} WARNING(S):")
            for issue in self._warnings:
                lines.append(f"  [{issue.rule}] {issue.message}")
                lines.append(f"    Path: {' > '.join(issue.node_path)}")

        return "\n".join(lines)


class BaseValidator(ABC):
    """Base class for all validators."""

    name: str  # Must be defined by subclasses
    is_error: bool  # Must be defined by subclasses

    def __init__(self) -> None:
        """Initialize validator."""
        self._issues: list[ValidationIssue] = []

    @abstractmethod
    def process_node(self, node: BaseNode) -> None:
        """Process a node and potentially add issues.

        Args:
            node: The node to validate
        """
        pass

    def get_issues(self) -> list[ValidationIssue]:
        """Get all issues found by this validator."""
        return self._issues

    def finalize(self) -> None:
        """Post-processing after all nodes have been processed."""
        pass


class DuplicateCheckNameValidator(BaseValidator):
    """Detects duplicate check names in the suite."""

    name = "duplicate_check_names"
    is_error = True

    def __init__(self) -> None:
        """Initialize validator."""
        super().__init__()
        self._check_names: dict[str, list[CheckNode]] = defaultdict(list)

    def process_node(self, node: BaseNode) -> None:
        """Collect check names for duplicate detection."""
        if isinstance(node, CheckNode):
            self._check_names[node.name].append(node)

    def finalize(self) -> None:
        """Process collected data and generate issues."""
        for name, nodes in self._check_names.items():
            if len(nodes) > 1:
                self._issues.append(
                    ValidationIssue(
                        rule=self.name,
                        message=f"Duplicate check name: '{name}' appears {len(nodes)} times",
                        node_path=["root", f"check:{name}"],
                    )
                )


class EmptyCheckValidator(BaseValidator):
    """Detects checks with no assertions."""

    name = "empty_checks"
    is_error = False  # This produces warnings

    def process_node(self, node: BaseNode) -> None:
        """Check if a check node has no children."""
        if isinstance(node, CheckNode) and len(node.children) == 0:
            self._issues.append(
                ValidationIssue(
                    rule=self.name,
                    message=f"Check '{node.name}' has no assertions",
                    node_path=["root", f"check:{node.name}"],
                )
            )


class DuplicateAssertionNameValidator(BaseValidator):
    """Detects duplicate assertion names within each check."""

    name = "duplicate_assertion_names"
    is_error = True

    def process_node(self, node: BaseNode) -> None:
        """Check for duplicate assertion names within a check."""
        if isinstance(node, CheckNode):
            assertion_names: dict[str, int] = defaultdict(int)

            # Count assertion names
            for child in node.children:
                if isinstance(child, AssertionNode) and child.name:
                    assertion_names[child.name] += 1

            # Report duplicates
            for name, count in assertion_names.items():
                if count > 1:
                    self._issues.append(
                        ValidationIssue(
                            rule=self.name,
                            message=(f"Assertion name '{name}' appears {count} times in check '{node.name}'"),
                            node_path=["root", f"check:{node.name}", f"assertion:{name}"],
                        )
                    )


class DatasetValidator(BaseValidator):
    """Detects dataset mismatches between CheckNodes and their AssertionNodes' symbols."""

    name = "dataset_mismatch"
    is_error = True

    def __init__(self, provider: "MetricProvider") -> None:
        """Initialize validator with provider."""
        super().__init__()
        self._provider = provider

    def process_node(self, node: BaseNode) -> None:
        """Process a node to check for dataset mismatches and ambiguities."""
        if not isinstance(node, AssertionNode):
            return

        parent_check = node.parent

        # Only validate if parent check has datasets specified
        if not parent_check.datasets:
            return

        parent_datasets = parent_check.datasets

        # Extract symbols from assertion expression
        symbols = node.actual.free_symbols

        for symbol in symbols:
            try:
                metric = self._provider.get_symbol(symbol)
            except Exception:
                # Symbol not found in provider - skip validation for this symbol
                continue

            if metric.dataset is None:
                # If check has multiple datasets, this is ambiguous
                if len(parent_datasets) > 1:
                    self._issues.append(
                        ValidationIssue(
                            rule=self.name,
                            message=(
                                f"Symbol '{metric.name}' in assertion '{node.name}' "
                                f"has no dataset specified, but parent check '{parent_check.name}' "
                                f"has multiple datasets: {parent_datasets}. Unable to determine which dataset to use."
                            ),
                            node_path=["root", f"check:{parent_check.name}", f"assertion:{node.name}"],
                        )
                    )
                # If check has exactly one dataset, imputation will handle it
                continue

            # Validate symbol's dataset is in parent's datasets
            if metric.dataset not in parent_datasets:
                self._issues.append(
                    ValidationIssue(
                        rule=self.name,
                        message=(
                            f"Symbol '{metric.name}' in assertion '{node.name}' "
                            f"has dataset '{metric.dataset}' which is not in "
                            f"parent check '{parent_check.name}' datasets: {parent_datasets}"
                        ),
                        node_path=["root", f"check:{parent_check.name}", f"assertion:{node.name}"],
                    )
                )

            # Check required_metrics for dataset consistency
            for required_symbol in metric.required_metrics:
                try:
                    required_metric = self._provider.get_symbol(required_symbol)
                except Exception:
                    # Required symbol not found - skip validation
                    continue

                # Both metric and required metric have datasets specified
                if metric.dataset and required_metric.dataset and metric.dataset != required_metric.dataset:
                    self._issues.append(
                        ValidationIssue(
                            rule=self.name,
                            message=(
                                f"Required metric '{required_metric.name}' has dataset '{required_metric.dataset}' "
                                f"but metric '{metric.name}' has dataset '{metric.dataset}'. "
                                f"Required metrics must use the same dataset as their parent."
                            ),
                            node_path=["root", f"check:{parent_check.name}", f"assertion:{node.name}"],
                        )
                    )


class UnusedSymbolValidator(BaseValidator):
    """Detects symbols that are defined but not used in any assertion."""

    name = "unused_symbols"
    is_error = False  # This produces warnings

    def __init__(self, provider: "MetricProvider") -> None:
        """Initialize validator with provider."""
        super().__init__()
        self._provider = provider
        self._used_symbols: set[sp.Symbol] = set()
        self._removed_symbols: list[str] = []  # Store removed symbol names

    def process_node(self, node: BaseNode) -> None:
        """Collect symbols used in assertions."""
        if isinstance(node, AssertionNode):
            # Extract all symbols from the assertion expression
            self._used_symbols.update(node.actual.free_symbols)

    @property
    def removed_symbols(self) -> list[str]:
        """Get the list of removed symbol names."""
        return self._removed_symbols

    def finalize(self) -> None:
        """Compare defined vs used symbols and generate warnings."""
        # Get all defined symbols from provider
        defined_symbols = set(self._provider.symbols())

        # Find unused symbols
        unused_symbols = defined_symbols - self._used_symbols

        # Sort unused symbols by numeric index (x_4 before x_14)
        sorted_unused = sorted(unused_symbols, key=lambda s: int(str(s).split("_")[1]))

        # Generate warnings for each unused symbol, excluding dependencies
        for symbol in sorted_unused:
            # Skip symbols that are required by other metrics
            is_required = False
            for other_metric in self._provider.metrics:
                if symbol in other_metric.required_metrics:
                    is_required = True
                    break

            if is_required:
                continue

            # Format: symbol_name ← metric_name
            symbol_repr = f"{symbol} ← {self._provider.get_symbol(symbol).name}"

            self._issues.append(
                ValidationIssue(
                    rule=self.name,
                    message=f"Unused symbol: {symbol_repr}",
                    node_path=["root", symbol_repr],
                )
            )

            # Remove the orphaned symbol from the provider to keep things clean
            self._removed_symbols.append(str(symbol))
            self._provider.remove_symbol(symbol)


class CompositeValidationVisitor:
    """Runs multiple validators in a single graph traversal for performance."""

    def __init__(self, validators: list[BaseValidator]) -> None:
        """Initialize with a list of validators to run.

        Args:
            validators: List of validator instances to run during traversal
        """
        self._validators = validators
        self._nodes: list[BaseNode] = []

    def visit(self, node: BaseNode) -> Any:
        """Visit a node and collect it for processing.

        This method is called by the graph traversal. We collect nodes
        here and process them later in get_all_issues.
        """
        self._nodes.append(node)

    def get_all_issues(self) -> dict[str, list[ValidationIssue]]:
        """Get all issues from all validators after traversal.

        Returns:
            Dict with 'errors' and 'warnings' lists
        """
        # Process all nodes
        for node in self._nodes:
            for validator in self._validators:
                validator.process_node(node)

        # Run finalize on validators that have it
        for validator in self._validators:
            validator.finalize()

        # Collect issues by type
        errors = []
        warnings = []

        for validator in self._validators:
            issues = validator.get_issues()
            if validator.is_error:
                errors.extend(issues)
            else:
                warnings.extend(issues)

        return {"errors": errors, "warnings": warnings}

    async def visit_async(self, node: BaseNode) -> None:
        """Async visit method required by NodeVisitor protocol.

        Since validation is synchronous, this just delegates to visit.
        """
        self.visit(node)


class SuiteValidator:
    """Main validator that runs all validation rules efficiently."""

    def validators(self, provider: MetricProvider) -> list[BaseValidator]:
        return [
            DuplicateCheckNameValidator(),
            EmptyCheckValidator(),
            DuplicateAssertionNameValidator(),
            DatasetValidator(provider),
            UnusedSymbolValidator(provider),
        ]

    def validate(self, graph: Graph, provider: MetricProvider) -> ValidationReport:
        """Run validation on a graph.

        Args:
            graph: The graph to validate
            provider: MetricProvider for dataset validation (required)

        Returns:
            ValidationReport with all issues found
        """
        # Create validators
        validators = self.validators(provider)

        # Create composite with all validators
        composite = CompositeValidationVisitor(validators)

        # Single-pass traversal
        graph.bfs(composite)

        # Get all issues
        issues = composite.get_all_issues()

        # Build report
        report = ValidationReport()
        for error in issues["errors"]:
            report.add_error(error)
        for warning in issues["warnings"]:
            report.add_warning(warning)

        # Get removed symbols from UnusedSymbolValidator
        removed_symbols = []
        for validator in validators:
            if isinstance(validator, UnusedSymbolValidator):
                removed_symbols = validator.removed_symbols
                break

        # Log removed symbols after validation warnings
        if removed_symbols:
            # Sort symbols by numeric index (x_9 before x_14)
            sorted_symbols = sorted(removed_symbols, key=lambda s: int(s.split("_")[1]))
            logger.info("Removed %d unused symbols: %s", len(sorted_symbols), ", ".join(sorted_symbols))

        return report
