"""Suite validator that orchestrates all validation rules."""

from __future__ import annotations

import logging

from dqx.graph.traversal import Graph
from dqx.provider import MetricProvider

from .common import (
    BaseValidator,
    CompositeValidationVisitor,
    DatasetValidator,
    DuplicateAssertionNameValidator,
    DuplicateCheckNameValidator,
    EmptyCheckValidator,
    UnusedSymbolValidator,
    ValidationReport,
)

logger = logging.getLogger(__name__)


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
