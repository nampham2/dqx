"""DQL (Data Quality Language) parser.

This module provides parsing of DQL files for data quality checks.
DQL programs are executed via VerificationSuite with the dql parameter.
"""

from dqx.dql.ast import (
    Annotation,
    Assertion,
    Check,
    Collection,
    Expr,
    Severity,
    SourceLocation,
    Suite,
    Tunable,
)
from dqx.dql.errors import DQLError, DQLSyntaxError
from dqx.dql.parser import parse, parse_file

__all__ = [
    # AST nodes
    "Suite",
    "Check",
    "Assertion",
    "Collection",
    "Annotation",
    "Tunable",
    "Expr",
    "Severity",
    "SourceLocation",
    # Parser
    "parse",
    "parse_file",
    # Errors
    "DQLError",
    "DQLSyntaxError",
]
