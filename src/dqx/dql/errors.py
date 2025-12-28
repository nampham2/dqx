"""DQL error classes with source location support."""

from enum import StrEnum

from dqx.dql.ast import SourceLocation


class DQLError(Exception):
    """Base class for DQL errors."""

    def __init__(self, message: str, loc: SourceLocation | None = None):
        self.message = message
        self.loc = loc
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.loc:
            return f"{self.loc}: {self.message}"
        return self.message


class DQLSyntaxError(DQLError):
    """Syntax error during parsing."""

    def __init__(
        self,
        message: str,
        loc: SourceLocation | None = None,
        source_line: str | None = None,
        suggestion: str | None = None,
    ):
        self.source_line = source_line
        self.suggestion = suggestion
        super().__init__(message, loc)

    def _format_message(self) -> str:
        parts = []

        # Error header
        if self.loc:
            parts.append(f"error: {self.message}")
            parts.append(f"  --> {self.loc}")
        else:
            parts.append(f"error: {self.message}")

        # Source context
        if self.source_line and self.loc:
            parts.append("   |")
            parts.append(f"{self.loc.line:>3} | {self.source_line}")
            if self.loc.column > 0:
                # column is 1-based, so use (column - 1) + 5 = column + 4
                pointer = " " * (self.loc.column + 4) + "^"
                parts.append(f"   | {pointer}")

        # Suggestion
        if self.suggestion:
            parts.append(f"   = help: {self.suggestion}")

        return "\n".join(parts)


class DQLSemanticError(DQLError):
    """Semantic error (e.g., undefined constant, duplicate name)."""

    pass


class DQLImportError(DQLError):
    """Import resolution error."""

    pass


class ErrorCode(StrEnum):
    """Standard error codes for DQL errors."""

    E001 = "E001"  # Unknown metric
    E002 = "E002"  # Duplicate assertion name
    E003 = "E003"  # Undefined constant
    E004 = "E004"  # Invalid severity
    E005 = "E005"  # Circular import
    E006 = "E006"  # Import not found
    W001 = "W001"  # Assertion has no name (warning)
