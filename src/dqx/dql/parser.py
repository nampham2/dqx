"""DQL parser implementation using Lark.

Parses DQL source into AST nodes.
"""

from datetime import date
from pathlib import Path
from typing import Any

from lark import Lark, Transformer, v_args
from lark.exceptions import UnexpectedCharacters, UnexpectedToken

from dqx.dql.ast import (
    Annotation,
    Assertion,
    Check,
    Const,
    DateExpr,
    DisableRule,
    Expr,
    Import,
    Profile,
    Sample,
    ScaleRule,
    SetSeverityRule,
    Severity,
    SourceLocation,
    Suite,
)
from dqx.dql.errors import DQLSyntaxError

# Load grammar from file
_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
if not _GRAMMAR_PATH.exists():
    raise FileNotFoundError(f"DQL grammar file not found: {_GRAMMAR_PATH}")
_GRAMMAR = _GRAMMAR_PATH.read_text(encoding="utf-8")

# Create parser instance (cached)
_parser = Lark(
    _GRAMMAR,
    start="start",
    parser="lalr",
    propagate_positions=True,
)


def _make_loc(tree: Any, filename: str | None = None) -> SourceLocation | None:
    """Extract source location from a Lark tree node."""
    if hasattr(tree, "meta") and tree.meta:
        return SourceLocation(
            line=tree.meta.line,
            column=tree.meta.column,
            end_line=tree.meta.end_line,
            end_column=tree.meta.end_column,
            filename=filename,
        )
    return None


def _parse_string(s: str) -> str:
    """Parse a DQL string literal, handling escape sequences."""
    # Remove surrounding quotes
    s = s[1:-1]
    # Handle escape sequences
    s = s.replace('\\"', '"')
    s = s.replace("\\\\", "\\")
    s = s.replace("\\n", "\n")
    s = s.replace("\\r", "\r")
    s = s.replace("\\t", "\t")
    return s


def _parse_number(s: str) -> float | int:
    """Parse a number, returning int if no decimal point."""
    if "." in s:
        return float(s)
    return int(s)


def _parse_percent(s: str) -> float:
    """Parse a percentage like '5%' to decimal 0.05."""
    return float(s.rstrip("%")) / 100


class DQLTransformer(Transformer):
    """Transform Lark parse tree into DQL AST nodes."""

    def __init__(self, filename: str | None = None):
        super().__init__()
        self.filename = filename
        self._pending_annotations: list[Annotation] = []

    def _loc(self, tree: Any) -> SourceLocation | None:
        return _make_loc(tree, self.filename)

    # === Tokens ===

    def STRING(self, token: Any) -> str:
        return _parse_string(str(token))

    def NUMBER(self, token: Any) -> float | int:
        return _parse_number(str(token))

    def PERCENT(self, token: Any) -> float:
        return _parse_percent(str(token))

    def DATE(self, token: Any) -> date:
        return date.fromisoformat(str(token))

    def SEVERITY(self, token: Any) -> Severity:
        return Severity(str(token))

    def IDENT(self, token: Any) -> str:
        return str(token)

    def ESCAPED_IDENT(self, token: Any) -> str:
        # Remove backticks
        return str(token)[1:-1]

    def COMP_OP(self, token: Any) -> str:
        return str(token)

    # === Identifiers ===

    def ident(self, items: list) -> str:
        return items[0]

    def qualified_ident(self, items: list) -> str:
        return ".".join(items)

    # === Expressions ===

    def expr(self, items: list) -> Expr:
        # Reconstruct expression as string for sympy
        # items alternates: term, op, term, op, term...
        if len(items) == 1:
            return items[0] if isinstance(items[0], Expr) else Expr(text=str(items[0]))

        parts = []
        for item in items:
            if isinstance(item, Expr):
                parts.append(item.text)
            elif hasattr(item, "value"):  # Token (operator)
                parts.append(str(item))
            else:  # pragma: no cover - defensive fallback
                parts.append(str(item))
        text = " ".join(parts)
        loc = items[0].loc if isinstance(items[0], Expr) else None
        return Expr(text=text, loc=loc)

    def term(self, items: list) -> Expr:
        # items alternates: factor, op, factor, op, factor...
        if len(items) == 1:
            return items[0] if isinstance(items[0], Expr) else Expr(text=str(items[0]))

        parts = []
        for item in items:
            if isinstance(item, Expr):
                parts.append(item.text)
            elif hasattr(item, "value"):  # Token (operator)
                parts.append(str(item))
            else:  # pragma: no cover - defensive fallback
                parts.append(str(item))
        return Expr(text=" ".join(parts))

    def factor(self, items: list) -> Expr:
        item = items[0]
        if isinstance(item, Expr):
            return item
        if isinstance(item, (int, float)):
            return Expr(text=str(item))
        # String from ident or other token
        return Expr(text=str(item))

    def neg(self, items: list) -> Expr:
        return Expr(text=f"-{items[0].text}")

    def none_literal(self, items: list) -> Expr:
        return Expr(text="None")

    def call(self, items: list) -> Expr:
        func_name = items[0]
        if len(items) > 1 and items[1] is not None:
            args = items[1]
            return Expr(text=f"{func_name}({args})")
        return Expr(text=f"{func_name}()")

    def args(self, items: list) -> str:
        parts = []
        for item in items:
            if isinstance(item, Expr):
                parts.append(item.text)
            elif isinstance(item, str):
                # Named arg string like "lag 1"
                parts.append(item)
            elif isinstance(item, list):
                # List arg
                parts.append(f"[{', '.join(item)}]")
            else:  # pragma: no cover - defensive fallback
                parts.append(str(item))
        return ", ".join(parts)

    def arg(self, items: list) -> Expr | str | list:
        return items[0]

    def named_arg(self, items: list) -> str:  # pragma: no cover - grammar routes to specific handlers
        return items[0]

    def arg_lag(self, items: list) -> str:
        return f"lag={items[0]}"

    def arg_dataset(self, items: list) -> str:
        return f"dataset={items[0]}"

    def arg_order_by(self, items: list) -> str:
        return f"order_by={items[0]}"

    def arg_n(self, items: list) -> str:
        return f"n={items[0]}"

    def list_arg(self, items: list) -> list:
        return list(items)

    # === Conditions ===

    def condition(self, items: list) -> tuple[str, Expr | None, Expr | None, str | None]:
        # Returns (op, threshold, threshold_upper, keyword)
        # All condition types return tuples, so this just passes through
        return items[0]

    def comparison(self, items: list) -> tuple[str, Expr | None, Expr | None, str | None]:
        op = items[0]
        threshold = items[1]
        return (op, threshold, None, None)

    def between_term(self, items: list) -> Expr:
        item = items[0]
        if isinstance(item, Expr):
            return item
        # NUMBER/PERCENT tokens are converted to int/float, ident to str
        return Expr(text=str(item))

    def between_term_neg(self, items: list) -> Expr:
        # Handle negative numbers in between bounds: "-" NUMBER or "-" PERCENT
        return Expr(text=f"-{items[0]}")

    def between_bound(self, items: list) -> Expr:
        # Similar to term - handles between_term (MUL_OP between_term)*
        if len(items) == 1:
            return items[0] if isinstance(items[0], Expr) else Expr(text=str(items[0]))
        parts = []
        for item in items:
            if isinstance(item, Expr):
                parts.append(item.text)
            else:  # Token (MUL_OP)
                parts.append(str(item))
        return Expr(text=" ".join(parts))

    def condition_between(self, items: list) -> tuple[str, Expr | None, Expr | None, str | None]:
        # items may contain AND token between bounds, filter it out
        bounds = [x for x in items if isinstance(x, Expr)]
        return ("between", bounds[0], bounds[1], None)

    def condition_is(self, items: list) -> tuple[str, Expr | None, Expr | None, str | None]:
        keyword = items[0]
        return ("is", None, None, keyword)

    def kw_positive(self, items: list) -> str:
        return "positive"

    def kw_negative(self, items: list) -> str:
        return "negative"

    def kw_none(self, items: list) -> str:
        return "None"

    def kw_not_none(self, items: list) -> str:
        return "not None"

    # === Modifiers ===

    def modifiers(self, items: list) -> dict:
        return items[0]

    def name_mod(self, items: list) -> dict:
        return {"name": items[0]}

    def tolerance_mod(self, items: list) -> dict:
        # items[0] is the TOLERANCE_KW token, items[1] is the number
        return {"tolerance": items[1]}

    def severity_mod(self, items: list) -> dict:
        return {"severity": items[0]}

    def tags_mod(self, items: list) -> dict:
        return {"tags": tuple(items)}

    def sample_mod(self, items: list) -> dict:
        sample = items[0]
        if len(items) > 1:
            sample = Sample(
                value=sample.value,
                is_percentage=sample.is_percentage,
                seed=items[1],
            )
        return {"sample": sample}

    def sample_value(self, items: list) -> Sample:  # pragma: no cover - grammar routes to specific handlers
        return items[0]

    def sample_percent(self, items: list) -> Sample:
        return Sample(value=items[0], is_percentage=True)

    def sample_rows(self, items: list) -> Sample:
        return Sample(value=items[0], is_percentage=False)

    # === Annotations ===

    def annotation(self, items: list) -> Annotation:
        name = items[0]
        args = {}
        if len(items) > 1 and items[1] is not None:
            args = items[1]
        ann = Annotation(name=name, args=args)
        self._pending_annotations.append(ann)
        return ann

    def ann_args(self, items: list) -> dict:
        result = {}
        for item in items:
            if isinstance(item, tuple):
                result[item[0]] = item[1]
        return result

    def ann_arg(self, items: list) -> tuple:
        return (items[0], items[1])

    # === Assertions ===

    @v_args(tree=True)
    def annotated_assertion(self, tree: Any) -> Assertion:
        items = tree.children

        # Collect annotations that were processed before this assertion
        annotations = tuple(self._pending_annotations)
        self._pending_annotations = []

        # Find the expr (first Expr after any annotations)
        expr_idx = 0
        for i, item in enumerate(items):
            if isinstance(item, Expr):
                expr_idx = i
                break

        expr = items[expr_idx]

        # Find condition
        cond_idx = expr_idx + 1
        condition_data = items[cond_idx]
        op, threshold, threshold_upper, keyword = condition_data

        # Collect modifiers
        modifiers: dict[str, Any] = {}
        for item in items[cond_idx + 1 :]:
            if isinstance(item, dict):
                modifiers.update(item)

        return Assertion(
            expr=expr,
            condition=op,
            threshold=threshold,
            threshold_upper=threshold_upper,
            keyword=keyword,
            name=modifiers.get("name"),
            severity=modifiers.get("severity", Severity.P1),
            tolerance=modifiers.get("tolerance"),
            tags=modifiers.get("tags", ()),
            sample=modifiers.get("sample"),
            annotations=annotations,
            loc=self._loc(tree),
        )

    # === Checks ===

    def datasets(self, items: list) -> tuple[str, ...]:
        return tuple(items)

    def check_body(self, items: list) -> list[Assertion]:
        # All items should be Assertions now (annotated_assertion rule)
        return list(items)

    @v_args(tree=True)
    def check(self, tree: Any) -> Check:
        items = tree.children
        name = items[0]
        datasets = items[1]
        assertions = tuple(items[2])
        return Check(
            name=name,
            datasets=datasets,
            assertions=assertions,
            loc=self._loc(tree),
        )

    # === Constants ===

    def tunable(self, items: list) -> tuple[Expr, Expr]:
        return (items[0], items[1])

    def EXPORT(self, token: Any) -> str:
        return "export"

    @v_args(tree=True)
    def const(self, tree: Any) -> Const:
        # Filter out None values from optional elements
        items = [x for x in tree.children if x is not None]

        export = False

        # Check for 'export' keyword (now a token)
        if items and items[0] == "export":
            export = True
            items = items[1:]

        # Now items should be: [name, value] or [name, value, bounds]
        name = items[0]
        value = items[1]

        bounds = None
        if len(items) > 2:
            bounds = items[2]

        return Const(
            name=name,
            value=value,
            tunable=bounds is not None,
            bounds=bounds,
            export=export,
            loc=self._loc(tree),
        )

    # === Profiles ===

    def date_expr(self, items: list) -> DateExpr:
        item = items[0]
        if isinstance(item, date):
            return DateExpr(value=item)
        # DateExpr from date_func or date arithmetic
        return item

    def date_func(self, items: list) -> DateExpr:
        func_name = items[0]
        # Filter out None from optional date_func_args
        args_items = [x for x in items[1:] if x is not None]
        if args_items:
            # Has arguments from date_func_args
            args_list = args_items[0] if isinstance(args_items[0], list) else args_items
            args = ", ".join(str(a) for a in args_list)
            return DateExpr(value=f"{func_name}({args})")
        return DateExpr(value=f"{func_name}()")

    def date_func_args(self, items: list) -> list:
        return list(items)

    def date_add(self, items: list) -> DateExpr:
        base = items[0]
        offset = items[1]
        return DateExpr(value=base.value, offset=base.offset + offset)

    def date_sub(self, items: list) -> DateExpr:
        base = items[0]
        offset = items[1]
        return DateExpr(value=base.value, offset=base.offset - offset)

    # === Rules ===

    def rule(self, items: list) -> DisableRule | ScaleRule | SetSeverityRule:
        return items[0]

    def disable_rule(self, items: list) -> DisableRule:
        return items[0]

    def disable_check(self, items: list) -> DisableRule:
        return DisableRule(target_type="check", target_name=items[0])

    def disable_assertion(self, items: list) -> DisableRule:
        return DisableRule(target_type="assertion", target_name=items[0], in_check=items[1])

    def scale_rule(self, items: list) -> ScaleRule:
        selector = items[0]
        multiplier = items[1]
        return ScaleRule(
            selector_type=selector[0],
            selector_name=selector[1],
            multiplier=multiplier,
        )

    def set_rule(self, items: list) -> SetSeverityRule:
        selector = items[0]
        severity = items[1]
        return SetSeverityRule(
            selector_type=selector[0],
            selector_name=selector[1],
            severity=severity,
        )

    def selector(self, items: list) -> tuple[str, str]:  # pragma: no cover - grammar routes to specific handlers
        return items[0]

    def sel_check(self, items: list) -> tuple[str, str]:
        return ("check", items[0])

    def sel_tag(self, items: list) -> tuple[str, str]:
        return ("tag", items[0])

    def profile_body(self, items: list) -> tuple[str, DateExpr, DateExpr, tuple]:
        profile_type = items[0]
        from_date = items[1]
        to_date = items[2]
        rules = tuple(items[3:])
        return (profile_type, from_date, to_date, rules)

    @v_args(tree=True)
    def profile(self, tree: Any) -> Profile:
        items = tree.children
        name = items[0]
        body = items[1]
        profile_type, from_date, to_date, rules = body
        return Profile(
            name=name,
            profile_type=profile_type,
            from_date=from_date,
            to_date=to_date,
            rules=rules,
            loc=self._loc(tree),
        )

    # === Imports ===

    @v_args(tree=True)
    def import_simple(self, tree: Any) -> Import:
        items = tree.children
        path = items[0]
        alias = items[1] if len(items) > 1 else None
        return Import(path=path, alias=alias, loc=self._loc(tree))

    @v_args(tree=True)
    def import_selective(self, tree: Any) -> Import:
        items = tree.children
        names = items[0]
        path = items[1]
        return Import(path=path, names=names, loc=self._loc(tree))

    def import_names(self, items: list) -> tuple[str, ...]:
        return tuple(items)

    # === Metadata ===

    def metadata(self, items: list) -> dict:
        return {"availability_threshold": items[0]}

    # === Suite ===

    def suite_body(self, items: list) -> dict:
        result: dict[str, Any] = {
            "checks": [],
            "profiles": [],
            "constants": [],
            "imports": [],
            "availability_threshold": None,
        }
        for item in items:
            if isinstance(item, Check):
                result["checks"].append(item)
            elif isinstance(item, Profile):
                result["profiles"].append(item)
            elif isinstance(item, Const):
                result["constants"].append(item)
            elif isinstance(item, Import):
                result["imports"].append(item)
            elif isinstance(item, dict):
                if "availability_threshold" in item:
                    result["availability_threshold"] = item["availability_threshold"]
        return result

    @v_args(tree=True)
    def suite(self, tree: Any) -> Suite:
        items = tree.children
        name = items[0]
        body = items[1]
        return Suite(
            name=name,
            checks=tuple(body["checks"]),
            profiles=tuple(body["profiles"]),
            constants=tuple(body["constants"]),
            imports=tuple(body["imports"]),
            availability_threshold=body["availability_threshold"],
            loc=self._loc(tree),
        )

    def start(self, items: list) -> Suite:
        # Clear any stale annotations (defensive)
        self._pending_annotations = []
        return items[0]


def parse(source: str, filename: str | None = None) -> Suite:
    """Parse DQL source code into an AST.

    Args:
        source: DQL source code string
        filename: Optional filename for error reporting

    Returns:
        Suite AST node

    Raises:
        DQLSyntaxError: If the source contains syntax errors
    """
    try:
        tree = _parser.parse(source)
        transformer = DQLTransformer(filename=filename)
        return transformer.transform(tree)
    except UnexpectedCharacters as e:
        lines = source.splitlines()
        source_line = lines[e.line - 1] if e.line <= len(lines) else None
        loc = SourceLocation(line=e.line, column=e.column, filename=filename)
        raise DQLSyntaxError(
            message=f"Unexpected character: {e.char!r}",
            loc=loc,
            source_line=source_line,
        ) from None
    except UnexpectedToken as e:
        lines = source.splitlines()
        source_line = lines[e.line - 1] if e.line <= len(lines) else None
        loc = SourceLocation(line=e.line, column=e.column, filename=filename)
        expected = ", ".join(sorted(e.expected)[:5])
        raise DQLSyntaxError(
            message=f"Unexpected token: {e.token!r}",
            loc=loc,
            source_line=source_line,
            suggestion=f"Expected one of: {expected}" if expected else None,
        ) from None


def parse_file(path: str | Path) -> Suite:
    """Parse a DQL file into an AST.

    Args:
        path: Path to the DQL file

    Returns:
        Suite AST node

    Raises:
        DQLSyntaxError: If the file contains syntax errors
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    return parse(source, filename=str(path))
