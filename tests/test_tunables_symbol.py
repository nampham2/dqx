"""Tests for TunableSymbol and arithmetic operators."""

from __future__ import annotations

import sympy as sp

from dqx.tunables import (
    TunableChoice,
    TunableFloat,
    TunableInt,
    TunablePercent,
    TunableSymbol,
)


class TestTunableSymbol:
    """Tests for TunableSymbol class."""

    def test_create_tunable_symbol_from_tunable_percent(self) -> None:
        """Can create TunableSymbol from TunablePercent."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        symbol = TunableSymbol(threshold)

        assert isinstance(symbol, sp.Symbol)
        assert isinstance(symbol, TunableSymbol)
        assert symbol.name == "THRESHOLD"
        assert symbol.tunable is threshold
        assert symbol.value == 0.05

    def test_create_tunable_symbol_from_tunable_float(self) -> None:
        """Can create TunableSymbol from TunableFloat."""
        tolerance = TunableFloat("TOL", value=0.001, bounds=(0.0, 1.0))
        symbol = TunableSymbol(tolerance)

        assert symbol.name == "TOL"
        assert symbol.tunable is tolerance
        assert symbol.value == 0.001

    def test_create_tunable_symbol_from_tunable_int(self) -> None:
        """Can create TunableSymbol from TunableInt."""
        min_rows = TunableInt("MIN_ROWS", value=1000, bounds=(100, 10000))
        symbol = TunableSymbol(min_rows)

        assert symbol.name == "MIN_ROWS"
        assert symbol.tunable is min_rows
        assert symbol.value == 1000

    def test_create_tunable_symbol_from_tunable_choice(self) -> None:
        """Can create TunableSymbol from TunableChoice."""
        method = TunableChoice("METHOD", value="mean", choices=("mean", "median", "max"))
        symbol = TunableSymbol(method)

        assert symbol.name == "METHOD"
        assert symbol.tunable is method
        assert symbol.value == "mean"

    def test_tunable_symbol_value_updates_with_tunable(self) -> None:
        """TunableSymbol.value reflects current tunable value."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        symbol = TunableSymbol(threshold)

        assert symbol.value == 0.05

        # Update tunable
        threshold.set(0.10)

        # Symbol should reflect new value
        assert symbol.value == 0.10

    def test_tunable_symbol_in_sympy_expression(self) -> None:
        """TunableSymbol can be used in SymPy expressions."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        symbol = TunableSymbol(threshold)
        x = sp.Symbol("x")

        # Create expression
        expr = x - symbol

        # Expression should contain both symbols
        assert x in expr.free_symbols
        assert symbol in expr.free_symbols

    def test_tunable_symbol_atoms(self) -> None:
        """TunableSymbol can be extracted from expressions using atoms()."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        symbol = TunableSymbol(threshold)
        x = sp.Symbol("x")

        expr = x - symbol

        # Extract TunableSymbol instances
        tunable_symbols = expr.atoms(TunableSymbol)

        assert len(tunable_symbols) == 1
        assert symbol in tunable_symbols

    def test_multiple_tunable_symbols_in_expression(self) -> None:
        """Multiple TunableSymbol instances can be used in one expression."""
        t1 = TunablePercent("T1", value=0.05, bounds=(0.0, 0.20))
        t2 = TunablePercent("T2", value=0.10, bounds=(0.0, 0.20))
        s1 = TunableSymbol(t1)
        s2 = TunableSymbol(t2)
        x = sp.Symbol("x")

        expr = x - s1 + s2

        tunable_symbols = expr.atoms(TunableSymbol)
        assert len(tunable_symbols) == 2
        assert s1 in tunable_symbols
        assert s2 in tunable_symbols

    def test_tunable_symbol_preserves_reference(self) -> None:
        """TunableSymbol maintains reference to original Tunable."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        symbol = TunableSymbol(threshold)

        # The tunable reference should be the exact same object
        assert symbol.tunable is threshold
        assert id(symbol.tunable) == id(threshold)


class TestTunableArithmeticOperators:
    """Tests for arithmetic operators on Tunable base class."""

    def test_tunable_addition(self) -> None:
        """Tunable supports addition operator."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        x = sp.Symbol("x")

        # Tunable + x
        expr = threshold + x
        assert isinstance(expr, sp.Expr)
        assert len(expr.atoms(TunableSymbol)) == 1

        # x + Tunable (reverse)
        expr2 = x + threshold
        assert isinstance(expr2, sp.Expr)
        assert len(expr2.atoms(TunableSymbol)) == 1

    def test_tunable_subtraction(self) -> None:
        """Tunable supports subtraction operator."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        x = sp.Symbol("x")

        # Tunable - x
        expr = threshold - x
        assert isinstance(expr, sp.Expr)
        assert len(expr.atoms(TunableSymbol)) == 1

        # x - Tunable (reverse)
        expr2 = x - threshold
        assert isinstance(expr2, sp.Expr)
        assert len(expr2.atoms(TunableSymbol)) == 1

    def test_tunable_multiplication(self) -> None:
        """Tunable supports multiplication operator."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        x = sp.Symbol("x")

        # Tunable * x
        expr = threshold * x
        assert isinstance(expr, sp.Expr)
        assert len(expr.atoms(TunableSymbol)) == 1

        # x * Tunable (reverse)
        expr2 = x * threshold
        assert isinstance(expr2, sp.Expr)
        assert len(expr2.atoms(TunableSymbol)) == 1

    def test_tunable_division(self) -> None:
        """Tunable supports division operator."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        x = sp.Symbol("x")

        # Tunable / x
        expr = threshold / x
        assert isinstance(expr, sp.Expr)
        assert len(expr.atoms(TunableSymbol)) == 1

        # x / Tunable (reverse)
        expr2 = x / threshold
        assert isinstance(expr2, sp.Expr)
        assert len(expr2.atoms(TunableSymbol)) == 1

    def test_tunable_negation(self) -> None:
        """Tunable supports negation operator."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        expr = -threshold
        assert isinstance(expr, sp.Expr)
        assert len(expr.atoms(TunableSymbol)) == 1

    def test_tunable_comparison_less_than(self) -> None:
        """Tunable supports < comparison."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        assert threshold < 0.10
        assert not (threshold < 0.01)
        assert not (threshold < 0.05)  # Not less than itself

    def test_tunable_comparison_less_equal(self) -> None:
        """Tunable supports <= comparison."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        assert threshold <= 0.10
        assert threshold <= 0.05  # Equal
        assert not (threshold <= 0.01)

    def test_tunable_comparison_greater_than(self) -> None:
        """Tunable supports > comparison."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        assert threshold > 0.01
        assert not (threshold > 0.10)
        assert not (threshold > 0.05)  # Not greater than itself

    def test_tunable_comparison_greater_equal(self) -> None:
        """Tunable supports >= comparison."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        assert threshold >= 0.01
        assert threshold >= 0.05  # Equal
        assert not (threshold >= 0.10)

    def test_tunable_arithmetic_with_constants(self) -> None:
        """Tunable arithmetic works with numeric constants."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))

        expr1 = threshold + 0.10
        expr2 = threshold - 0.02
        expr3 = threshold * 2
        expr4 = threshold / 2

        # All should be valid expressions
        assert isinstance(expr1, sp.Expr)
        assert isinstance(expr2, sp.Expr)
        assert isinstance(expr3, sp.Expr)
        assert isinstance(expr4, sp.Expr)

    def test_complex_expression_with_tunable(self) -> None:
        """Tunable can be used in complex expressions."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        # Complex expression: (x + threshold) * y - 2 * threshold
        expr = (x + threshold) * y - 2 * threshold

        # Should contain the tunable symbol
        tunable_symbols = expr.atoms(TunableSymbol)
        assert len(tunable_symbols) == 1
        assert tunable_symbols.pop().tunable is threshold

    def test_tunable_in_assertion_pattern(self) -> None:
        """Tunable works in typical assertion patterns."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        null_rate = sp.Symbol("null_rate")

        # Typical pattern: null_rate < threshold
        expr = null_rate - threshold  # For is_lt comparison

        # Should contain tunable symbol
        tunable_symbols = expr.atoms(TunableSymbol)
        assert len(tunable_symbols) == 1

    def test_tunable_different_types_arithmetic(self) -> None:
        """Different Tunable types can be used in arithmetic."""
        percent = TunablePercent("PCT", value=0.05, bounds=(0.0, 1.0))
        int_val = TunableInt("INT", value=100, bounds=(0, 1000))
        float_val = TunableFloat("FLT", value=0.5, bounds=(0.0, 1.0))

        x = sp.Symbol("x")

        # All should work
        expr1 = x + percent
        expr2 = x + int_val
        expr3 = x + float_val

        assert len(expr1.atoms(TunableSymbol)) == 1
        assert len(expr2.atoms(TunableSymbol)) == 1
        assert len(expr3.atoms(TunableSymbol)) == 1


class TestTunableSymbolExtraction:
    """Tests for extracting TunableSymbol from expressions."""

    def test_extract_single_tunable_from_expression(self) -> None:
        """Can extract a single tunable from an expression."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        x = sp.Symbol("x")

        expr = x - threshold

        # Extract tunables
        tunable_symbols = expr.atoms(TunableSymbol)
        assert len(tunable_symbols) == 1

        # Get the tunable
        symbol = tunable_symbols.pop()
        assert symbol.tunable is threshold

    def test_extract_multiple_tunables_from_expression(self) -> None:
        """Can extract multiple tunables from an expression."""
        t1 = TunablePercent("T1", value=0.05, bounds=(0.0, 0.20))
        t2 = TunableInt("T2", value=100, bounds=(0, 1000))
        x = sp.Symbol("x")

        expr = x - t1 + t2

        # Extract tunables
        tunable_symbols = expr.atoms(TunableSymbol)
        assert len(tunable_symbols) == 2

        # Get the tunables (use list since tunables are not hashable)
        tunables = [ts.tunable for ts in tunable_symbols]
        assert t1 in tunables
        assert t2 in tunables

    def test_extract_tunable_from_nested_expression(self) -> None:
        """Can extract tunable from nested expressions."""
        threshold = TunablePercent("THRESHOLD", value=0.05, bounds=(0.0, 0.20))
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        # Nested: (x + threshold) / y
        expr = (x + threshold) / y

        tunable_symbols = expr.atoms(TunableSymbol)
        assert len(tunable_symbols) == 1
        assert tunable_symbols.pop().tunable is threshold

    def test_no_tunables_in_regular_expression(self) -> None:
        """Regular expressions without tunables return empty set."""
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        expr = x + y - 5

        tunable_symbols = expr.atoms(TunableSymbol)
        assert len(tunable_symbols) == 0
