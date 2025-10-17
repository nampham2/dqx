# Test Coverage 100% Implementation Plan v1

## Overview

This plan outlines the implementation strategy to achieve 100% test coverage for the DQX project. Current coverage is at 98% (2173/2211 lines covered), with 38 lines missing across 9 files.

## Current Coverage Status

```
Name                               Stmts   Miss  Cover   Missing
----------------------------------------------------------------
src/dqx/analyzer.py                  114      1    99%   228
src/dqx/api.py                       199     10    95%   148-149, 159, 163, 308, 310, 423, 495, 549, 552
src/dqx/display.py                    94      1    99%   51
src/dqx/evaluator.py                  76      5    93%   115, 120, 222, 232-234
src/dqx/graph/base.py                 47      1    98%   112
src/dqx/graph/traversal.py            55      7    87%   150-157, 354
src/dqx/graph/visitors.py             61      3    95%   102, 168, 231
src/dqx/specs.py                     321      1    99%   468
src/dqx/validator.py                 156      9    94%   85, 121, 129, 162-163, 328, 332-334
----------------------------------------------------------------
TOTAL                               2211     38    98%
```

## Implementation Groups

### Group 1: Quick Wins - Simple Coverage Gaps

**Files to modify:**
- `tests/test_display.py`
- `tests/graph/test_base.py`
- `tests/test_validator.py`
- `tests/test_specs.py`

**Task 1.1: Test SimpleNodeFormatter with node_name() method**
```python
# In tests/test_display.py
def test_simple_node_formatter_with_node_name():
    """Test formatter with a node that has node_name() method."""
    from dqx.display import SimpleNodeFormatter
    from dqx.graph.nodes import CheckNode, RootNode

    formatter = SimpleNodeFormatter()
    root = RootNode("test_suite")
    check = root.add_check(name="test_check", tags=["test"])

    # CheckNode has node_name() method
    result = formatter.format_node(check)
    assert result == "test_check"
```

**Task 1.2: Test BaseNode.is_leaf() NotImplementedError**
```python
# In tests/graph/test_base.py
def test_base_node_is_leaf_not_implemented():
    """Test that BaseNode.is_leaf() raises NotImplementedError."""
    from dqx.graph.base import BaseNode

    # Create a minimal concrete implementation for testing
    class TestNode(BaseNode):
        pass

    node = TestNode(parent=None)
    with pytest.raises(NotImplementedError):
        node.is_leaf()
```

**Task 1.3: Test ValidationReport.__str__() with no issues**
```python
# In tests/test_validator.py
def test_validation_report_str_no_issues():
    """Test ValidationReport string representation with no issues."""
    from dqx.validator import ValidationReport

    report = ValidationReport()
    assert str(report) == "No validation issues found."
```

**Task 1.4: Test BaseValidator.get_issues()**
```python
# In tests/test_validator.py
def test_base_validator_get_issues():
    """Test BaseValidator.get_issues() method."""
    from dqx.validator import EmptyCheckValidator

    validator = EmptyCheckValidator()
    issues = validator.get_issues()
    assert issues == []
```

**Task 1.5: Identify and test specs.py line 468**
```python
# First, identify what's on line 468 of specs.py by examining the code
# Then write appropriate test in tests/test_specs.py
```

**Verification:** Run `uv run pytest tests/test_display.py tests/graph/test_base.py tests/test_validator.py tests/test_specs.py -v` and check coverage.

**Commit:** After verification passes:
```bash
git add tests/test_display.py tests/graph/test_base.py tests/test_validator.py tests/test_specs.py
git commit -m "test: add coverage for simple gaps - Group 1

- Add test for SimpleNodeFormatter with node_name() method
- Add test for BaseNode.is_leaf() NotImplementedError
- Add test for ValidationReport.__str__() with no issues
- Add test for BaseValidator.get_issues()
- Add test for specs.py line 468
- Coverage improved from 98% to ~99%
- Addresses lines: display.py:51, graph/base.py:112, validator.py:85,121, specs.py:468"
```

### Group 2: Async Method Coverage

**Files to modify:**
- `tests/graph/test_traversal.py`
- `tests/graph/test_visitor.py`
- `tests/test_validator.py`

**Task 2.1: Test Graph.async_bfs() method**
```python
# In tests/graph/test_traversal.py
@pytest.mark.asyncio
async def test_graph_async_bfs():
    """Test asynchronous breadth-first traversal."""
    from dqx.graph.nodes import RootNode
    from dqx.graph.traversal import Graph
    from dqx.graph.visitors import NodeCollector

    root = RootNode("test_suite")
    check1 = root.add_check("check1")
    check2 = root.add_check("check2")

    graph = Graph(root)
    collector = NodeCollector(CheckNode)

    await graph.async_bfs(collector)

    assert len(collector.results) == 2
    assert all(isinstance(node, CheckNode) for node in collector.results)
```

**Task 2.2: Test visitor async methods**
```python
# In tests/graph/test_visitor.py
@pytest.mark.asyncio
async def test_dataset_imputation_visitor_async():
    """Test DatasetImputationVisitor async visit method."""
    from dqx.graph.visitors import DatasetImputationVisitor
    from dqx.graph.nodes import RootNode

    visitor = DatasetImputationVisitor(["dataset1"], None)
    root = RootNode("test")

    await visitor.visit_async(root)
    assert root.datasets == ["dataset1"]

@pytest.mark.asyncio
async def test_node_collector_async():
    """Test NodeCollector async visit method."""
    from dqx.graph.visitors import NodeCollector
    from dqx.graph.nodes import CheckNode, RootNode

    collector = NodeCollector(CheckNode)
    root = RootNode("test")
    check = root.add_check("test_check")

    await collector.visit_async(root)
    await collector.visit_async(check)

    assert len(collector.results) == 1
    assert collector.results[0] == check
```

**Task 2.3: Test CompositeValidationVisitor.visit_async()**
```python
# In tests/test_validator.py
@pytest.mark.asyncio
async def test_composite_validation_visitor_async():
    """Test CompositeValidationVisitor async visit method."""
    from dqx.validator import CompositeValidationVisitor, EmptyCheckValidator
    from dqx.graph.nodes import RootNode

    validators = [EmptyCheckValidator()]
    composite = CompositeValidationVisitor(validators)

    root = RootNode("test")
    await composite.visit_async(root)

    # Verify node was collected
    assert len(composite._nodes) == 1
```

**Verification:** Run `uv run pytest tests/graph/test_traversal.py tests/graph/test_visitor.py tests/test_validator.py -v -k async` and check coverage.

**Commit:** After verification passes:
```bash
git add tests/graph/test_traversal.py tests/graph/test_visitor.py tests/test_validator.py
git commit -m "test: add async method coverage - Group 2

- Add test for Graph.async_bfs() method
- Add tests for visitor async methods (DatasetImputationVisitor, NodeCollector)
- Add test for CompositeValidationVisitor.visit_async()
- Addresses lines: graph/traversal.py:150-157, graph/visitors.py:168,231, validator.py:328"
```

### Group 3: Error Path Coverage - Analyzer

**Files to modify:**
- `tests/test_analyzer.py`

**Task 3.1: Test AnalysisReport with empty report logging**
```python
# In tests/test_analyzer.py
def test_analysis_report_persist_empty(caplog):
    """Test persisting empty AnalysisReport logs warning."""
    from dqx.analyzer import AnalysisReport
    from unittest.mock import Mock

    report = AnalysisReport()
    mock_db = Mock()

    with caplog.at_level(logging.WARNING):
        report.persist(mock_db)

    assert "Try to save an EMPTY analysis report!" in caplog.text
    mock_db.persist.assert_not_called()
```

**Verification:** Run `uv run pytest tests/test_analyzer.py -v -k empty` and check coverage.

**Commit:** After verification passes:
```bash
git add tests/test_analyzer.py
git commit -m "test: add analyzer error path coverage - Group 3

- Add test for AnalysisReport.persist() with empty report
- Verifies warning log message for empty report
- Addresses line: analyzer.py:228"
```

### Group 4: Error Path Coverage - API

**Files to modify:**
- `tests/test_api.py`

**Task 4.1: Test AssertionDraft error conditions**
```python
# In tests/test_api.py
def test_assertion_draft_where_empty_name():
    """Test AssertionDraft.where() with empty name."""
    from dqx.api import AssertionDraft
    import sympy as sp

    draft = AssertionDraft(sp.Symbol("x"))

    with pytest.raises(ValueError, match="Assertion name cannot be empty"):
        draft.where(name="")

    with pytest.raises(ValueError, match="Assertion name cannot be empty"):
        draft.where(name="   ")

def test_assertion_draft_where_long_name():
    """Test AssertionDraft.where() with name too long."""
    from dqx.api import AssertionDraft
    import sympy as sp

    draft = AssertionDraft(sp.Symbol("x"))
    long_name = "a" * 256

    with pytest.raises(ValueError, match="Assertion name is too long"):
        draft.where(name=long_name)
```

**Task 4.2: Test AssertionReady without context**
```python
# In tests/test_api.py
def test_assertion_ready_without_context():
    """Test AssertionReady methods without context."""
    from dqx.api import AssertionReady
    import sympy as sp

    # Create AssertionReady without context
    ready = AssertionReady(
        actual=sp.Symbol("x"),
        name="test",
        severity="P1",
        context=None
    )

    # Should not raise, just return None
    ready.is_positive()
```

**Task 4.3: Test VerificationSuite error conditions**
```python
# In tests/test_api.py
def test_verification_suite_graph_before_build():
    """Test accessing graph before building."""
    from dqx.api import VerificationSuite, check
    from dqx.orm.repositories import MetricDB

    @check(name="test")
    def test_check(mp, ctx):
        pass

    db = MetricDB()
    suite = VerificationSuite([test_check], db, "Test Suite")

    with pytest.raises(DQXError, match="Graph not built yet"):
        _ = suite.graph

def test_verification_suite_validation_errors():
    """Test suite validation with errors."""
    from dqx.api import VerificationSuite, check
    from dqx.orm.repositories import MetricDB

    @check(name="duplicate")
    def check1(mp, ctx):
        pass

    @check(name="duplicate")
    def check2(mp, ctx):
        pass

    db = MetricDB()
    suite = VerificationSuite([check1, check2], db, "Test Suite")

    with pytest.raises(DQXError, match="Suite validation failed"):
        suite.build_graph(suite._context, ResultKey.from_date(date.today()))

def test_verification_suite_already_executed():
    """Test running suite that was already executed."""
    from dqx.api import VerificationSuite, check
    from dqx.orm.repositories import MetricDB

    @check(name="test")
    def test_check(mp, ctx):
        pass

    db = MetricDB()
    suite = VerificationSuite([test_check], db, "Test Suite")

    # Mark as already evaluated
    suite.is_evaluated = True

    with pytest.raises(DQXError, match="already been executed"):
        suite.run({}, ResultKey.from_date(date.today()))

def test_verification_suite_collect_before_run():
    """Test collecting results/symbols before run."""
    from dqx.api import VerificationSuite, check
    from dqx.orm.repositories import MetricDB

    @check(name="test")
    def test_check(mp, ctx):
        pass

    db = MetricDB()
    suite = VerificationSuite([test_check], db, "Test Suite")

    with pytest.raises(DQXError, match="Cannot collect results before suite execution"):
        suite.collect_results()

    with pytest.raises(DQXError, match="Cannot collect symbols before suite execution"):
        suite.collect_symbols()

def test_verification_suite_collect_without_key():
    """Test collecting results without key (edge case)."""
    from dqx.api import VerificationSuite, check
    from dqx.orm.repositories import MetricDB

    @check(name="test")
    def test_check(mp, ctx):
        pass

    db = MetricDB()
    suite = VerificationSuite([test_check], db, "Test Suite")
    suite.is_evaluated = True  # Force evaluated state
    suite._key = None  # But no key

    with pytest.raises(DQXError, match="No ResultKey available"):
        suite.collect_results()
```

**Verification:** Run `uv run pytest tests/test_api.py -v` and check coverage.

**Commit:** After verification passes:
```bash
git add tests/test_api.py
git commit -m "test: add API error path coverage - Group 4

- Add tests for AssertionDraft validation errors (empty/long names)
- Add test for AssertionReady without context
- Add tests for VerificationSuite error conditions
- Addresses lines: api.py:148-149,159,163,308,310,423,495,549,552"
```

### Group 5: Error Path Coverage - Evaluator

**Files to modify:**
- `tests/test_evaluator.py`

**Task 5.1: Test missing symbols in _gather()**
```python
# In tests/test_evaluator.py
def test_evaluator_gather_missing_symbol():
    """Test _gather() with missing symbol."""
    from dqx.evaluator import Evaluator
    from dqx.provider import MetricProvider
    from dqx.common import ResultKey
    import sympy as sp
    from datetime import date

    provider = MetricProvider(MetricDB())
    key = ResultKey.from_date(date.today())
    evaluator = Evaluator(provider, key, "test_suite")

    # Try to gather a symbol that doesn't exist
    expr = sp.Symbol("missing_symbol")

    with pytest.raises(DQXError, match="not found in collected metrics"):
        evaluator._gather(expr)

def test_evaluator_metric_for_symbol_not_found():
    """Test metric_for_symbol() with non-existent symbol."""
    from dqx.evaluator import Evaluator
    from dqx.provider import MetricProvider
    from dqx.common import ResultKey
    import sympy as sp
    from datetime import date

    provider = MetricProvider(MetricDB())
    key = ResultKey.from_date(date.today())
    evaluator = Evaluator(provider, key, "test_suite")

    # This will be called internally by _gather
    symbol = sp.Symbol("unknown")

    with pytest.raises(Exception):  # Provider will raise when symbol not found
        evaluator.metric_for_symbol(symbol)
```

**Task 5.2: Test complex number evaluation**
```python
# In tests/test_evaluator.py
def test_evaluator_complex_number_result():
    """Test evaluation resulting in complex number."""
    from dqx.evaluator import Evaluator
    from dqx.provider import MetricProvider
    from dqx.common import ResultKey
    from dqx.specs import MetricSpec
    import sympy as sp
    from datetime import date
    from returns.result import Success

    # Create a scenario that produces complex numbers
    provider = MetricProvider(MetricDB())
    key = ResultKey.from_date(date.today())

    # Register a metric that will produce negative value
    metric_spec = MetricSpec.build_average("value", "test_table")
    provider.register("x", metric_spec, key, dataset="test_ds")

    evaluator = Evaluator(provider, key, "test_suite")

    # Mock the metrics to return a value that will produce complex result
    evaluator._metrics = {sp.Symbol("x_1"): Success(-1.0)}

    # sqrt(-1) should produce complex number
    expr = sp.sqrt(sp.Symbol("x_1"))
    result = evaluator.evaluate(expr)

    assert result.is_failure()
    failure = result.failure()[0]
    assert "complex" in failure.error_message
```

**Task 5.3: Test evaluation error handling**
```python
# In tests/test_evaluator.py
def test_evaluator_general_exception():
    """Test evaluation with unexpected exception."""
    from dqx.evaluator import Evaluator
    from dqx.provider import MetricProvider
    from dqx.common import ResultKey
    import sympy as sp
    from datetime import date
    from unittest.mock import patch

    provider = MetricProvider(MetricDB())
    key = ResultKey.from_date(date.today())
    evaluator = Evaluator(provider, key, "test_suite")

    # Mock _gather to return valid values
    with patch.object(evaluator, '_gather') as mock_gather:
        mock_gather.return_value = ({sp.Symbol("x"): 1.0}, [])

        # Mock sp.N to raise an exception
        with patch('sympy.N', side_effect=Exception("Unexpected error")):
            expr = sp.Symbol("x")
            result = evaluator.evaluate(expr)

            assert result.is_failure()
            failure = result.failure()[0]
            assert "Error evaluating expression" in failure.error_message
```

**Verification:** Run `uv run pytest tests/test_evaluator.py -v` and check coverage.

**Commit:** After verification passes:
```bash
git add tests/test_evaluator.py
git commit -m "test: add evaluator error path coverage - Group 5

- Add tests for missing symbols in _gather()
- Add test for complex number evaluation
- Add test for general exception handling
- Addresses lines: evaluator.py:115,120,222,232-234"
```

### Group 6: Error Path Coverage - Graph Components

**Files to modify:**
- `tests/graph/test_traversal.py`
- `tests/graph/test_visitor.py`

**Task 6.1: Test dataset imputation errors**
```python
# In tests/graph/test_traversal.py
def test_graph_impute_datasets_with_errors():
    """Test dataset imputation that raises errors."""
    from dqx.graph.nodes import RootNode
    from dqx.graph.traversal import Graph
    from dqx.provider import MetricProvider
    from dqx.orm.repositories import MetricDB
    from dqx.common import DQXError

    root = RootNode("test_suite")
    check = root.add_check("test_check", datasets=["unknown_dataset"])

    graph = Graph(root)
    provider = MetricProvider(MetricDB())

    with pytest.raises(DQXError, match="not in parent datasets"):
        graph.impute_datasets(["dataset1"], provider)
```

**Task 6.2: Test assertion node visit without provider**
```python
# In tests/graph/test_visitor.py
def test_dataset_imputation_visitor_assertion_no_provider():
    """Test visiting assertion node without provider."""
    from dqx.graph.visitors import DatasetImputationVisitor
    from dqx.graph.nodes import RootNode, AssertionNode
    import sympy as sp

    visitor = DatasetImputationVisitor(["dataset1"], None)

    root = RootNode("test")
    check = root.add_check("test_check")
    assertion = AssertionNode(
        parent=check,
        actual=sp.Symbol("x"),
        name="test",
        severity="P1",
        validator=None
    )

    # Should not raise, just return early
    visitor.visit(assertion)
```

**Verification:** Run `uv run pytest tests/graph/test_traversal.py tests/graph/test_visitor.py -v` and check coverage.

**Commit:** After verification passes:
```bash
git add tests/graph/test_traversal.py tests/graph/test_visitor.py
git commit -m "test: add graph components error path coverage - Group 6

- Add test for dataset imputation errors
- Add test for assertion node visit without provider
- Addresses lines: graph/traversal.py:354, graph/visitors.py:102"
```

### Group 7: Error Path Coverage - Validator

**Files to modify:**
- `tests/test_validator.py`

**Task 7.1: Test validator finalize method**
```python
# In tests/test_validator.py
def test_duplicate_check_name_validator_finalize():
    """Test DuplicateCheckNameValidator finalize method."""
    from dqx.validator import DuplicateCheckNameValidator
    from dqx.graph.nodes import CheckNode, RootNode

    validator = DuplicateCheckNameValidator()

    # Create duplicate checks
    root = RootNode("test")
    check1 = root.add_check("duplicate_name")
    check2 = root.add_check("duplicate_name")

    validator.process_node(check1)
    validator.process_node(check2)
    validator.finalize()

    issues = validator.get_issues()
    assert len(issues) == 1
    assert "duplicate_name" in issues[0].message
```

**Task 7.2: Test validator error vs warning collection**
```python
# In tests/test_validator.py
def test_composite_validation_visitor_error_warning_separation():
    """Test that CompositeValidationVisitor separates errors and warnings."""
    from dqx.validator import (
        CompositeValidationVisitor,
        DuplicateCheckNameValidator,
        EmptyCheckValidator
    )
    from dqx.graph.nodes import RootNode

    # DuplicateCheckNameValidator produces errors
    # EmptyCheckValidator produces warnings
    validators = [DuplicateCheckNameValidator(), EmptyCheckValidator()]
    composite = CompositeValidationVisitor(validators)

    root = RootNode("test")
    check1 = root.add_check("dup")
    check2 = root.add_check("dup")
    check3 = root.add_check("empty")  # No assertions = warning

    composite.visit(root)
    composite.visit(check1)
    composite.visit(check2)
    composite.visit(check3)

    issues = composite.get_all_issues()

    assert len(issues["errors"]) > 0  # Duplicate name error
    assert len(issues["warnings"]) > 0  # Empty check warning
```

**Task 7.3: Test validator reset**
```python
# In tests/test_validator.py
def test_composite_validation_visitor_reset():
    """Test CompositeValidationVisitor reset method."""
    from dqx.validator import CompositeValidationVisitor, EmptyCheckValidator
    from dqx.graph.nodes import RootNode

    validators = [EmptyCheckValidator()]
    composite = CompositeValidationVisitor(validators)

    # Add some nodes
    root = RootNode("test")
    composite.visit(root)

    assert len(composite._nodes) == 1

    # Reset
    composite.reset()

    assert len(composite._nodes) == 0
    for validator in composite._validators:
        assert validator.get_issues() == []
```

**Verification:** Run `uv run pytest tests/test_validator.py -v` and check coverage.

**Commit:** After verification passes:
```bash
git add tests/test_validator.py
git commit -m "test: add validator error path coverage - Group 7

- Add test for DuplicateCheckNameValidator finalize method
- Add test for CompositeValidationVisitor error/warning separation
- Add test for CompositeValidationVisitor reset method
- Addresses lines: validator.py:129,162-163,332-334"
```

### Final Verification

**Task: Run all tests and verify 100% coverage**
```bash
# Run all tests with coverage
uv run pytest --cov=dqx --cov-report=term-missing --cov-report=html

# Run pre-commit hooks
bin/run-hooks.sh

# Verify specific files if needed
uv run pytest --cov=dqx.analyzer --cov-report=term-missing tests/test_analyzer.py
uv run pytest --cov=dqx.api --cov-report=term-missing tests/test_api.py
# ... etc for other modules
```

**Final Commit:** After achieving 100% coverage:
```bash
git add .
git commit -m "test: achieve 100% test coverage

- Completed all 7 groups of test additions
- Coverage improved from 98% (2173/2211) to 100% (2211/2211)
- All 38 missing lines now covered
- All tests pass with pre-commit hooks"
```

## Notes

1. **TDD Approach**: Write tests first, verify they fail, then run to see coverage improvement
2. **Minimal Changes**: Each test should be focused on covering specific missing lines
3. **Mock Usage**: Use mocks sparingly, only when necessary to trigger specific error paths
4. **Async
