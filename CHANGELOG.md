## 0.3.0 (2025-10-17)

### BREAKING CHANGE

- severity parameter no longer accepts None. All assertions now require a severity level, defaulting to P1 if not specified.

### Feat

- Symbol natural ordering
- Add is_between assertion to API with validation
- Add is_between function with comprehensive tests
- Update api.py to use analyzer.report.persist() instead of analyzer.persist()
- Add persist methods to AnalysisReport with corrected self-references
- Complete DuplicateCount implementation with state, spec, and provider integration
- Add DuplicateCount op with SQL dialect translation
- implement graph property and rename collect() to build_graph()
- add defensive graph property with explicit _graph_built flag
- add yamllint for YAML file validation
- add shfmt for shell script formatting
- update CI quality script to check examples directory
- add examples directory to mypy type checking
- add shellcheck to pre-commit for shell script validation
- implement evaluator validation logic
- add AssertionStatus type alias for validation results
- add complex number detection to evaluator
- implement evaluation failure refactoring plan v5
- add EvaluationFailure and SymbolInfo dataclasses
- update API to pass provider to validator
- implement dataset validation with ambiguity detection
- add minimal DatasetValidator class structure
- add comprehensive validation framework for VerificationSuite
- integrate validator with VerificationSuite
- implement complete validator in single file
- update CheckNode validation to use parent datasets
- add _visit_root_node to set datasets on RootNode
- add datasets field to RootNode for hierarchical imputation
- make assertion severity mandatory with P1 default
- **check**: @check decorator should have mandatory name field
- **api**: change the assertion builder interface: on -> with, label -> name
- refactor datasource interface, decoupling the need of importing dqx in child projects
- integrate dialect in analyzer and remove legacy sql in ops
- **dialect.py**: add dialect module and duckdb dialect
- **whole-code-base**: improve readability via ai vibe coding

### Fix

- Implement lag date handling fix
- move rich from dev to main dependencies
- remove unnecessary dataset fallback in symbol collection plan
- update assertion result collection tests for new validation behavior
- handle complex infinity (zoo) in evaluator
- add __str__ methods to all MetricSpec classes
- resolve mypy errors in evaluator and tests
- correct type annotation for failures dict in Evaluator._gather
- address v5 plan feedback - fix demo typo, add constant expression test, add demo reference
- **check**: improve check type hinting
- **test_provider**: fix tests in test_provider
- Update test_provider.py to match new SymbolicMetric API
- **test_provider**: fix tests in test_provider
- **graph.py**: fix the assertion evaluation condition

### Refactor

- Complete move of persist methods to AnalysisReport
- remove batch implementations and update tests
- simplify analyzer to single-source only
- remove batch processing protocols
- use graph property instead of _context._graph
- implement evaluator validation refactoring
- remove boolean value support from Evaluator
- remove unused bak files
- **display**: display a graph on the console
- everything
- **graph.py**: change graph design
- **analyzer.py**: code coverage for analyzer
- **pyarrow_ds.py**: improved datasource constructors
- **dialect.py**: better dialect integration with datasources
- **graph.py**: improve graph abstraction
- **graph.py**: consolidate graph tests
- **graph.py**: improve graph display and test coverage
- **graph.py**: remove backward compatibility code
- **graph.py**: rewrite graph data structure and 100% test coverage
- **graph.py,-api.py**: improve graph structure

### Perf

- **toolz**: remove dependency on toolz
