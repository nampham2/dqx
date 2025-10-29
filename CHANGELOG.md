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

## v0.5.4 (2025-10-29)

### Fix

- resolve GitHub release character limit issue
- resolve release workflow issues and update dqlib documentation (#7)

## v0.5.3 (2025-10-29)

### Fix

- resolve release workflow issues and update dqlib documentation

## v0.5.2 (2025-10-29)

### Fix

- **ci**: remove docker-publish job from release workflow (#5)

## v0.5.1 (2025-10-29)

### Fix

- **ci**: remove docker-publish job from release workflow

## v0.5.0 (2025-10-29)

### Feat

- add __version__ attribute to dqx module

### Fix

- **ci**: fix test-release job to handle PEP 668 system Python protection
- **ci**: remove redundant pull_request_target from release-drafter workflow
- **ci**: add pull-requests write permission to docs workflow
- **ci**: correct codecov action parameter from 'file' to 'files'
- **ci**: add --system flag to release workflow test installation

## v0.4.0 (2025-10-29)

### BREAKING CHANGE

- Remove ResultKeyProvider from public API and replace key parameter with lag parameter
- execution_id is no longer available in tags, use metadata.execution_id instead
- VerificationSuite.run() now takes a list of datasources
instead of a dict. The method internally creates the dict using each
datasource's name property.
- SymbolInfo no longer has a suite field. The collect_symbols()
method no longer accepts a suite_name parameter.
- collect_symbols() is no longer available on VerificationSuite.
Use suite.provider.collect_symbols(key, suite_name) instead.
- SymbolicMetric now includes parent_symbol field
- analyze_batch method removed from Analyzer protocol.
Use analyze method instead which now handles all date-based grouping internally.
- ArrowDataSource is no longer available. Use DuckRelationDataSource.from_arrow() instead.
- AuditPlugin output format changed from tables to text
- register_plugin() now accepts a fully qualified class name string
instead of a plugin instance. This enables lazy loading and better separation of
concerns.
- VerificationSuite no longer accepts plugin_manager parameter.
Use suite.plugin_manager.register_plugin() instead.
- Removed approx_cardinality method and all sketch-related functionality
- Logger output now goes to stdout instead of stderr
- severity parameter no longer accepts None. All assertions now require a severity level, defaulting to P1 if not specified.

### Feat

- improve logging for symbol deduplication and unused symbol removal
- **execution-id**: add execution_id tracking throughout the system
- investigate and document data integrity warning behavior
- **orm**: add get_by_execution_id method to MetricDB
- add UniqueCount metric implementation
- add uv run cleanup command
- replace bin/run-hooks.sh with uv run hooks command
- add data discrepancy display to AuditPlugin
- **data**: add metric trace statistics functionality
- implement symbol deduplication and refactor lag handling
- **data,display**: add metric_trace function and symbol-sorted displays
- **data**: add symbols_to_pyarrow_table function
- **data**: add PyArrow transformation functions for metrics and analysis reports
- add analysis_reports property to VerificationSuite
- add analysis report symbol mapping and display functionality
- **display**: add print_metrics_by_execution_id function
- **metadata**: add metadata support to Analyzer and persistence
- **metadata**: add metadata field to database schema and models
- **metadata**: add Metadata dataclass and MetadataType for persistence
- **api**: add execution ID tracking and retrieval functionality
- add dataset persistence to Metric model
- **provider**: add print_symbols convenience method to SymbolicMetricBase
- reverse parent-child relationship for extended metrics
- add parent-child dataset validation for extended metrics
- enhance dataset validation for parent-child symbol relationships
- update extended metrics fix plan with efficient children tracking
- complete CountValues implementation with all components
- add @overload decorators to count_values method
- implement CountValues op and fix dialect registry test isolation
- **compute**: add week_over_week metric and optimize timeseries checking
- **api**: add noop assertion for metric collection without validation
- **validator**: add UnusedSymbolValidator to detect unused symbols
- **dialect**: add BigQuery SQL dialect support
- **dialect**: implement MAP-based batch SQL optimization
- add DuckDB query optimization demo with GROUP BY approach
- **analyzer**: improve logging for analyze_batch method
- **analyzer**: implement batch analysis functionality
- migrate DuckRelationDataSource to datasource module with TDD
- implement VerificationSuite caching and is_critical method
- implement caching for collect_results and collect_symbols methods
- **plugins**: implement comprehensive validation for plugin instance registration
- implement minimal plugin instance registration with passing test
- implement plugin architecture with lazy loading and public API
- **examples**: add comprehensive plugin system demo
- **plugins**: integrate plugin system into VerificationSuite API
- **plugins**: add core plugin infrastructure with data structures
- add timer module and update plugin architecture plan v2
- update audit plugin to use Rich table display
- Pretty format SQL
- **logging**: add Rich logging with enhanced formatting
- add test coverage 100% implementation plan
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

- **ci**: remove invalid backslash in release workflow
- **ci**: resolve documentation build and deployment issues
- **ci**: remove license-check job from CodeQL workflow
- **ci**: disable examples workflow and fix CodeQL issues
- **ci**: add coverage relative paths and fix deprecated set-output
- **ci**: add project flag to BigQuery emulator and test fixtures
- symbol removal should remove dependent symbols
- properly propagate lag values for nested extended metrics
- resolve TypeError for nested extended metrics hashing
- add missing execution_id parameter to PluginExecutionContext in tests
- ensure latest metrics are returned when multiple exist with same key
- update setup-dev-env.sh to use new uv run commands
- resolve mypy type errors and remove unstable stdout tests
- update MetricKey to include dataset as third element in tuple
- simple metric double lag issue
- fix extended metrics
- remove provider.pyi and fix mypy errors in tests
- **analyzer**: ensure unique symbols for lag operations and include all computed symbols
- move pyarrow import and fix type annotation
- move execution_id from tags to metadata column
- **provider**: correct lag metric dates in symbol collection
- **graph**: add recursive dataset imputation for child dependencies in visitors
- resolve hanging test and improve extended metric tracking
- **provider**: fix ExtendedMetricProvider methods to accept symbols instead of MetricSpec
- resolve mypy type errors in ops and provider
- **examples**: add type annotations to fix mypy errors in BigQuery demo
- resolve batch analysis deduplication issue with date grouping
- display full extended metric names in symbol info
- **plugins**: use highlight=False to prevent Rich color bleeding in duration
- **plugins**: escape duration value to prevent Rich color bleeding
- **tests**: update regex pattern in test_plugin_metadata_not_callable
- eliminate duplicate pre-commit output by adding explicit stages
- update remaining test error messages to match source code
- update test error messages to match source code
- resolve mypy errors in test_plugin_integration.py
- **tests**: fix test_plugin_validation_uses_isinstance to use local test classes
- update plugin architecture to avoid naming conflicts
- **pre-commit**: fix commitizen hook to properly validate commit messages
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

- reorganize common.py for better readability
- rename Stddev lag parameter to offset for clarity
- use Python wrapper for commit-msg hook
- move development scripts to src/scripts for proper uv integration
- **display**: reorder columns in print_metric_trace to show Value Analysis before Value DB
- remove lag(N) prefix from metric names
- consolidate visitor classes and fix test timeouts
- **plan**: update v2 plan to use 'lag' consistently instead of 'lag_offset'
- update SqlDataSource protocol with read-only name property
- remove suite field from SymbolInfo dataclass
- **api**: move collect_symbols from VerificationSuite to SymbolicMetricBase
- simplify _children_map using defaultdict
- remove test_print_symbols_hierarchical and update display module
- separate type hints into provider.pyi stub file
- **dialect**: extract common batch CTE query logic into utility functions
- remove analyze_batch method and simplify Analyzer interface
- remove nominal_date parameter from SqlDataSource.query method
- remove unused tags parameter from @check decorator
- remove old extensions module after successful migration
- update example files to import from datasource module
- update suite and remaining tests to import from datasource module
- update API tests to import from datasource module
- update validation and integration tests to import from datasource module
- update analyzer tests to import from datasource module
- migrate from pyarrow_ds to duckds implementation
- **plugins**: simplify AuditPlugin to use text-based output
- rename _register_from_string to _register_from_class in v2 plan
- **plugins**: rename ResultProcessor to PostProcessor protocol
- simplify plugin registration to use class names instead of instances
- extract plugin execution to separate method in api.py
- consolidate documentation tasks in plugin architecture plan
- **ops**: remove unused OpsType type alias and fix examples
- remove sketch-based cardinality estimation functionality
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

- optimize get_symbol() calls in extended metrics
- **toolz**: remove dependency on toolz
