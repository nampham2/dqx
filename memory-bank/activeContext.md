# DQX Active Context

## Current Work Focus

### Recent Major Changes (v0.4.0)
1. **Breaking: Removed Assertion Chaining**
   - Assertions now return None instead of AssertBuilder
   - Each assertion is completely independent
   - Cleaner, more predictable API

2. **Breaking: Removed Listener Pattern**
   - AssertBuilder no longer accepts listeners parameter
   - Direct assertion node creation without indirection
   - Simplified architecture

3. **Immutable AssertionNode**
   - Removed setter methods (set_label, set_severity, set_validator)
   - All properties set at construction time
   - Improved reliability and predictability

### Active Development Areas

#### 1. **Symbol Table System Refactoring**
- Recently eliminated redundancy in SymbolEntry
- Now uses @property decorators for computed fields
- Single source of truth: SymbolicMetric.dependencies
- Maintained backward compatibility

#### 2. **Graph Architecture Improvements**
- Moved symbol tracking from CheckNode to AssertionNode
- CheckNode now focuses purely on aggregating assertion results
- Enhanced observer pattern for symbol state changes
- Better separation of concerns

#### 3. **Test Coverage Initiative**
- Achieved 100% coverage for:
  - graph.py module
  - display.py module
  - analyzer.py module
- Focus on maintaining 98%+ overall coverage
- Comprehensive e2e tests as ground truth

## Recent Decisions and Considerations

### 1. **No Backward Compatibility Policy**
- Explicit permission required from Nam before implementing ANY backward compatibility
- Allows for cleaner refactoring and better architecture
- Reduces technical debt

### 2. **Severity Model Simplification**
- Currently uses binary failure model (any assertion failure = check failure)
- Severity levels stored but not used for failure propagation
- Future enhancement opportunity for severity-aware execution

### 3. **Hybrid Dataset Approach**
- Single dataset: No specification required (backward compatible)
- Multiple datasets: Explicit specification required
- Clear error messages guide users
- Balances simplicity with flexibility

### 4. **Pre-commit Hook Expansion**
- Added comprehensive hooks for Python, shell, YAML, and docs
- Automated code quality enforcement
- Includes mypy type checking in pre-commit

## Important Patterns and Preferences

### Code Organization
1. **Protocol-Based Design**: Use Protocol classes for interfaces
2. **Composite Pattern**: For graph node hierarchy
3. **Visitor Pattern**: For graph traversal
4. **Builder Pattern**: For suite construction
5. **Single Responsibility**: Each class has one clear purpose

### Naming Conventions
1. **No Implementation Details in Names**: `Tool` not `MCPToolWrapper`
2. **No Temporal Context**: Avoid "New", "Legacy", "Enhanced"
3. **Domain-Focused**: Names tell what code does, not how

### Testing Patterns
1. **TDD Mandatory**: Write failing test first
2. **Prefer Native Objects**: Minimize mocks
3. **Isolated Tests**: Each test completely independent
4. **Never Modify e2e Tests**: Critical ground truth

### Documentation Standards
1. **Google Format Docstrings**: Consistent style
2. **Type Annotations**: All functions fully typed
3. **No Historical Comments**: Document current state only
4. **README Sync**: Keep README updated with changes

## Current Technical Debt

### 1. **Dialect Implementation**
- Successfully removed legacy .sql property
- All data sources now require dialect
- Unified SQL generation approach

### 2. **Display Logic Separation**
- Successfully extracted from graph.py to display.py
- Clear separation of concerns
- Both modules at 100% coverage

### 3. **Analyzer Optimization**
- Refactored to use sets for deduplication
- Simpler and more efficient than defaultdict
- Achieved 100% test coverage

## Next Steps

### Immediate (This Week)
1. Continue maintaining high test coverage
2. Review and update documentation for v0.4.0 changes
3. Ensure all examples work with new non-chaining API
4. Monitor for any issues with breaking changes

### Short Term (This Month)
1. Implement streaming data source support
2. Add more advanced metrics (entropy, ML scores)
3. Optimize performance for GPU acceleration
4. Create comprehensive migration guide

### Medium Term (Q1 2025)
1. Web dashboard development
2. Data catalog integrations
3. Workflow tool operators (Airflow, Prefect)
4. Enhanced error messages with suggestions

## Key Learnings

### 1. **Simplicity Wins**
- Removing assertion chaining made API clearer
- Direct approaches often better than clever patterns
- User confusion is a design smell

### 2. **Breaking Changes Can Be Good**
- When done thoughtfully, improves architecture
- Clear communication essential
- Semantic versioning helps users

### 3. **Test Coverage Pays Off**
- 100% coverage catches edge cases
- Refactoring confidence increases
- Documentation through tests

### 4. **Type System as Documentation**
- Strong typing reduces errors
- Protocols clarify interfaces
- Type hints improve IDE support

## Active Experiments

### 1. **Performance Benchmarking**
- Testing DuckDB vs other SQL engines
- Measuring memory usage of sketches
- Optimizing batch size for throughput

### 2. **API Ergonomics**
- Gathering feedback on v0.4.0 changes
- Considering severity-aware execution
- Exploring natural language assertions

### 3. **Integration Patterns**
- Best practices for Airflow integration
- DBT compatibility testing
- Real-time streaming prototypes

## Communication Patterns

### With Nam (Human Partner)
1. **Direct and Technical**: No sugar-coating
2. **Push Back When Needed**: Challenge bad ideas
3. **Ask for Clarification**: Don't assume
4. **Document Decisions**: Keep memory bank updated

### Code Reviews
1. **Focus on Correctness**: Right is better than fast
2. **Explain Why**: Not just what to change
3. **Suggest Alternatives**: Multiple solutions
4. **Test Everything**: No untested code

## Current Environment

### Development Setup
- Python 3.11/3.12 with uv package manager
- DuckDB as primary SQL engine
- Pre-commit hooks for quality
- VS Code as primary IDE

### Key Dependencies
- DuckDB ≥ 1.3.2: Analytical engine
- PyArrow ≥ 21.0.0: Columnar processing
- DataSketches ≥ 5.2.0: Statistical sketches
- SymPy ≥ 1.14.0: Symbolic mathematics
- SQLAlchemy ≥ 2.0.43: Database abstraction

### Testing Infrastructure
- pytest with coverage
- mypy for type checking
- ruff for linting and formatting
- Pre-commit for automation
