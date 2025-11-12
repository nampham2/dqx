# Active Context - DQX

## Current Focus
- Memory bank updates to align with current codebase (v0.5.9)
- Working on `main` branch

## Recent Changes

### CustomSQL Operation Implementation (2025-11-07)
- Added CustomSQL operation with universal parameter support
- Allows user-defined SQL expressions as metrics
- Parameters passed to CTE level for filtering/grouping
- Comprehensive test coverage across multiple test files
- Both DuckDB and BigQuery dialect support

### Date Exclusion Feature (2025-11-07)
- Implemented comprehensive date exclusion with data availability tracking
- Added `skip_dates` parameter to datasources
- Allows excluding specific dates from calculations
- Supports better handling of data gaps and maintenance windows

### Previous Work
- Logger API refactoring and type safety improvements
- DoD/WoW calculation fixes to use percentage change
- BigQuery SQL generation compatibility improvements
- Numpy dependency removal and project structure cleanup
- Cache system performance improvements with statistics tracking

## Next Steps
- Continue performance optimization efforts
- Expand plugin system beyond audit capabilities
- Monitor production usage and gather feedback
- Consider additional database dialect support

## Important Patterns and Preferences

### Development Methodology
- **3D Methodology (Design-Driven Development)**: Think before you build, build with intention, ship with confidence
- **KISS/YAGNI Principles**: Start simple, evolve thoughtfully
- **TDD Mandatory**: Follow 5-step process for every feature/bugfix

### Coding Standards
- **Type Hints Required**: Modern PEP 604 syntax (`list[str]` not `List[str]`)
- **F-strings Always**: `f"Value: {var}"` not string concatenation
- **No Temporal Names**: Avoid "new", "old", "legacy", "enhanced" in names
- **No Implementation Details in Names**: Use purpose, not technology

### Testing Philosophy
- **Real Objects Over Mocks**: Use in-memory databases, not mocks
- **Pattern Matching for Result/Maybe**: Never use isinstance
- **100% Coverage Required**: Maintain or exceed current levels
- **Pristine Test Output**: Capture and validate all expected errors

### Git Workflow
- **Conventional Commits Required**: feat, fix, docs, style, refactor, perf, test, build, ci, chore
- **Always Use --no-pager**: `git --no-pager log`
- **Never Skip Pre-commit Hooks**: Fix issues and retry
- **Commit Frequently**: Small, focused commits

## Recent Learnings

### CustomSQL Design
- Hash-based naming for unique identification
- SQL expression passed as-is to dialect
- Parameters handled at CTE level, not in SQL string
- Consistent with other operations' parameter handling

### Date Exclusion Pattern
- Implemented as set of dates on datasource
- Allows flexible data availability management
- Integrates cleanly with existing analyzer logic

### Performance Optimization
- Lock-free designs where possible
- Batch operations for efficiency
- Cache statistics for monitoring
- Single-pass SQL generation

## Recently Fixed Issues

### Version and Project Identity (FIXED 2025-11-07)
- Updated project version from 0.3.0 to 0.5.9
- Clarified project distributed as `dqlib` package
- Aligned with pyproject.toml configuration

### Branch Context (FIXED 2025-11-07)
- Updated from feat/bkng-integration to main branch
- Removed outdated branch references
- Aligned with current git state

### Missing Features Documentation (FIXED 2025-11-07)
- Added CustomSQL operation documentation
- Added date exclusion feature documentation
- Updated with recent architectural improvements
