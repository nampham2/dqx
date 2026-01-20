# DQX Feature Development Workflow - Complete Example

This document walks through a complete feature development workflow using the new agent orchestration system.

## Scenario: Adding Metric Caching

**User Goal**: Add LRU caching to avoid recomputing expensive metrics.

**Expected Outcome**:
- Production-ready caching feature
- 100% test coverage
- Comprehensive documentation
- PR ready for review

---

## Phase 1: Planning (~30 minutes)

### User Initiates Request

```
User: "I want to add metric caching to avoid recomputing expensive metrics. The cache should support LRU eviction and optional TTL."
```

### Core Agent Response

```
I'll help you plan the metric caching feature. Let me start by exploring the codebase and designing the solution.

[Launches dqx-plan agent]
```

### dqx-plan Agent Execution

**Step 1: Exploration** (delegated to specialized agents)
```
[Launches dqx-explore agent]
Prompt: "Quick search: Find existing caching patterns in DQX. Look for:
- MetricProvider implementation
- Similar optimization patterns
- Related tests
Return: file paths and key insights."

[Launches dqx-api agent]
Prompt: "Review API design patterns for caching in DQX. Check for:
- Protocol usage patterns
- Error handling approaches
- Integration points
Return: patterns to follow."
```

**Step 2: Design Document Generation**

dqx-plan creates three modular documents:

**`docs/plans/metric_caching_technical_spec.md` (312 lines)**
```markdown
# Metric Caching Technical Specification

## Problem Statement
MetricProvider currently recomputes metrics on every access, which is
inefficient for expensive operations (aggregations, complex transformations).
This leads to slow validation suite execution, especially with multiple
assertions using the same metrics.

## Architecture Decisions

### Decision 1: LRU Cache with TTL
**Rationale**: LRU provides predictable memory usage while TTL handles
data staleness. Combination balances performance and correctness.
**Alternatives considered**:
- Pure LRU (no TTL) - Rejected: can't handle stale data
- Pure TTL - Rejected: unbounded memory growth
- LFU - Rejected: more complex, marginal benefits

### Decision 2: Protocol-based Cache Backend
**Rationale**: Allows custom implementations (Redis, Memcached) without
changing core code. Follows DQX's Protocol pattern.
**Alternatives considered**:
- Hardcoded in-memory only - Rejected: not extensible
- Abstract base class - Rejected: Protocol is more Pythonic

## API Design

### CacheBackend Protocol
```python
@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for cache backend implementations."""

    def get(self, key: str) -> Any | None:
        """Retrieve value or None if not found/expired."""
        ...

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store value with optional TTL in seconds."""
        ...

    def clear(self) -> None:
        """Clear all cached entries."""
        ...
```

{... rest of API design ...}

## Integration Points

### MetricProvider (`src/dqx/provider.py`)
**Changes needed**: Wrap metric computations with cache lookup
**Impact**: Backward compatible - caching is optional

{... rest of spec ...}
```

**`docs/plans/metric_caching_implementation_guide.md` (478 lines)**
```markdown
# Metric Caching Implementation Guide

## Overview
Implement LRU cache with TTL support for MetricProvider metrics.

## Phase Breakdown

### Phase 1: Core Cache Data Structures
**Goal**: Implement LRU cache with TTL support
**Duration**: 1.5 hours

**Files to create**:
- `src/dqx/cache.py` - Cache implementation

**Tests to write**:
```python
def test_lru_cache_creation(): ...
def test_lru_cache_set_and_get(): ...
def test_lru_cache_eviction_on_full(): ...
def test_lru_cache_ttl_expiration(): ...
def test_lru_cache_update_access_order(): ...
def test_lru_cache_clear(): ...
```

**Success criteria**:
- [ ] All tests passing
- [ ] Coverage: 100%
- [ ] Pre-commit: passing

**Commit**: `feat(cache): add core cache data structures`

{... more phases ...}
```

**`docs/plans/metric_caching_context.md` (245 lines)**
```markdown
# Metric Caching Context for Implementation

## DQX Architecture Overview

### MetricProvider (`src/dqx/provider.py`)
- Purpose: Provides metric computation API (num_rows, sum, average, etc.)
- Key methods: See lines 45-120
- How caching relates: Wrap each metric method with cache lookup

### Analyzer (`src/dqx/analyzer.py`)
- Purpose: Converts symbolic expressions to SQL
- How caching relates: Cache key should include SQL query

## Code Patterns to Follow

### Pattern 1: Protocol-based Interfaces
**When to use**: Defining extensible interfaces

**Example from DQX**:
```python
@runtime_checkable
class SqlDataSource(Protocol):
    def execute(self, query: str) -> pa.Table: ...
```

**Apply to caching**: Define CacheBackend as Protocol

{... more patterns ...}
```

### Core Agent Presents Summary

```
Design documents created for metric caching:

📄 Technical Specification: docs/plans/metric_caching_technical_spec.md (312 lines)
   Key decisions:
   - LRU cache with TTL support (balances performance and correctness)
   - Protocol-based backend (extensible for Redis/Memcached)
   - Optional caching (backward compatible)

📋 Implementation Guide: docs/plans/metric_caching_implementation_guide.md (478 lines)
   - 4 phases
   - Estimated time: ~5 hours
   - Phase 1: Core cache data structures (1.5h)
   - Phase 2: Cache backend protocol (1.5h)
   - Phase 3: MetricProvider integration (1.5h)
   - Phase 4: E2E tests + docs (0.5h)

📚 Context Document: docs/plans/metric_caching_context.md (245 lines)
   - 3 DQX components covered
   - 4 code patterns documented
   - 3 common pitfalls highlighted

Total: 1,035 lines across 3 modular documents
(vs typical 1,500+ lines in single monolithic doc)

Ready to proceed with implementation? Or would you like to review/modify the design first?
```

### User Approves Design

```
User: "Looks good, proceed with implementation"
```

---

## Phase 2: Implementation (~5 hours)

### Core Agent Response

```
Starting implementation...

[Launches dqx-implement agent]
```

### dqx-implement Agent Execution

**Phase 1: Core Cache Data Structures**

```
Loading context:
  ✓ Implementation guide - Phase 1 section
  ✓ Technical spec - LRUCache design
  ✓ Context doc - Protocol pattern

Step 1: Writing tests FIRST (TDD)
  Creating tests/test_cache.py:
    ✓ test_lru_cache_creation
    ✓ test_lru_cache_set_and_get
    ✓ test_lru_cache_eviction_on_full
    ✓ test_lru_cache_ttl_expiration
    ✓ test_lru_cache_update_access_order
    ✓ test_lru_cache_clear
    ✓ test_lru_cache_max_size_zero
    ✓ test_lru_cache_none_value

  Running tests (expecting failures):
    $ uv run pytest tests/test_cache.py
    ✗ All tests fail (expected - no implementation yet)

Step 2: Implementing LRUCache
  Creating src/dqx/cache.py:
    ✓ Imports and type hints
    ✓ LRUCache class with __init__
    ✓ set() method with LRU tracking
    ✓ get() method with TTL check
    ✓ clear() method
    ✓ _evict() helper method
    ✓ Complete docstrings (Google style)

  Running tests:
    $ uv run pytest tests/test_cache.py -v
    ✓ All 8 tests passing

Step 3: Quality Gate - Coverage
  $ uv run pytest tests/test_cache.py --cov=src/dqx/cache --cov-report=term-missing
  ✓ Coverage: 100%

Step 4: Quality Gate - Pre-commit
  $ uv run pre-commit run --all-files
  ✓ ruff format: passing
  ✓ ruff check: passing
  ✓ mypy: passing

Step 5: Quality Gate - Full Test Suite
  $ uv run pytest
  ✓ All 847 tests passing (no regressions)

Step 6: Commit
  $ git add src/dqx/cache.py tests/test_cache.py
  $ git commit -m "feat(cache): add core cache data structures"
  ✓ Committed: abc1234

✓ Phase 1/4: Core cache data structures
  • Tests written: 8 tests
  • Implementation: complete
  • All tests: passing
  • Coverage: 100%
  • Pre-commit hooks: passing
  • Committed: abc1234 - feat(cache): add core cache data structures
```

**Phase 2: Cache Backend Protocol**

```
Loading context:
  ✓ Implementation guide - Phase 2 section
  ✓ Technical spec - CacheBackend protocol
  ✓ Context doc - Protocol pattern

Step 1: Writing tests FIRST
  Updating tests/test_cache.py:
    ✓ test_cache_backend_protocol_check
    ✓ test_in_memory_cache_backend_get_miss
    ✓ test_in_memory_cache_backend_get_hit
    ✓ test_in_memory_cache_backend_set
    ✓ test_in_memory_cache_backend_ttl
    ✓ test_in_memory_cache_backend_clear
    {... 6 more tests ...}

  Running new tests:
    $ uv run pytest tests/test_cache.py::test_cache_backend* -v
    ✗ All new tests fail (expected)

Step 2: Implementing CacheBackend Protocol
  Updating src/dqx/cache.py:
    ✓ CacheBackend Protocol definition
    ✓ InMemoryCacheBackend implementation
    ✓ Integration with LRUCache
    ✓ Complete docstrings

  Running tests:
    $ uv run pytest tests/test_cache.py -v
    ✓ All 20 tests passing

Step 3: Quality Gates
  ✓ Coverage: 100%
  ✓ Pre-commit: passing
  ✓ Full test suite: 859 tests passing

Step 4: Commit
  $ git commit -m "feat(cache): implement cache backend protocol"
  ✓ Committed: def5678

✓ Phase 2/4: Cache backend protocol
  • Tests written: 12 tests
  • Implementation: complete
  • All tests: passing (20 total in test_cache.py)
  • Coverage: 100%
  • Pre-commit hooks: passing
  • Committed: def5678 - feat(cache): implement cache backend protocol
```

**Phase 3: MetricProvider Integration**

```
Loading context:
  ✓ Implementation guide - Phase 3 section
  ✓ Technical spec - MetricProvider integration
  ✓ src/dqx/provider.py (existing code)

Step 1: Writing tests FIRST
  Creating tests/test_provider_cache_integration.py:
    ✓ test_metric_provider_with_cache
    ✓ test_metric_provider_cache_hit
    ✓ test_metric_provider_cache_miss
    ✓ test_metric_provider_cache_disabled
    ✓ test_metric_provider_cache_invalidation
    {... 10 more tests ...}

  Running new tests:
    ✗ All fail (expected - no integration yet)

Step 2: Implementing Integration
  Updating src/dqx/provider.py:
    ✓ Add cache parameter to __init__
    ✓ Wrap num_rows() with cache
    ✓ Wrap sum() with cache
    ✓ Wrap average() with cache
    {... all metric methods ...}
    ✓ Add cache key generation
    ✓ Update docstrings

  Updating src/dqx/__init__.py:
    ✓ Export cache classes

  Running tests:
    $ uv run pytest tests/test_provider_cache_integration.py -v
    ✓ All 15 tests passing

Step 3: Quality Gates
  ✓ Coverage: 100%
  ✓ Pre-commit: passing
  ✓ Full test suite: 874 tests passing

Step 4: Commit
  $ git commit -m "feat(cache): integrate caching with MetricProvider"
  ✓ Committed: ghi9012

✓ Phase 3/4: MetricProvider integration
  • Tests written: 15 tests
  • Implementation: complete
  • All tests: passing (35 total for caching)
  • Coverage: 100%
  • Pre-commit hooks: passing
  • Committed: ghi9012 - feat(cache): integrate caching with MetricProvider
```

**Phase 4: E2E Tests + Documentation**

```
Loading context:
  ✓ Implementation guide - Phase 4 section
  ✓ Technical spec - full document

Step 1: Writing E2E tests
  Creating tests/e2e/test_cache_e2e.py:
    ✓ test_verification_suite_with_caching
    ✓ test_cache_performance_improvement
    ✓ test_cache_across_multiple_checks
    {... 3 more tests ...}

  Running E2E tests:
    $ uv run pytest tests/e2e/test_cache_e2e.py -v
    ✓ All 6 tests passing

Step 2: Documentation
  Updating docs/api-reference.md:
    ✓ Added CacheBackend protocol docs
    ✓ Added InMemoryCacheBackend docs
    ✓ Added usage examples

  Updating README.md:
    ✓ Added caching example

Step 3: Quality Gates
  ✓ Coverage: 100%
  ✓ Pre-commit: passing
  ✓ Full test suite: 880 tests passing

Step 4: Commit
  $ git commit -m "docs(cache): add caching documentation and examples"
  ✓ Committed: jkl3456

✓ Phase 4/4: E2E tests + docs
  • Tests written: 6 tests
  • Documentation: updated
  • All tests: passing (41 total for caching)
  • Coverage: 100%
  • Pre-commit hooks: passing
  • Committed: jkl3456 - docs(cache): add caching documentation
```

### Core Agent Presents Summary

```
Implementation complete!

📊 Summary:
• Phases completed: 4/4
• Total tests: 41 (8 + 12 + 15 + 6)
• Coverage: 100%
• Commits created: 4
• All pre-commit hooks: passing
• No regressions: 880 total tests passing

📝 Commit history:
abc1234 feat(cache): add core cache data structures
def5678 feat(cache): implement cache backend protocol
ghi9012 feat(cache): integrate caching with MetricProvider
jkl3456 docs(cache): add caching documentation

Ready to create pull request?
```

### User Approves PR Creation

```
User: "Yes, create PR"
```

---

## Phase 3: PR Creation (~5 minutes)

### Core Agent Response

```
Creating pull request...

[Launches dqx-pr agent]
```

### dqx-pr Agent Execution

```
Step 1: Analyzing feature branch
  ✓ Branch: feature/metric-caching
  ✓ Commits since main: 4
  ✓ Files changed: 6
  ✓ Lines added: +523
  ✓ Lines removed: -12

Step 2: Loading design documents
  ✓ Technical spec: docs/plans/metric_caching_technical_spec.md
  ✓ Implementation guide: docs/plans/metric_caching_implementation_guide.md
  Extracting key information...

Step 3: Verifying quality gates
  $ uv run pytest
  ✓ All 880 tests passing

  $ uv run pytest --cov=src/dqx --cov-report=term
  ✓ Coverage: 100%

  $ uv run pre-commit run --all-files
  ✓ All hooks passing

Step 4: Generating PR description
  ✓ Summary extracted from technical spec
  ✓ Architecture decisions documented
  ✓ Phase breakdown with commit references
  ✓ Test verification included
  ✓ Design doc links added

Step 5: Creating PR
  $ git push -u origin feature/metric-caching
  ✓ Pushed to remote

  $ gh pr create --title "feat(cache): add LRU cache with TTL support" --body "..."
  ✓ PR created: #123

Step 6: PR created successfully
  URL: https://github.com/user/dqx/pull/123
```

### Core Agent Presents PR Details

```
Pull request created successfully!

🔗 PR URL: https://github.com/user/dqx/pull/123

📊 Summary:
• Commits: 4
• Files changed: 6
• Tests added: 41
• Coverage: 100%

📁 Design docs: 3 files in docs/plans/
• Technical spec: metric_caching_technical_spec.md
• Implementation guide: metric_caching_implementation_guide.md
• Context doc: metric_caching_context.md

✅ Quality gates:
• All tests passing (880 total)
• Coverage: 100%
• Pre-commit hooks: passing

🔄 Next steps:
1. CodeRabbit will automatically review the PR
2. Address any feedback with targeted commits
3. Request human review when ready
4. Merge after approval

The PR description includes links to all design documents for reviewers.
```

---

## Phase 4: Feedback Iteration (~1-2 hours)

### CodeRabbit Reviews PR

CodeRabbit posts 8 review comments:
- 2 P0 (blockers): Missing type hint, missing test
- 4 P1 (important): Naming improvements, docstring additions
- 2 P2 (nice-to-have): Refactoring suggestions

### User Requests Feedback Resolution

```
User: "Address CodeRabbit feedback"
```

### Core Agent Response

```
Analyzing review comments...

[Launches dqx-feedback agent]
```

### dqx-feedback Agent Execution

```
Step 1: Fetching comments
  $ gh pr view 123 --json comments
  ✓ Fetched 8 review comments

Step 2: Grouping and prioritizing
  ✓ P0 (blockers): 2 comments
  ✓ P1 (important): 4 comments
  ✓ P2 (nice-to-have): 2 comments

Step 3: Addressing P0 comments

Comment #1 (P0): src/dqx/cache.py:78
  Issue: "Missing return type hint for evict method"

  Loading minimal context:
    ✓ src/dqx/cache.py

  Fix:
    Updated line 78:
    - def evict(self):
    + def evict(self) -> tuple[str, Any]:

  Verification:
    $ uv run mypy src/dqx/cache.py
    ✓ Type checking passing

    $ uv run pytest tests/test_cache.py -v
    ✓ All tests passing

  Commit:
    $ git commit -m "fix: add type hint for evict method"
    ✓ Committed: aaa1111

Comment #2 (P0): tests/test_cache.py:120
  Issue: "Add test for cache full with TTL edge case"

  Loading minimal context:
    ✓ tests/test_cache.py
    ✓ src/dqx/cache.py (lines 45-90)

  Fix:
    Added test at line 120:
    def test_cache_full_with_expired_entries(self) -> None:
        """Test eviction considers expired entries."""
        cache = LRUCache(max_size=2)
        cache.set("a", 1, ttl=0.1)
        time.sleep(0.2)  # Let "a" expire
        cache.set("b", 2)
        cache.set("c", 3)  # Should NOT evict "b"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

  Verification:
    $ uv run pytest tests/test_cache.py::test_cache_full_with_expired_entries -v
    ✓ New test passing

  Commit:
    $ git commit -m "test: add cache full with TTL edge case"
    ✓ Committed: bbb2222

✓ P0 issues: 2/2 fixed

Step 4: Addressing P1 comments

{Similar process for 4 P1 comments...}

✓ P1 issues: 4/4 fixed

Step 5: Addressing P2 comments

{Similar process for 2 P2 comments...}

✓ P2 issues: 2/2 fixed

Step 6: Final verification
  $ uv run pytest
  ✓ All 882 tests passing (2 new tests added)

  $ uv run pytest --cov=src/dqx --cov-report=term
  ✓ Coverage: 100%

  $ uv run pre-commit run --all-files
  ✓ All hooks passing

Step 7: Pushing fixes
  $ git push
  ✓ Pushed 8 fix commits

Step 8: Commenting on PR
  $ gh pr comment 123 --body "..."
  ✓ Posted summary comment
```

### Core Agent Presents Summary

```
Review feedback addressed successfully!

📊 Summary:
• Total comments: 8
• P0 (blockers): 2 - All fixed ✓
• P1 (important): 4 - All fixed ✓
• P2 (nice-to-have): 2 - All fixed ✓

📝 Changes made:
src/dqx/cache.py (4 changes)
  • Added type hint for evict method
  • Improved variable naming in _evict()
  • Added docstring for _is_expired()
  • Refactored TTL check logic

tests/test_cache.py (3 changes)
  • Added cache full with TTL edge case test
  • Added integration test for cache invalidation
  • Added performance benchmark test

docs/plans/metric_caching_technical_spec.md (1 change)
  • Clarified TTL behavior in decision section

✅ Quality gates:
• Full test suite: passing (882 tests, +2 new)
• Coverage: 100%
• Pre-commit hooks: passing

💾 Commits pushed: 8
aaa1111 - fix: add type hint for evict method
bbb2222 - test: add cache full with TTL edge case
ccc3333 - style: improve variable naming in evict logic
ddd4444 - docs: add docstring for is_expired helper
eee5555 - refactor: simplify TTL check logic
fff6666 - test: add cache invalidation integration test
ggg7777 - test: add performance benchmark for caching
hhh8888 - docs: clarify TTL behavior in spec

Ready for next review cycle!
```

---

## Phase 5: Merge & Completion

### Human Reviewer Approves

After reviewing the PR and all feedback responses, human reviewer approves.

```
User: "Merge the PR"
```

### User Merges

```
$ gh pr merge 123 --squash
✓ Merged PR #123 into main

Feature complete! Metric caching is now available in DQX.
```

---

## Workflow Benefits Demonstrated

### Context Management
- **Planning**: 1,035 lines across 3 docs (vs 1,500+ monolithic)
- **Implementation**: Only ~300-500 lines loaded per phase
- **Feedback**: Only relevant files loaded per comment
- **Total context reduction**: ~60%

### Automation
- **Manual interventions**: 5 (approve design, approve impl, approve PR, request feedback, merge)
- **Automated steps**: 40+ (tests, coverage, commits, quality gates, fixes)
- **Automation rate**: ~90%

### Quality Assurance
- **Coverage**: 100% throughout (enforced automatically)
- **Pre-commit**: Passed after every phase
- **Regressions**: 0 (verified after each phase)
- **Code style**: Consistent (auto-formatted)

### Time Efficiency
- **Planning**: 30 min (including exploration)
- **Implementation**: 5 hours (4 phases, automated)
- **PR creation**: 5 min (fully automated)
- **Feedback**: 1-2 hours (8 comments addressed)
- **Total**: ~7 hours for production-ready feature

### Documentation
- **Design docs**: 3 comprehensive documents
- **PR description**: Links to all design context
- **Code comments**: Complete docstrings
- **Examples**: Added to API docs and README

---

## Comparison: Old vs New Workflow

### Old Workflow (Manual)
```
Time: ~12 hours
Context: Monolithic design doc (1,500+ lines)
Iterations: Multiple back-and-forth for quality issues
Coverage: Often 95-98% (manual verification)
Commits: Often lumped together
Documentation: Written after implementation
```

### New Workflow (Automated)
```
Time: ~7 hours
Context: Modular docs (3 × 300-400 lines)
Iterations: Minimal (quality enforced automatically)
Coverage: Always 100% (automated verification)
Commits: Atomic, one per phase
Documentation: Generated during implementation
```

**Improvement**: ~40% time reduction, higher quality, better documentation.
