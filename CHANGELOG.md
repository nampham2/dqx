## v0.5.11 (2025-12-26)

### Feat

- **api**: add tags support to assertions (#34)
- **profiles**: add metric_multiplier support for assertion overrides (#35)
- add YAML/JSON configuration support for verification suites (#36)

### Docs

- add DQL language design specification (#37)

### Refactor

- remove BigQuery e2e test and ERROR assertion status (#33)

## v0.5.10 (2025-11-11)

### Feat

- **ops**: add CustomSQL operation with universal parameter support (#29)
- **date-exclusion**: implement comprehensive date exclusion with data availability tracking (#28)
- **bkng-integration**: refactor logger API and enhance type safety for integration (#27)

## v0.5.9 (2025-11-04)

### Fix

- **tests**: remove format_string parameter from logger tests
- **plugins**: enhance audit plugin display formatting and add logging support
- correct DoD/WoW calculations to use percentage change (#26)
- enforce string-only values in Tags type
- **tests**: consolidate logger tests and update to use setup_logger

## v0.5.8 (2025-11-03)

### Fix

- correct DoD/WoW calculations to use percentage change
- BigQuery SQL generation compatibility
- BigQuery SQL generation compatibility fixes

## v0.5.7 (2025-11-03)

### Perf

- **analyzer**: optimize SQL logging and formatting

## v0.5.7a4 (2025-11-03)

### Fix

- **analyzer**: ensure correct date alignment in analyze_sql_ops
- **analyzer**: handle BigQuery dict format in batch query results

## v0.5.7a1 (2025-11-03)

### Fix

- **analyzer**: handle BigQuery uppercase KEY in batch query results

### Refactor

- **analyzer**: consolidate SQL ops analysis into single batch function

## v0.5.7a0 (2025-11-03)

### Fix

- correct package name in version import
- resolve BigQuery UNION ALL incompatibility in batch optimization

### Refactor

- remove backward compatibility in analyzer
- remove numpy dependency and cleanup project structure (#24)

## v0.5.6 (2025-11-03)

### Fix

- improve metric handling and analyzer architecture (#19)

### Refactor

- remove numpy dependency from analyzer
- **cache**: enhance metric storage and add performance tracking (#23)
- **cache**: enhance metric storage and add performance tracking

### Perf

- optimize metric cache and improve code quality (#20)

## v0.5.5 (2025-10-30)

### Feat

- Add CommercialDataSource with date filtering support (#17)
- add metric expiration methods to MetricDB (#15)
- **ci**: standardize workflows and add GitHub settings (#9)

### Fix

- resolve deprecation warning for invalid escape sequence (#16)
- **orm**: remove unnecessary lambda from uuid default value (#14)
- address Python special method protocol violation in test (#11)
- resolve GitHub release character limit issue (#8)

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
