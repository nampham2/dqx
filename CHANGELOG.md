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
