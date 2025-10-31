# Changelog

All notable changes to DQX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.5] - 2024-10-30

### Added
- CommercialDataSource with date filtering support (#17)
- Metric expiration methods to MetricDB (#15)
- Standardized CI workflows and GitHub settings (#9)

### Fixed
- Resolve deprecation warning for invalid escape sequence (#16)
- Remove unnecessary lambda from UUID default value in ORM (#14)
- Address Python special method protocol violation in test (#11)
- Resolve GitHub release character limit issue (#8)

## [0.5.4] - 2024-10-29

### Fixed
- Resolve GitHub release character limit issue
- Resolve release workflow issues and update dqlib documentation (#7)

## [0.5.3] - 2024-10-29

### Fixed
- Resolve release workflow issues and update dqlib documentation

## [0.5.2] - 2024-10-29

### Fixed
- Remove docker-publish job from release workflow (#5)

## [0.5.1] - 2024-10-29

### Fixed
- Remove docker-publish job from release workflow

## [0.5.0] - 2024-10-29

### Added
- `__version__` attribute to dqx module

### Fixed
- Test-release job to handle PEP 668 system Python protection
- Remove redundant pull_request_target from release-drafter workflow
- Add pull-requests write permission to docs workflow
- Correct codecov action parameter from 'file' to 'files'
- Add --system flag to release workflow test installation

[Unreleased]: https://github.com/nampham2/dqx/compare/v0.5.5...HEAD
[0.5.5]: https://github.com/nampham2/dqx/compare/v0.5.4...v0.5.5
[0.5.4]: https://github.com/nampham2/dqx/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/nampham2/dqx/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/nampham2/dqx/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/nampham2/dqx/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/nampham2/dqx/releases/tag/v0.5.0

---

For detailed release notes, see the [GitHub Releases](https://github.com/nampham2/dqx/releases) page.
