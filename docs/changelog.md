# Changelog

All notable changes to DQX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial documentation structure with MkDocs
- Comprehensive CI/CD pipeline with GitHub Actions
- CodeRabbit AI code review integration
- Dependabot dependency management
- Release Drafter for automated release notes

### Changed
- Enhanced documentation with user guides and API reference

### Fixed
- Navigation warnings in MkDocs configuration

## [0.3.0] - 2024-10-27

### Added
- Core validation engine with fluent API
- Built-in checks for completeness, consistency, accuracy, and uniqueness
- Support for multiple data sources (Pandas, SQL, files)
- Plugin system for custom extensions
- Comprehensive test suite with 100% coverage

### Changed
- Migrated from DQGuard architecture to DQX
- Improved performance with parallel processing
- Enhanced error messages and debugging

### Fixed
- Memory optimization for large datasets
- Connection pooling for database sources

## [0.2.0] - 2024-09-15

### Added
- Statistical validation checks
- Cross-column validation support
- Validation pipeline functionality
- HTML report generation

### Changed
- Refactored check system for better extensibility
- Improved API consistency

### Deprecated
- Legacy validation methods (will be removed in 0.4.0)

## [0.1.0] - 2024-08-01

### Added
- Basic validation framework
- Support for null checks and range validation
- Simple reporting functionality
- Initial documentation

[Unreleased]: https://github.com/yourusername/dqx/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/yourusername/dqx/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yourusername/dqx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/dqx/releases/tag/v0.1.0

---

For detailed release notes, see the [GitHub Releases](https://github.com/yourusername/dqx/releases) page.
