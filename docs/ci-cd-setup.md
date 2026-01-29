# CI/CD Setup Guide for DQX

This guide documents the complete CI/CD setup for the DQX project, including all required secrets and configurations.

## Related Documentation

- **[GitHub CI/CD Setup Guide](./github-cicd-setup-guide.md)** - Step-by-step setup instructions
- **[GitHub CI/CD Operations Guide](./github-cicd-operations-guide.md)** - Daily operations and usage

This document provides a technical overview and reference. For practical guides, see the documents above.

## GitHub Actions Workflows

### 1. Test & Coverage Workflow
**File**: `.github/workflows/test.yml`
- Runs on: Push to main/develop, pull requests
- Tests multiple Python versions (3.11, 3.12, 3.13)
- Generates coverage reports and badges
- Includes type coverage reporting

### 2. Documentation Workflow
**File**: `.github/workflows/docs.yml`
- Builds MkDocs documentation
- Tests documentation build
- Publishes to GitHub Pages (optional)

### 3. Release Workflow
**File**: `.github/workflows/release.yml`
- Triggers on: GitHub releases
- Builds Python packages (wheel and sdist)
- Publishes to PyPI
- Creates GitHub release artifacts

### 4. Security Scanning
**File**: `.github/workflows/codeql.yml`
- Runs CodeQL analysis
- Performs dependency review on PRs
- Runs pip-audit for vulnerability scanning

### 5. Example Validation
**File**: `.github/workflows/examples.yml`
- Validates all example scripts
- Ensures examples stay up-to-date

## Required Secrets

### PyPI Publishing
- **`PYPI_API_TOKEN`**: PyPI API token for package publishing
  - Create at: https://pypi.org/manage/account/token/
  - Scope: Can upload to project "dqx"

### Test PyPI (Optional)
- **`TEST_PYPI_API_TOKEN`**: Test PyPI API token
  - Create at: https://test.pypi.org/manage/account/token/
  - Use for testing releases before production

### GitHub Pages (Optional)
- **`GITHUB_TOKEN`**: Automatically provided by GitHub Actions
  - No setup required
  - Used for publishing documentation

### Google Analytics (Optional)
- **`GOOGLE_ANALYTICS_KEY`**: Google Analytics tracking ID
  - Format: `G-XXXXXXXXXX`
  - Used in MkDocs for documentation analytics

## External Services

### 1. ReadTheDocs
**Configuration**: `.readthedocs.yml`
- Sign up at: https://readthedocs.org
- Import your GitHub repository
- Documentation will be available at: https://dqx.readthedocs.io

### 2. CodeRabbit
**Configuration**: `.coderabbit.yaml`
- Install from: https://github.com/marketplace/coderabbit
- Provides AI-powered code reviews
- No additional configuration needed

### 3. Dependabot
**Configuration**: `.github/dependabot.yml`
- Automatically enabled for GitHub repositories
- Creates PRs for dependency updates
- Review and merge security updates promptly

### 4. Release Drafter
**Configuration**: `.github/release-drafter.yml`
- Automatically creates draft releases
- Categorizes changes based on PR labels
- Updates release notes with each merge to main

## Setting Up Secrets

### Via GitHub Web Interface
1. Go to Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each secret with its name and value

### Via GitHub CLI
```bash
# Install GitHub CLI
brew install gh  # macOS
# or visit: https://cli.github.com/

# Authenticate
gh auth login

# Add secrets
gh secret set PYPI_API_TOKEN --body="pypi-..."
gh secret set TEST_PYPI_API_TOKEN --body="pypi-..."
```

## Workflow Permissions

Ensure workflows have proper permissions:

1. Go to Settings → Actions → General
2. Under "Workflow permissions", select:
   - "Read and write permissions"
   - "Allow GitHub Actions to create and approve pull requests"

## Branch Protection

Recommended branch protection rules for `main`:

1. Require pull request reviews
2. Require status checks:
   - `test (3.11)` - Main test suite
   - `analyze` - CodeQL security
   - `docs` - Documentation build
3. Include administrators
4. Allow force pushes (only for admins)

## Local Development

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
uv run pre-commit run --all-files

# Skip specific hooks
SKIP=mypy git commit -m "..."
```

### Testing Workflows Locally
```bash
# Install act (GitHub Actions local runner)
brew install act  # macOS

# Run specific workflow
act -W .github/workflows/test.yml

# Run with secrets
act -W .github/workflows/release.yml --secret-file .secrets
```

## Monitoring & Maintenance

### GitHub Actions Dashboard
- Monitor at: `https://github.com/<owner>/dqx/actions`
- Set up notifications for failed workflows
- Review workflow run times and optimize if needed

### Dependency Updates
- Review Dependabot PRs weekly
- Run security audits: `uv run pip-audit`
- Keep GitHub Actions versions updated

### Documentation
- Verify ReadTheDocs builds after each release
- Check for broken links: `mkdocs serve --strict`
- Update screenshots and examples regularly

## Troubleshooting

### Common Issues

1. **PyPI Upload Fails**
   - Verify API token has upload permissions
   - Check package version isn't already published
   - Ensure package builds locally: `uv build`

2. **Documentation Build Fails**
   - Check MkDocs configuration: `mkdocs build --strict`
   - Verify all referenced files exist
   - Check for Python version compatibility

3. **Tests Fail on CI but Pass Locally**
   - Check for environment differences
   - Verify all test dependencies are installed
   - Look for timing-dependent tests

4. **Coverage Reports Not Updating**
   - Ensure coverage files are generated
   - Check GitHub token permissions
   - Verify badge URLs are correct

## Release Process

1. **Prepare Release**
   ```bash
   # Update version in pyproject.toml
   # Update CHANGELOG.md
   git commit -m "chore: prepare release v0.4.0"
   git tag v0.4.0
   git push origin main --tags
   ```

2. **Create GitHub Release**
   - Go to Releases → Draft a new release
   - Choose tag `v0.4.0`
   - Release notes are auto-generated
   - Publish release

3. **Verify Deployment**
   - Check PyPI: https://pypi.org/project/dqx/
   - Verify docs: https://dqx.readthedocs.io
   - Test installation: `pip install dqx==0.4.0`

## Security Considerations

- Rotate API tokens annually
- Use environment-specific secrets
- Enable 2FA on PyPI and GitHub accounts
- Review security alerts promptly
- Keep workflows up-to-date

## Contact & Support

For CI/CD issues:
- Check GitHub Actions logs
- Review workflow configuration
- Open an issue with the `ci` label
- Tag @phamducnam for urgent issues
