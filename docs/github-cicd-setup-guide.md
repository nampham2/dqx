# GitHub CI/CD Setup Guide

This guide walks through setting up GitHub Actions CI/CD for the DQX project.

## Prerequisites

- GitHub account with repository access
- PyPI account for package publishing
- Basic understanding of GitHub Actions

## Step 1: Create PyPI Account and API Token

### Create PyPI Account

1. Visit [pypi.org](https://pypi.org)
2. Click "Register"
3. Fill the registration form:
   - Username: Choose a unique identifier
   - Email: Use your primary email
   - Password: Create a strong password
4. Verify your email address

### Generate API Token

1. Log into PyPI
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Configure the token:
   - Token name: `github-actions-dqx`
   - Scope: "Project: dqx" (or "Entire account" for first release)
   - Click "Add token"
5. **Copy the token immediately** - it shows only once
   - Format: `pypi-AgEIcHlwaS5vcmcCJGE4ZjY5...`
   - Save it securely

### Set Up Test PyPI (Optional but Recommended)

1. Visit [test.pypi.org](https://test.pypi.org)
2. Create a separate account (can use same email)
3. Generate API token following same steps
4. Use for testing releases before production

## Step 2: Configure GitHub Repository Secrets

### Add PyPI Token

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add the secret:
   - Name: `PYPI_API_TOKEN`
   - Value: Paste your PyPI token
   - Click "Add secret"

### Add Test PyPI Token (Optional)

Repeat the above with:
- Name: `TEST_PYPI_API_TOKEN`
- Value: Your Test PyPI token

### Verify Secrets

Your secrets page should show:
- `PYPI_API_TOKEN` (updated just now)
- `TEST_PYPI_API_TOKEN` (updated just now)

## Step 3: Set Up ReadTheDocs

### Create Account

1. Visit [readthedocs.org](https://readthedocs.org)
2. Sign up with GitHub OAuth (recommended) or create account
3. Authorize ReadTheDocs to access your repositories

### Import Project

1. Click "Import a Project"
2. Select your repository from the list
   - Or click "Import Manually" and enter repo URL
3. Configure project details:
   - Name: `dqx`
   - Repository URL: `https://github.com/yourusername/dqx`
   - Repository type: Git
   - Default branch: `main`
4. Click "Next"

### Configure Build

1. In project dashboard, go to Admin → Advanced Settings
2. Set Python configuration:
   - Python interpreter: CPython 3.11
   - Install method: pip
   - Requirements file: Leave empty (using pyproject.toml)
3. Enable these options:
   - Build pull requests
   - Public versions
   - Single version docs (if you want /latest/ only)
4. Save changes

### First Build

1. Go to Builds tab
2. Click "Build Version"
3. Monitor the build log
4. Once successful, view docs at `https://dqx.readthedocs.io`

### Webhook Setup (Automatic)

ReadTheDocs automatically creates a webhook in your GitHub repo. Verify:
1. GitHub repo → Settings → Webhooks
2. Look for ReadTheDocs webhook
3. Should trigger on push events

## Step 4: Install GitHub Apps

### CodeRabbit

1. Visit [GitHub Marketplace - CodeRabbit](https://github.com/marketplace/coderabbit)
2. Click "Set up a plan"
3. Choose pricing plan (free tier available)
4. Select repositories:
   - Choose "Only select repositories"
   - Select your DQX repository
5. Complete installation

CodeRabbit will automatically review PRs once installed.

### Dependabot (Already Enabled)

Dependabot is automatically available. Ensure it's active:
1. Go to Settings → Security & analysis
2. Enable:
   - Dependency graph
   - Dependabot alerts
   - Dependabot security updates

### Release Drafter

The Release Drafter workflow is already configured. It will:
- Auto-generate release notes from PRs
- Categorize changes by labels
- Create draft releases automatically

## Step 5: Configure GitHub Actions Permissions

### Repository Permissions

1. Go to Settings → Actions → General
2. Under "Actions permissions":
   - Select "Allow all actions and reusable workflows"
3. Under "Workflow permissions":
   - Select "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"
4. Click "Save"

### Branch Protection

1. Go to Settings → Branches
2. Click "Add rule"
3. Configure protection for `main`:
   - Branch name pattern: `main`
   - Enable:
     - Require pull request before merging
     - Require status checks:
       - `test (3.11)`
       - `analyze`
       - `docs`
     - Require branches to be up to date
     - Include administrators (optional)
   - Click "Create"

## Step 6: Test the Setup

### Verify Workflows

1. Create a test branch:
   ```bash
   git checkout -b test/ci-setup
   ```

2. Make a small change:
   ```bash
   echo "# Test" >> test.md
   git add test.md
   git commit -m "test: verify CI setup"
   git push origin test/ci-setup
   ```

3. Create a pull request
4. Watch the checks run:
   - Tests should pass
   - CodeRabbit should comment
   - All status checks should be green

### Test Documentation Build

1. Make a docs change:
   ```bash
   echo "Test content" >> docs/test.md
   git add docs/test.md
   git commit -m "docs: test documentation build"
   git push
   ```

2. Check ReadTheDocs:
   - Go to your project builds
   - Verify build triggered and passed

### Test Release Process (Dry Run)

1. Update version in `pyproject.toml`
2. Create and push a tag:
   ```bash
   git tag v0.3.1-rc1
   git push origin v0.3.1-rc1
   ```

3. Check Actions tab for release workflow
4. Verify it would publish correctly (without creating release)

## Step 7: Local Documentation Development

### Install MkDocs

```bash
# In your project directory
uv pip install mkdocs mkdocs-material mkdocstrings[python]
```

### Run Documentation Server

```bash
# Start local server
mkdocs serve

# Output:
# INFO - Building documentation...
# INFO - Cleaning site directory
# INFO - Documentation built in 2.34 seconds
# INFO - Serving on http://127.0.0.1:8000
```

### Live Development

1. Open browser to `http://localhost:8000`
2. Edit any markdown file in `docs/`
3. Save the file
4. Browser auto-refreshes with changes

### Build Documentation

```bash
# Build static site
mkdocs build

# Output creates site/ directory
# Upload site/ contents to any web server
```

### Test Strict Mode

```bash
# Catch broken links and references
mkdocs build --strict

# Fails on warnings - same as CI
```

### Preview Different Themes

Edit `mkdocs.yml`:
```yaml
theme:
  name: material
  palette:
    scheme: slate  # Try 'default' for light
```

Save and see instant changes.

## Troubleshooting

### PyPI Upload Fails

**Token scope too narrow:**
- First release needs "Entire account" scope
- After first release, can limit to project

**Version already exists:**
- PyPI never allows reusing versions
- Increment version in `pyproject.toml`

**Authentication failed:**
- Verify secret name is exactly `PYPI_API_TOKEN`
- Check token starts with `pypi-`
- Regenerate token if needed

### ReadTheDocs Build Fails

**Import error:**
- Check `.readthedocs.yml` syntax
- Verify Python version matches project

**MkDocs not found:**
- Ensure MkDocs installed in build commands
- Check pip install succeeds in logs

**Theme missing:**
- Add `mkdocs-material` to dependencies
- Clear build cache and retry

### GitHub Actions Timeout

**Long test runs:**
- Add `timeout-minutes: 30` to job
- Split tests into parallel jobs
- Cache dependencies

**Hanging process:**
- Check for infinite loops
- Add proper test timeouts
- Kill subprocess in tests

## Security Best Practices

1. **Rotate tokens annually**
   - Set calendar reminder
   - Update both PyPI and GitHub

2. **Use environment protection**
   - Limit production deployments
   - Require reviews for releases

3. **Enable 2FA everywhere**
   - GitHub (required for Actions)
   - PyPI (highly recommended)
   - ReadTheDocs

4. **Monitor security alerts**
   - Check GitHub Security tab weekly
   - Act on Dependabot alerts quickly
   - Review CodeQL findings

## Next Steps

With CI/CD configured:

1. **Make your first release**
   - Update version and changelog
   - Create GitHub release
   - Monitor PyPI publication

2. **Customize workflows**
   - Add deployment environments
   - Include performance tests
   - Add container builds

3. **Enhance documentation**
   - Add API references
   - Include tutorials
   - Create video guides

4. **Monitor metrics**
   - Track build times
   - Review test coverage
   - Analyze deployment frequency

---

For issues, check workflow logs first, then consult this guide's troubleshooting section.

## Related Documentation

- [GitHub CI/CD Operations Guide](./github-cicd-operations-guide.md) - Daily operations and interactions
- [CI/CD Overview](./ci-cd-setup.md) - Technical reference and file descriptions
