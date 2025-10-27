# Installation

DQX can be installed using pip from PyPI or directly from source.

## Requirements

- Python 3.11 or higher
- pip package manager

## Install from PyPI

The simplest way to install DQX:

```bash
pip install dqx
```

For specific version:

```bash
pip install dqx==0.3.0
```

## Install from Source

To install the latest development version:

```bash
git clone https://github.com/yourusername/dqx.git
cd dqx
pip install -e .
```

## Install with Extras

DQX provides optional dependencies for specific features:

### Development Dependencies

For contributing to DQX:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest for testing
- ruff for linting
- mypy for type checking
- pre-commit hooks

### Documentation Dependencies

For building documentation:

```bash
pip install -e ".[docs]"
```

### All Dependencies

To install all optional dependencies:

```bash
pip install -e ".[dev,docs]"
```

## Using UV (Recommended for Development)

DQX uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Install uv
pip install uv

# Clone and setup
git clone https://github.com/yourusername/dqx.git
cd dqx

# Install all dependencies
uv sync

# Run tests
uv run pytest
```

## Verify Installation

After installation, verify DQX is working:

```python
import dqx

print(dqx.__version__)
```

Or from command line:

```bash
python -c "import dqx; print(dqx.__version__)"
```

## Troubleshooting

### Python Version

DQX requires Python 3.11+. Check your version:

```bash
python --version
```

### Virtual Environment

We recommend using a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate     # On Windows

# Install DQX
pip install dqx
```

### Permission Issues

If you encounter permission errors:

```bash
pip install --user dqx
```

### Dependency Conflicts

If you have dependency conflicts, try installing in a clean environment:

```bash
python -m venv clean_env
source clean_env/bin/activate
pip install dqx
```

## Next Steps

- Follow the [Quick Start Guide](quickstart.md) to create your first data quality checks
- Read the [User Guide](user-guide.md) for detailed usage instructions
- Check out [examples](https://github.com/yourusername/dqx/tree/main/examples) for real-world usage

## Uninstalling

To remove DQX:

```bash
pip uninstall dqx
```

---

For issues with installation, please [open an issue](https://github.com/yourusername/dqx/issues) on GitHub.
