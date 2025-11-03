"""Tests for dqx version information."""

import re


def test_dqx_has_version() -> None:
    """Test that dqx module has __version__ attribute."""
    import dqx

    assert hasattr(dqx, "__version__")
    assert isinstance(dqx.__version__, str)


def test_version_format() -> None:
    """Test that version follows semantic versioning format."""
    import dqx

    # Match semantic version: major.minor.patch with optional pre-release (alpha, beta, rc) or dev version
    version_pattern = re.compile(
        r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"  # major.minor.patch
        r"(?:(?:a|b|rc)\d+)?(?:\.dev)?$"  # optional alpha/beta/rc + optional .dev
    )
    assert version_pattern.match(dqx.__version__) is not None


def test_version_not_empty() -> None:
    """Test that version string is not empty."""
    import dqx

    assert dqx.__version__
    assert len(dqx.__version__) > 0
