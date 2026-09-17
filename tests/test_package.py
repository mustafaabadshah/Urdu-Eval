"""Test basic package properties."""

import urdu_eval


def test_package_version() -> None:
    """Verify package version is set to 0.1.0."""
    assert urdu_eval.__version__ == "0.1.0"


def test_package_metadata() -> None:
    """Verify package metadata attributes."""
    assert urdu_eval.__license__ == "Apache-2.0"
    assert "UrduEval Contributors" in urdu_eval.__author__
