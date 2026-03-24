"""Tests for CLI utilities and download exceptions."""

from pathlib import Path

from wildfire_pipeline.cli import _resolve_config
from wildfire_pipeline.gee.download import TooManyFailuresError


class TestResolveConfig:
    def test_absolute_path_returned_as_is(self):
        p = Path("/absolute/path/config.json")
        result = _resolve_config(p)
        assert result == p

    def test_relative_path_resolved_from_repo_root(self):
        p = Path("config/fires.json")
        result = _resolve_config(p)
        assert result.is_absolute()
        assert str(result).endswith("config/fires.json")

    def test_current_dir_relative(self):
        p = Path("./config.json")
        result = _resolve_config(p)
        assert result.is_absolute()


class TestTooManyFailuresError:
    def test_message_formatting(self):
        err = TooManyFailuresError(failed=5, total=10, rate=0.5, hours=[0, 2, 4, 6, 8])
        assert "5/10" in str(err)
        assert "50.0%" in str(err)
        assert "[0, 2, 4, 6, 8]" in str(err)

    def test_attributes(self):
        err = TooManyFailuresError(failed=3, total=20, rate=0.15, hours=[1, 5, 9])
        assert err.failed == 3
        assert err.total == 20
        assert err.rate == 0.15
        assert err.hours == [1, 5, 9]

    def test_is_exception(self):
        err = TooManyFailuresError(failed=1, total=10, rate=0.1, hours=[5])
        assert isinstance(err, Exception)
