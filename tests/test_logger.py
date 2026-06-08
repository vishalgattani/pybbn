"""Tests for pybbn_assurance/logger.py — custom logging."""

import logging
import pytest

from pybbn_assurance.logger import (
    CustomFormatter,
    ExcludeSpecificLogsFilter,
    logger,
    excluded_strings,
)


class TestCustomFormatter:
    """Tests for the custom colored formatter."""

    def test_format_debug(self):
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="test.py",
            lineno=1, msg="debug msg", args=None, exc_info=None,
        )
        result = formatter.format(record)
        assert "debug msg" in result

    def test_format_info(self):
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="info msg", args=None, exc_info=None,
        )
        result = formatter.format(record)
        assert "info msg" in result
        assert "[INFO]" in result

    def test_format_warning(self):
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="test.py",
            lineno=1, msg="warn msg", args=None, exc_info=None,
        )
        result = formatter.format(record)
        assert "warn msg" in result

    def test_format_error(self):
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="test.py",
            lineno=1, msg="error msg", args=None, exc_info=None,
        )
        result = formatter.format(record)
        assert "error msg" in result

    def test_format_critical(self):
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name="test", level=logging.CRITICAL, pathname="test.py",
            lineno=1, msg="critical msg", args=None, exc_info=None,
        )
        result = formatter.format(record)
        assert "critical msg" in result


class TestExcludeSpecificLogsFilter:
    """Tests for the log exclusion filter."""

    def test_filter_passes_normal_log(self):
        f = ExcludeSpecificLogsFilter(["DEBUG:docker"])
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="test.py",
            lineno=1, msg="normal message", args=None, exc_info=None,
        )
        assert f.filter(record) is True

    def test_filter_excludes_matched_log(self):
        f = ExcludeSpecificLogsFilter(["DEBUG:docker"])
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="test.py",
            lineno=1, msg="DEBUG:docker something", args=None, exc_info=None,
        )
        assert f.filter(record) is False

    def test_filter_excludes_matplotlib(self):
        f = ExcludeSpecificLogsFilter(["DEBUG:matplotlib"])
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="test.py",
            lineno=1, msg="DEBUG:matplotlib figure created", args=None, exc_info=None,
        )
        assert f.filter(record) is False

    def test_filter_with_multiple_strings(self):
        f = ExcludeSpecificLogsFilter(["DEBUG:docker", "DEBUG:urllib3"])
        record1 = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="test.py",
            lineno=1, msg="DEBUG:urllib3 connection", args=None, exc_info=None,
        )
        assert f.filter(record1) is False

    def test_default_excluded_strings(self):
        assert "DEBUG:docker" in excluded_strings
        assert "DEBUG:matplotlib" in excluded_strings
        assert "DEBUG:urllib3" in excluded_strings
        assert "DEBUG:trimesh" in excluded_strings


class TestLogger:
    """Tests for the module-level logger."""

    def test_logger_exists(self):
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handlers(self):
        assert len(logger.handlers) > 0

    def test_logger_level_is_debug(self):
        assert logger.level == logging.DEBUG
