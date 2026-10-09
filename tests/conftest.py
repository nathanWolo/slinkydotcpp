"""Shared pytest fixtures: the Go rules-engine oracle (see tests/oracle.py)."""

from __future__ import annotations

import pytest

try:  # tests/ on sys.path (rootdir conftest, no tests/__init__.py)
    from oracle import OracleStepper, OracleUnavailable, build_oracle
except ImportError:  # tests/ imported as a package
    from tests.oracle import OracleStepper, OracleUnavailable, build_oracle


@pytest.fixture(scope="session")
def oracle_bin():
    """Path to the built oracle binary; skips the test if it can't be built."""
    try:
        return build_oracle()
    except OracleUnavailable as e:
        pytest.skip(f"rules-engine oracle unavailable: {e}")


@pytest.fixture(scope="session")
def oracle(oracle_bin):
    """A persistent ``oracle step`` process shared by the whole session."""
    with OracleStepper(oracle_bin) as stepper:
        yield stepper
