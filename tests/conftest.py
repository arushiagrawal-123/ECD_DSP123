import os
from pathlib import Path

import pytest

from ecg_monitor.acquisition.loader import load_record
from ecg_monitor.config.schema import load_config
from ecg_monitor.core.models import ECGPackage

_ROOT_DIR = Path(__file__).parents[1]
_DATA_DIR = _ROOT_DIR / "data"
_CONFIG_PATH = _ROOT_DIR / "config.yaml"

@pytest.fixture(scope="session")
def real_data_dir() -> Path:
    """Returns the path to the real data directory."""
    return _DATA_DIR

@pytest.fixture(scope="session")
def root_config():
    """Returns the parsed root config.yaml."""
    return load_config(_CONFIG_PATH)

@pytest.fixture(scope="session")
def record_100(real_data_dir) -> ECGPackage:
    """Returns the loaded Record 100."""
    return load_record("100", real_data_dir)

@pytest.fixture(scope="session")
def record_101(real_data_dir) -> ECGPackage:
    """Returns the loaded Record 101."""
    return load_record("101", real_data_dir)

@pytest.fixture(scope="session")
def record_105(real_data_dir) -> ECGPackage:
    """Returns the loaded Record 105."""
    return load_record("105", real_data_dir)
