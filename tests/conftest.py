from pathlib import Path
import pytest


@pytest.fixture
def root_dir() -> Path:
    return Path(__file__).parent.parent.absolute()


@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parent.absolute() / 'data'
