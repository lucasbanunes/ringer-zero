from pathlib import Path
import pytest


@pytest.fixture
def root_dir() -> Path:
    return Path(__file__).parent.parent.absolute()


@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parent.absolute() / 'data'


@pytest.fixture(scope='session')
def create_logs_dir() -> Path:
    logs_dir = Path('./logs')
    logs_dir.mkdir(exist_ok=True, parents=True)
    return logs_dir
