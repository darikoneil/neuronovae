from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def assets_path() -> Path:
    """
    Returns the path to the test assets folder.

    :return: The path to the assets folder
    """
    return Path(__file__).parents[1].joinpath("assets")
