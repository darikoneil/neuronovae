from pathlib import Path
import pytest
import numpy as np


IMAGE_2D = "dont_panic"


@pytest.fixture(scope="session")
def assets_path() -> Path:
    """
    Returns the path to the test assets folder.

    :return: The path to the assets folder
    """
    return Path(Path(__file__).parent).joinpath("assets")


@pytest.fixture(scope="function")
def image_case(request) -> tuple[Path, np.ndarray]:
    (dimensions, file_extension) = request.param
    match dimensions:
        case 2:
            filename = request.getfixturevalue("assets_path").joinpath(
                IMAGE_2D + file_extension
            )
            reference = np.load(filename, allow_pickle=False)
        case _:
            raise ValueError("Invalid dimensions")
    # noinspection PyUnboundLocalVariable
    return filename, reference
