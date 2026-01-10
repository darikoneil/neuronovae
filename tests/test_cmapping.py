import numpy as np
import pytest
from pydantic import ValidationError

from neuronovae.cmapping import Color


@pytest.mark.parametrize(
    ("hex_str", "expected_color"),
    [
        ("#FF0000", Color(1.0, 0.0, 0.0, 1.0)),
        ("#00FF00", Color(0.0, 1.0, 0.0, 1.0)),
        ("#0000FF", Color(0.0, 0.0, 1.0, 1.0)),
        ("#FFFFFF", Color(1.0, 1.0, 1.0, 1.0)),
        ("#000000", Color(0.0, 0.0, 0.0, 1.0)),
        ("#FF000080", Color(1.0, 0.0, 0.0, 0.5)),
    ],
    ids=[
        "red",
        "green",
        "blue",
        "white",
        "black",
        "red_with_alpha",
    ],
)
def test_hex_creates_correct_color(hex_str: str, expected_color: Color) -> None:
    """
    Test that Color.from_hex creates the correct Color instance.
    Args:
        hex_str (str): The hex string to convert.
        expected_color (Color): The expected Color instance.
    """
    assert np.all(
        np.allclose(color_hex, color_ref)
        for color_hex, color_ref in zip(
            Color.from_hex(hex_str), expected_color, strict=True
        )
    )


@pytest.mark.parametrize(
    "hex_str",
    [
        "#FFF",
        "#GGGGGG",
        "#12345",
        "123456",
        "#123456789",
    ],
    ids=[
        "short_hex",
        "invalid_characters",
        "too_short",
        "missing_hash",
        "too_long",
    ],
)
def test_hex_raises_value_error_on_invalid_input(hex_str: str) -> None:
    """
    Test that Color.from_hex raises ValueError or ValidationError for invalid hex strings.
    Args:
        hex_str (str): The invalid hex string to test.
    Raises:
        ValidationError: If the resulting Color instance fails validation.
    """
    with pytest.raises((ValueError, ValidationError)):
        Color.from_hex(hex_str)


@pytest.mark.parametrize(
    ("rgba", "expected_color"),
    [
        ((255, 0, 0), Color(1.0, 0.0, 0.0, 1.0)),
        ((0, 255, 0), Color(0.0, 1.0, 0.0, 1.0)),
        ((0, 0, 255), Color(0.0, 0.0, 1.0, 1.0)),
        ((255, 255, 255), Color(1.0, 1.0, 1.0, 1.0)),
        ((0, 0, 0), Color(0.0, 0.0, 0.0, 1.0)),
        ((255, 0, 0, 128), Color(1.0, 0.0, 0.0, 0.5)),
    ],
    ids=[
        "red",
        "green",
        "blue",
        "white",
        "black",
        "red_with_alpha",
    ],
)
def test_rgba_creates_correct_color(
    rgba: tuple[int, ...], expected_color: Color
) -> None:
    """
    Test that Color.from_rgba creates the correct Color instance.

    Args:
        rgba (tuple[int, ...]): The RGBA values to convert.
        expected_color (Color): The expected Color instance.
    """
    assert np.all(
        np.allclose(color_rgba, color_ref)
        for color_rgba, color_ref in zip(
            Color.from_rgba(*rgba), expected_color, strict=True
        )
    )
