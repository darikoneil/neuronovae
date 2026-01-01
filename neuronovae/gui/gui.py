from typing import TYPE_CHECKING, Any

from dearpygui import dearpygui as dpg
from pydantic.dataclasses import dataclass

from neuronovae.gui.menu import create_menu_bar

if TYPE_CHECKING:
    from neuronovae.dataset import Dataset


@dataclass(frozen=True)
class GUISettings:
    title: str = "neuronovae"
    width: int = 1000
    height: int = 600
    min_width: int = 100
    min_height: int = 600
    decorated: bool = True
    resizable: bool = True
    vsync: bool = True

    @classmethod
    def dump(cls) -> dict[str, Any]:
        return cls().__dict__


class GUI:
    """
    Stateholder for the neuronovae GUI.
    """

    def __init__(self):
        self.images_path: str = ""
        self.rois_path: str = ""
        self.dataset: Dataset | None = None
        self.downsample_factor: int = 1.0
        self.create_data_slots()

    def create_data_slots(self) -> None:
        """
        Create GUI slots for data loading and display.
        """
        with dpg.value_registry():
            dpg.add_string_value(default_value=self.images_path, tag="images_path")
            dpg.add_string_value(default_value=self.rois_path, tag="rois_path")


def start_gui(debug: bool = False) -> GUI:  # noqa: FBT001, FBT002
    """
    Start the DearPyGui application.
    Args:
        debug: Enable additional windows containing debug information.

    Returns: The GUI's stateholder

    """
    dpg.create_context()
    gui = GUI()
    if debug:
        dpg.show_documentation()
        dpg.show_style_editor()
        dpg.show_debug()
        dpg.show_about()
        dpg.show_metrics()
        dpg.show_font_manager()
        dpg.show_item_registry()
    dpg.create_viewport(**GUISettings.dump())
    create_menu_bar()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    return gui
