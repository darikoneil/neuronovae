from typing import Any

import dearpygui.dearpygui as dpg


def create_menu_bar() -> None:
    """Create the main menu bar for the application."""
    with dpg.viewport_menu_bar():  # noqa: SIM117
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="New", callback=lambda: print("New File"))
            dpg.add_menu_item(label="Load Images", callback=load_images_callback)
            dpg.add_menu_item(label="Load ROIs", callback=lambda: print("Open ROIs"))
            dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
        # with dpg.menu(label="Help"):
        #     dpg.add_menu_item(label="About", callback=lambda: print("About This Application"))
        # with dpg.menu(label="About"):
        #     ...


def set_images_path(sender: Any, app_data: Any) -> None:
    dpg.set_value("images_path", app_data.get("file_path_name"))


def load_images_callback(sender: Any, app_data: Any, user_data: Any) -> None:
    """Callback function to load images from a file dialog."""
    with dpg.file_dialog(
        directory_selector=False,
        show=True,
        callback=set_images_path,
        tag="select_images_file",
        width=600,
        height=400,
        modal=True,
    ):
        dpg.add_file_extension(".*")
