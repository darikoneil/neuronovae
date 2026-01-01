from dearpygui import dearpygui as dpg

from neuronovae.gui.gui import start_gui

if __name__ == "__main__":
    gui = start_gui()
    try:
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
    finally:
        dpg.destroy_context()
