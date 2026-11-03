"""
Tkinter dialog for choosing the photos to import.

Tk's native dialogs pick many files at once but only one folder at a time, so
this small window lets the user add files and folders repeatedly before
confirming. tkinter is imported inside the functions so the CLI starts fast and
still works without a display when paths are given on the command line.
"""

import math
import sys

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tkinter as tk


# Tk matches patterns case-sensitively on Linux, so list both cases.
PHOTO_PATTERNS = "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"


def enable_hidpi() -> None:
    """
    Stops Windows from drawing Tk windows at 96 DPI and stretching them.

    Python is not DPI-aware by default, so on a scaled display (e.g. 125%)
    Windows renders the window small and upscales the bitmap, which looks
    blurry. Declaring "system DPI aware" lets Tk draw at the real DPI; the
    native file dialogs become sharp too. It must run before the first Tk
    window is created.

    Limitations:
    - Multiple monitors: per-monitor awareness is avoided because Tk cannot
      re-layout when moved between monitors with different scaling, so on a
      monitor scaled differently from the main one, Windows stretches the
      window again and it looks slightly blurry there.
    - Only fonts follow the DPI. Sizes given in pixels (e.g. `padx=8`) stay
      fixed, so spacing looks tighter on scaled displays.
    - macOS needs nothing: its Tk is already Retina-aware.
    - Linux/WSL is left alone: Tk takes the X server's DPI, so if the window
      looks small there, set `Xft.dpi` in the user's X resources.
    """
    if sys.platform != "win32": return

    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Windows 8.1+
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()  # older Windows


def apply_window_scale(root: "tk.Tk", scale: float) -> None:
    """
    Makes every font of a Tk window `scale` times larger, where 1.0 keeps
    Tk's normal size. It must run before the window's widgets are created.

    Only fonts are changed here. Sizes a window gives in pixels, such as
    padding or thumbnail sizes, must be multiplied by `scale` where they are set.
    """
    from tkinter import font

    # Tk keeps sizes in points when positive and in pixels when negative, so
    # the sign is kept and only the amount grows.
    for name in font.names(root):
        named = font.nametofont(name, root)
        size = int(named.cget("size"))
        named.configure(size=int(math.copysign(math.floor(abs(size) * scale + 0.5), size)))


def ask_photo_sources(scale: float = 1.0) -> list[Path]:
    """
    Shows the selection window and returns the chosen files and folders.
    Each folder's name becomes the photographer of the photos in it, unless
    a photo names its photographer in its EXIF. `scale` enlarges the window
    (see `apply_window_scale`).

    It returns an empty list when the user cancels or closes the window.
    """
    import tkinter as tk
    from tkinter import filedialog, ttk

    enable_hidpi()

    chosen: list[Path] = []
    confirmed = False

    root = tk.Tk()
    root.title("Organy - Select photos")
    apply_window_scale(root, scale)
    root.minsize(round(520 * scale), round(300 * scale))

    # Spacing in pixels, enlarged like the fonts.
    space = round(8 * scale)
    small_space = round(4 * scale)

    ttk.Label(
        root,
        text="Add each photographer's folder, or single photos. A folder's name is used as its photographer.",
        wraplength=round(600 * scale),
    ).pack(fill="x", padx=space, pady=(space, 0))

    listbox = tk.Listbox(root, width=80, height=12, selectmode="extended")
    listbox.pack(fill="both", expand=True, padx=space, pady=space)

    # Iterable, not list: askopenfilenames returns a tuple, or "" when cancelled.
    def add(paths: Iterable[str]) -> None:
        for name in paths:
            path = Path(name)
            if path not in chosen:
                chosen.append(path)
                listbox.insert("end", str(path))
        import_button.state(["!disabled"] if chosen else ["disabled"])

    def add_files() -> None:
        add(filedialog.askopenfilenames(
            parent=root,
            title="Select photos",
            filetypes=[("Photos", PHOTO_PATTERNS), ("All files", "*")],
        ))

    def add_folder() -> None:
        folder = filedialog.askdirectory(parent=root, title="Select a folder of photos", mustexist=True)
        if folder: add([folder])

    def remove_selected() -> None:
        # Removing from the end keeps the earlier indices valid.
        for index in sorted(listbox.curselection(), reverse=True):
            listbox.delete(index)
            del chosen[index]
        import_button.state(["!disabled"] if chosen else ["disabled"])

    def confirm() -> None:
        nonlocal confirmed
        confirmed = True
        root.destroy()

    buttons = ttk.Frame(root)
    buttons.pack(fill="x", padx=space, pady=(0, space))
    ttk.Button(buttons, text="Add folder...", command=add_folder).pack(side="left")
    ttk.Button(buttons, text="Add files...", command=add_files).pack(side="left", padx=small_space)
    ttk.Button(buttons, text="Remove", command=remove_selected).pack(side="left")
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right")
    import_button = ttk.Button(buttons, text="Import", command=confirm)
    import_button.pack(side="right", padx=small_space)
    import_button.state(["disabled"])

    root.mainloop()
    return chosen if confirmed else []
