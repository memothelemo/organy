"""
Tkinter dialog for choosing the photos to import.

Tk's native dialogs pick many files at once but only one folder at a time, so
this small window lets the user add files and folders repeatedly before
confirming. tkinter is imported inside the functions so the CLI starts fast and
still works without a display when paths are given on the command line.
"""

import sys

from collections.abc import Iterable
from pathlib import Path


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


def ask_photo_sources() -> list[Path]:
    """
    Shows the selection window and returns the chosen files and folders.
    Each folder's name becomes the photographer of the photos in it, unless
    a photo names its photographer in its EXIF.

    It returns an empty list when the user cancels or closes the window.
    """
    import tkinter as tk
    from tkinter import filedialog, ttk

    enable_hidpi()

    chosen: list[Path] = []
    confirmed = False

    root = tk.Tk()
    root.title("Organy - Select photos")
    root.minsize(520, 300)

    ttk.Label(
        root,
        text="Add each photographer's folder, or single photos. A folder's name is used as its photographer.",
        wraplength=600,
    ).pack(fill="x", padx=8, pady=(8, 0))

    listbox = tk.Listbox(root, width=80, height=12, selectmode="extended")
    listbox.pack(fill="both", expand=True, padx=8, pady=8)

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
    buttons.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Button(buttons, text="Add folder...", command=add_folder).pack(side="left")
    ttk.Button(buttons, text="Add files...", command=add_files).pack(side="left", padx=4)
    ttk.Button(buttons, text="Remove", command=remove_selected).pack(side="left")
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right")
    import_button = ttk.Button(buttons, text="Import", command=confirm)
    import_button.pack(side="right", padx=4)
    import_button.state(["disabled"])

    root.mainloop()
    return chosen if confirmed else []
