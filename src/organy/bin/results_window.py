"""
Tkinter window that shows Organy's results once every stage has finished:
the events on the left, and the stacks of the chosen event on the right as
thumbnails, with the suggested best photo of each stack marked.

The CLI imports this module only when the window is shown, so `organy --help`
stays fast and nothing needs a display when the window is not used.
"""

import base64
import cv2
import tkinter as tk

from tkinter import font, ttk

from organy.bin.dialogs import apply_window_scale, enable_hidpi
from organy.bin.results import ResultGroup, Results, format_quality, format_time
from organy.models import Photograph
from organy.tasks.selection import PhotoQuality
from organy.utils.image import downscale


# Longest side, in pixels, of each thumbnail and of the larger preview shown
# when a thumbnail is clicked.
THUMBNAIL_SIZE = 200
PREVIEW_SIZE = 900

# Layout of the cards, in pixels:
# - CARD_PADDING: space between a card's border and its contents
# - GAP: space between cards, and around all of them
CARD_PADDING = 8
GAP = 8

# Thumbnails made in one go before the window gets to redraw and respond
# again, so scrolling fills them in gradually instead of freezing the window.
THUMBNAILS_PER_TICK = 8

# List width and window size, in pixels.
LIST_WIDTH = 440
WINDOW_SIZE = (1280, 800)

# Colours of the card borders, and of a thumbnail's place until it is made.
BEST_COLOR = "#2e9d4f"
BORDER_COLOR = "#c4c4c4"
PLACEHOLDER_COLOR = "#e4e4e4"


def show_results(results: Results, scale: float = 1.0) -> None:
    """
    Opens the results window and returns once the user closes it. `scale`
    enlarges its text, thumbnails and spacing (see `apply_window_scale`).
    """
    enable_hidpi()

    root = tk.Tk()
    root.title("Organy - Results")
    apply_window_scale(root, scale)

    width, height = WINDOW_SIZE
    root.geometry(f"{round(width * scale)}x{round(height * scale)}")
    ResultsWindow(root, results, scale)
    root.mainloop()


def make_photo_image(photo: Photograph, size: int) -> tk.PhotoImage:
    """
    Turns a photo's downscaled copy into a Tk image whose longest side is at
    most `size` px. Tk reads PNG on its own, so no other image library is needed.
    """
    image = downscale(photo.decode_downscaled(), size)

    # Fast compression, since the PNG is decoded again right away.
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok: raise ValueError(f"cannot encode thumbnail of {photo.path}")
    return tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"))


class ResultsWindow:
    """
    The contents of the results window.

    Every card is drawn as items on one canvas (rectangles, images and
    text) instead of as widgets. A frame holding every card of a large group
    would be taller than an X11 window can be (32767 px), and its backing
    pixmap too large for the X server, which fails with BadAlloc. The canvas
    stays the size of the window however many photos it holds.

    Thumbnails are only made for cards near the visible part, and only the
    chosen group's thumbnails are kept, so even thousands of photos stay light.
    """

    def __init__(self, root: tk.Tk, results: Results, scale: float = 1.0) -> None:
        self._root = root

        # Every size in pixels is enlarged by `scale`, like the fonts (see
        # `apply_window_scale`).
        self._thumbnail_size = round(THUMBNAIL_SIZE * scale)
        self._preview_size = round(PREVIEW_SIZE * scale)
        self._card_padding = round(CARD_PADDING * scale)
        self._gap = round(GAP * scale)

        # Room taken by one card, including the gap after it.
        self._card_width = self._thumbnail_size + 2 * self._card_padding + self._gap

        # Thumbnails are made this far above and below the visible part, so
        # they are usually ready before the user scrolls to them.
        self._fill_margin = 2 * self._thumbnail_size

        self._results = results
        self._group: ResultGroup | None = None
        self._columns = 0

        # Tk forgets an image that Python no longer holds, so the made
        # thumbnails and the preview are kept here.
        self._thumbnails: dict[Photograph, tk.PhotoImage] = {}
        self._preview_image: tk.PhotoImage | None = None

        # Cards whose thumbnail is not made yet: the placeholder rectangle
        # marking where the thumbnail goes, with the image item to fill and its photo.
        self._pending: dict[int, tuple[int, Photograph]] = {}
        self._fill_scheduled = False

        style = ttk.Style(root)
        if "clam" in style.theme_names(): style.theme_use("clam")
        self._text_color = style.lookup("TLabel", "foreground") or "black"

        # Fonts are copies of Tk's default font, so they follow the system's
        # font and size. A family name like "TkDefaultFont" in a tuple would
        # be looked up as a real family instead and fall back to another font.
        base = font.nametofont("TkDefaultFont")
        size = abs(int(base.actual("size")))
        self._font = base
        self._bold_font = base.copy()
        self._bold_font.configure(weight="bold")
        self._stack_font = base.copy()
        self._stack_font.configure(size=size + 2, weight="bold")
        self._heading_font = base.copy()
        self._heading_font.configure(size=size + 4, weight="bold")

        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Left: every group, one per row. The list keeps its width when the
        # window is resized, and the thumbnails take the rest.
        self._tree = ttk.Treeview(paned, show="tree", selectmode="browse")
        self._tree.column("#0", width=round(LIST_WIDTH * scale), minwidth=round(200 * scale))
        for index, group in enumerate(results.groups):
            text = f"{results.title(group)}  ({group.photo_count} photos, {len(group.stacks)} stacks)"
            self._tree.insert("", "end", iid=str(index), text=text)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        paned.add(self._tree, weight=0)

        # Right: the chosen group's stacks, drawn on a scrollable canvas.
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._heading = ttk.Label(right, font=self._heading_font, padding=(round(10 * scale), round(8 * scale)))
        self._heading.pack(fill="x")

        self._canvas = tk.Canvas(
            right, highlightthickness=0, background=style.lookup("TFrame", "background") or "white",
        )
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=self._canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # Every scroll, by the wheel or the scrollbar, also fills in the
        # thumbnails that came into view.
        def on_scroll(first: str, last: str) -> None:
            scrollbar.set(first, last)
            self._schedule_fill()

        self._canvas.configure(yscrollcommand=on_scroll)
        self._canvas.bind("<Configure>", self._on_resize)

        # Tk 9 sends <MouseWheel> on every system. Tk 8.6 on Linux sends
        # buttons 4 and 5 instead.
        self._canvas.bind("<MouseWheel>", lambda event: self._scroll(-1 if event.delta > 0 else 1))
        self._canvas.bind("<Button-4>", lambda _: self._scroll(-1))
        self._canvas.bind("<Button-5>", lambda _: self._scroll(1))

        if results.groups:
            self._tree.selection_set("0")
        else:
            self._heading.configure(text="No photos to show.")

    def _on_select(self, _: tk.Event) -> None:
        selection = self._tree.selection()
        if not selection: return

        group = self._results.groups[int(selection[0])]
        if group is not self._group:
            self._group = group
            self._thumbnails.clear()
            self._render()

    def _on_resize(self, event: tk.Event) -> None:
        # Cards are laid out again only when a different number fits per row.
        columns = max(1, (event.width - self._gap) // self._card_width)
        if columns != self._columns:
            self._columns = columns
            self._render()
        else:
            self._schedule_fill()

    def _scroll(self, step: int) -> None:
        self._canvas.yview_scroll(step, "units")

    def _render(self) -> None:
        """Draws the chosen group's stacks, then fills in the visible thumbnails."""
        group = self._group
        if group is None: return

        self._canvas.delete("all")
        self._pending.clear()

        self._heading.configure(
            text=f"{self._results.title(group)}: {group.photo_count} photo(s) in {len(group.stacks)} stack(s), "
                 "best photo of each stack outlined in green",
        )

        columns = max(1, self._columns)
        y = self._gap
        for number, stack in enumerate(group.stacks, start=1):
            header = self._canvas.create_text(
                self._gap, y + self._gap, text=f"Stack {number}: {len(stack.photos)} photo(s)",
                anchor="nw", font=self._stack_font, fill=self._text_color,
            )
            y = self._bottom(header) + self._gap

            for start in range(0, len(stack.photos), columns):
                row = stack.photos[start:start + columns]
                cards = [
                    self._draw_card(self._gap + column * self._card_width, y, photo, stack.qualities[photo], photo is stack.best)
                    for column, photo in enumerate(row)
                ]

                # Every border in a row reaches the bottom of its tallest card.
                row_bottom = max(bottom for _, bottom in cards)
                for border, _ in cards:
                    x0, y0, x1, _ = self._canvas.coords(border)
                    self._canvas.coords(border, x0, y0, x1, row_bottom)
                y = row_bottom + self._gap

        self._canvas.configure(scrollregion=(0, 0, self._gap + columns * self._card_width, y + self._gap))
        self._canvas.yview_moveto(0)
        self._schedule_fill()

    def _draw_card(
        self, x: int, y: int, photo: Photograph, quality: PhotoQuality, best: bool,
    ) -> tuple[int, int]:
        """
        Draws one photo's card with its top left corner at (x, y): a place for
        its thumbnail, its details, and a border, green when it is the
        suggested best photo. It returns the border and the card's bottom.
        """
        left = x + self._card_padding
        top = y + self._card_padding

        # Each card gets its own tag, so a click anywhere on its thumbnail opens the preview.
        tag = f"card-{id(photo)}"
        placeholder = self._canvas.create_rectangle(
            left, top, left + self._thumbnail_size, top + self._thumbnail_size,
            fill=PLACEHOLDER_COLOR, outline="", tags=(tag,),
        )
        image = self._canvas.create_image(
            left + self._thumbnail_size // 2, top + self._thumbnail_size // 2, anchor="center", tags=(tag,),
        )
        self._pending[placeholder] = (image, photo)

        self._canvas.tag_bind(tag, "<Button-1>", lambda _: self._preview(photo))
        self._canvas.tag_bind(tag, "<Enter>", lambda _: self._canvas.configure(cursor="hand2"))
        self._canvas.tag_bind(tag, "<Leave>", lambda _: self._canvas.configure(cursor=""))

        lines: list[tuple[str, font.Font, str]] = [
            ("Suggested best", self._bold_font, BEST_COLOR) if best else ("", self._font, ""),
            (photo.path.name, self._bold_font, self._text_color),
            (photo.photographer, self._font, self._text_color),
            (format_time(self._results.times.get(photo)), self._font, self._text_color),
            (format_quality(quality), self._font, self._text_color),
            (f"tags {self._results.tags_text(photo)}", self._font, self._text_color),
        ]
        text_y = top + self._thumbnail_size + self._card_padding
        for text, text_font, color in lines:
            if not text: continue
            item = self._canvas.create_text(
                left, text_y, text=text, anchor="nw", width=self._thumbnail_size, font=text_font, fill=color,
            )
            text_y = self._bottom(item) + 2

        bottom = text_y + self._card_padding
        border = self._canvas.create_rectangle(
            x, y, x + self._thumbnail_size + 2 * self._card_padding, bottom,
            outline=BEST_COLOR if best else BORDER_COLOR, width=3,
        )
        return border, bottom

    def _bottom(self, item: int) -> int:
        return self._canvas.bbox(item)[3]

    def _schedule_fill(self) -> None:
        """Fills in the visible thumbnails once the window is idle, at most once at a time."""
        if self._fill_scheduled: return
        self._fill_scheduled = True
        self._root.after(1, self._fill_visible)

    def _fill_visible(self) -> None:
        """Makes a few thumbnails near the visible part, then lets the window redraw before making more."""
        self._fill_scheduled = False
        if not self._pending: return

        top = self._canvas.canvasy(0) - self._fill_margin
        bottom = self._canvas.canvasy(self._canvas.winfo_height()) + self._fill_margin
        nearby = [
            item for item in self._canvas.find_overlapping(0, top, self._canvas.winfo_width(), bottom)
            if item in self._pending
        ]

        for placeholder in nearby[:THUMBNAILS_PER_TICK]:
            image, photo = self._pending.pop(placeholder)
            if photo not in self._thumbnails:
                self._thumbnails[photo] = make_photo_image(photo, self._thumbnail_size)
            self._canvas.itemconfigure(image, image=self._thumbnails[photo])

        if len(nearby) > THUMBNAILS_PER_TICK: self._schedule_fill()

    def _preview(self, photo: Photograph) -> None:
        """Opens a larger copy of a photo in its own window."""
        window = tk.Toplevel(self._root)
        window.title(f"Organy - {photo.path.name}")

        self._preview_image = make_photo_image(photo, self._preview_size)
        ttk.Label(window, image=self._preview_image).pack()
        ttk.Label(
            window, text=f"{photo.path}\n{photo.photographer}, {format_time(self._results.times.get(photo))}",
            padding=self._card_padding,
        ).pack(anchor="w")
