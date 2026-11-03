"""
Tasks related to tagging photos with what they show, by comparing their CLIP
embeddings with the embeddings of short text prompts.
"""

import cv2
import json
import numpy as np
import re

from collections.abc import Mapping, Sequence
from functools import cache
from numpy.typing import NDArray
from pathlib import Path

from organy.models import Photograph
from organy.utils.model_files import loading_model
from organy.utils.os import read_bytes


# Tags given by default, each with the prompts that describe it to CLIP. Full
# sentences in the form "a photo of ..." work better than single words, since
# CLIP learnt from captions written that way. A tag scores by its closest
# prompt, so several prompts cover the different ways it can look, such as
# the different instruments under "instrument".
DEFAULT_TAGS: dict[str, tuple[str, ...]] = {
    "stage": (
        "a photo of a performance on a stage",
        "a photo of a musician performing live",
        "a photo of a concert",
    ),
    "group photo": (
        "a photo of a group of people posing together",
        "a group photo of people smiling at the camera",
        "a photo of friends posing for a picture",
    ),
    "portrait": (
        "a close-up portrait photo of a person",
        "a headshot of a person",
        "a close-up photo of a person's face",
    ),
    "crowd": (
        "a photo of a crowd of people",
        "a photo of an audience watching a show",
        "a photo of many people in a crowd",
    ),
    "food": (
        "a photo of food",
        "a photo of a meal on a plate",
        "a photo of drinks and snacks",
    ),
    "venue": (
        "a photo of the inside of a building",
        "a wide photo of a large hall",
        "a photo of an empty room",
    ),
    "outdoors": (
        "a photo taken outdoors",
        "a photo of people outside under the sky",
        "a photo with trees and sky in the background",
    ),
    "instrument": (
        "a photo of a musical instrument",
        "a photo of a person playing a guitar",
        "a photo of a person playing a keyboard",
        "a photo of a drum kit",
        "a photo of a person playing a violin",
        "a photo of a person playing a saxophone",
    ),
    "sign": (
        "a photo of a sign with text",
        "a photo of a banner with a logo",
        "a photo of a large printed sign",
    ),
    "product": (
        "a photo of a product on display",
        "a photo of items for sale at a booth",
        "a photo of products on a shelf",
    ),
}

# Plain captions that fit any photo. A photo's similarity with them is its
# starting point: some photos are close to every prompt, others to none, so a
# tag is judged by how much closer it is than these captions.
NEUTRAL_PROMPTS = ("a photo", "a photo of something", "a picture")

# Rules for keeping a photo's tags (see `tag_photos`):
# - MIN_TAG_GAIN: how much closer than the neutral captions a tag must be
# - MAX_TAGS: most tags kept for one photo
#
# Each tag is judged on its own, so a photo can have several tags even when
# one of them, such as "stage", is much closer than the rest.
#
# On the 49 stock photos, against hand-picked tags, counting only tags that
# were clearly right or wrong:
# - the rule used before (one prompt per tag, keeping tags within 0.015 of
#   the best one) gave right tags 100% of the time but found only 61% of them
# - this rule found 84% of them, and 92% of the tags it gave were right
# - a gain of 0.02 or 0.03 also kept both above 80%
#
# It was chosen on those same photos, so it still needs checking against
# real event photos.
MIN_TAG_GAIN = 0.025
MAX_TAGS = 3

# CLIP reads at most 77 tokens, including the start and end tokens.
CONTEXT_LENGTH = 77
START_TOKEN = "<|startoftext|>"
END_TOKEN = "<|endoftext|>"

# Splits text into words, numbers and punctuation the same way CLIP did when
# it was trained: common English contractions, runs of letters, single
# digits, and runs of anything else that is not a space.
WORD_PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+")


class ClipTokenizer:
    """
    Turns text into the token numbers the CLIP text model reads, using the
    files of the Xenova/clip-vit-base-patch32 repository on Hugging Face:
    "vocab.json" (every token and its number) and "merges.txt" (which pairs
    of characters are joined into longer tokens, most common first).
    """

    def __init__(self, vocab_path: Path, merges_path: Path) -> None:
        with loading_model(vocab_path):
            self._vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))

        # The first line of merges.txt is a version header, not a merge.
        with loading_model(merges_path):
            lines = merges_path.read_text(encoding="utf-8").splitlines()[1:]
        merges = [tuple(line.split()) for line in lines if line.strip()]
        self._ranks: dict[tuple[str, ...], int] = {pair: rank for rank, pair in enumerate(merges)}

        self._characters = _byte_characters()
        self._cache: dict[str, list[str]] = {}

    def encode(self, text: str) -> NDArray[np.int64]:
        """
        Turns text into `CONTEXT_LENGTH` token numbers: the start token, the
        text's tokens, the end token, then zeros. Text that is too long is
        cut off so the end token still fits.
        """
        tokens = [START_TOKEN]
        for word in WORD_PATTERN.findall(" ".join(text.lower().split())):
            # Each byte of the word is written as one printable character, so
            # any text, including emoji and accents, can be tokenized.
            tokens += self._merge("".join(self._characters[byte] for byte in word.encode("utf-8")))

        ids = [self._vocab[token] for token in tokens[:CONTEXT_LENGTH - 1]]
        ids.append(self._vocab[END_TOKEN])

        encoded = np.zeros((1, CONTEXT_LENGTH), dtype=np.int64)
        encoded[0, :len(ids)] = ids
        return encoded

    def _merge(self, word: str) -> list[str]:
        """
        Splits one word into tokens by byte pair encoding: start from single
        characters, with "</w>" marking the end of the word, and keep joining
        the neighbouring pair that comes first in merges.txt until no pair is
        left to join.
        """
        if word in self._cache: return self._cache[word]

        parts = list(word[:-1]) + [word[-1] + "</w>"]
        while len(parts) > 1:
            pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
            best = min(pairs, key=lambda pair: self._ranks.get(pair, len(self._ranks)))
            if best not in self._ranks: break

            # Join every place where the best pair appears, left to right.
            joined: list[str] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best:
                    joined.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    joined.append(parts[i])
                    i += 1
            parts = joined

        self._cache[word] = parts
        return parts


def load_text_model(path: Path) -> cv2.dnn.Net:
    """
    Loads the text part of a CLIP model from an ONNX file, such as
    "onnx/text_model.onnx" from the Xenova/clip-vit-base-patch32 repository
    on Hugging Face (about 250 MB, downloaded once). It must come from the
    same CLIP model as the image part, or their embeddings cannot be compared.

    It runs on the CPU.
    """
    # Read by Python for non-ASCII paths on Windows, see `read_bytes`.
    with loading_model(path):
        return cv2.dnn.readNetFromONNX(read_bytes(path))


def embed_texts(model: cv2.dnn.Net, tokenizer: ClipTokenizer, texts: Sequence[str]) -> NDArray[np.float32]:
    """
    Computes the embedding of every text in `texts` and returns them as a
    table, one row per text. Each row is scaled to a length of 1, like photo
    embeddings, so a photo's similarity with a text is `photo @ row`.

    Each text takes about 20 ms, and only has to be embedded once for all photos.
    """
    rows: list[NDArray[np.float32]] = []
    for text in texts:
        model.setInput(tokenizer.encode(text), "input_ids")
        embedding = np.asarray(model.forward(), dtype=np.float32).ravel()
        rows.append(embedding / np.linalg.norm(embedding))

    return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)


def embed_prompts(
    model: cv2.dnn.Net,
    tokenizer: ClipTokenizer,
    prompts: Mapping[str, Sequence[str]],
) -> dict[str, NDArray[np.float32]]:
    """
    Computes the embeddings of every tag's prompts in `prompts` (each tag with
    its prompts, like `DEFAULT_TAGS`) and returns them by tag, as tables from
    `embed_texts`. A tag without prompts is left out.
    """
    return {tag: embed_texts(model, tokenizer, texts) for tag, texts in prompts.items() if texts}


def tag_photos(
    photos: Mapping[Photograph, NDArray[np.float32]],
    prompts: Mapping[str, NDArray[np.float32]],
    neutral: NDArray[np.float32],
) -> dict[Photograph, list[tuple[str, float]]]:
    """
    Tags every photo in `photos` (each photo with its embedding, as
    `compute_embeddings` returns) with what it shows.

    `prompts` holds each tag's prompt embeddings (as `embed_prompts` returns),
    and `neutral` the embeddings of `NEUTRAL_PROMPTS` (as `embed_texts`
    returns). For each photo and tag:
    - the tag's similarity is the one of its closest prompt
    - its gain is how much higher that is than the photo's average similarity
      with the neutral captions
    - the tag is kept when its gain is at least `MIN_TAG_GAIN`

    Each photo gets up to `MAX_TAGS` tags with their gains, highest first. A
    photo that no tag fits well enough gets no tags. Gains are raw values
    for comparing tags, not confidences. Tags are meant for searching and
    naming stacks, not for grouping photos.
    """
    if not photos or not prompts or len(neutral) == 0: return {photo: [] for photo in photos}

    tags = list(prompts)
    table = np.vstack(list(photos.values()))

    # One row per photo and one column per tag, holding each tag's closest
    # prompt, minus the photo's average similarity with the neutral captions.
    similarity = np.stack([(table @ prompts[tag].T).max(axis=1) for tag in tags], axis=1)
    gains = similarity - (table @ neutral.T).mean(axis=1, keepdims=True)

    tagged: dict[Photograph, list[tuple[str, float]]] = {}
    for photo, row in zip(photos, gains):
        order = np.argsort(-row)[:MAX_TAGS]
        tagged[photo] = [(tags[index], float(row[index])) for index in order if row[index] >= MIN_TAG_GAIN]

    return tagged


@cache
def _byte_characters() -> dict[int, str]:
    """
    Gives every byte (0 to 255) a printable character, as CLIP's tokenizer
    does. Printable bytes keep their own character and the others are moved
    to characters from 256 on, so no token holds a space or control character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    characters = {byte: chr(byte) for byte in printable}
    extra = 0
    for byte in range(256):
        if byte not in characters:
            characters[byte] = chr(256 + extra)
            extra += 1

    return characters
