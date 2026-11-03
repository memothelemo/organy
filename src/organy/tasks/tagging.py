"""
Tasks related to tagging photos with what they show, by comparing their CLIP
embeddings with the embeddings of short text prompts.
"""

import cv2
import json
import numpy as np
import re

from collections.abc import Mapping
from functools import cache
from numpy.typing import NDArray
from pathlib import Path

from organy.models import Photograph
from organy.utils.os import read_bytes


# Tags given by default, each with the prompt that describes it to CLIP.
# Full sentences in the form "a photo of ..." work better than single words,
# since CLIP learnt from captions written that way.
DEFAULT_TAGS = {
    "stage": "a photo of a performance on a stage",
    "group photo": "a photo of a group of people posing together",
    "portrait": "a close-up portrait photo of a person",
    "crowd": "a photo of a crowd of people",
    "food": "a photo of food",
    "venue": "a photo of the inside of a building",
    "outdoors": "a photo taken outdoors",
    "instrument": "a photo of a musical instrument",
    "sign": "a photo of a sign with text",
    "product": "a photo of a product on display",
}

# Rules for keeping a photo's tags, by the similarity of its embedding with
# each tag's prompt embedding (see `tag_photos`):
# - MIN_TAG_SIMILARITY: lowest similarity of the best tag for the photo to be
#   tagged at all. CLIP image-to-text similarities are much lower than
#   image-to-image ones, about 0.15 to 0.35.
# - TAG_MARGIN: other tags are kept if they are at most this far below the
#   best tag.
# - MAX_TAGS: most tags kept for one photo.
#
# These numbers still need checking against real event photos.
MIN_TAG_SIMILARITY = 0.2
TAG_MARGIN = 0.015
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
        self._vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))

        # The first line of merges.txt is a version header, not a merge.
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
    return cv2.dnn.readNetFromONNX(read_bytes(path))


def embed_prompts(
    model: cv2.dnn.Net,
    tokenizer: ClipTokenizer,
    prompts: Mapping[str, str],
) -> dict[str, NDArray[np.float32]]:
    """
    Computes the embedding of every prompt in `prompts` (each tag with its
    prompt, like `DEFAULT_TAGS`) and returns them by tag. Each embedding is
    scaled to a length of 1, like photo embeddings, so a photo's similarity
    with a prompt is `photo @ prompt`.

    Each prompt takes about 20 ms, and only has to be embedded once for all photos.
    """
    embeddings: dict[str, NDArray[np.float32]] = {}
    for tag, prompt in prompts.items():
        model.setInput(tokenizer.encode(prompt), "input_ids")
        embedding = np.asarray(model.forward(), dtype=np.float32).ravel()
        embeddings[tag] = embedding / np.linalg.norm(embedding)

    return embeddings


def tag_photos(
    photos: Mapping[Photograph, NDArray[np.float32]],
    prompts: Mapping[str, NDArray[np.float32]],
) -> dict[Photograph, list[tuple[str, float]]]:
    """
    Tags every photo in `photos` (each photo with its embedding, as
    `compute_embeddings` returns) with the tags in `prompts` (each tag with
    its prompt embedding, as `embed_prompts` returns) whose prompt is closest
    to the photo.

    Each photo gets its tags with their similarity, most similar first:
    - the best tag, when its similarity is at least `MIN_TAG_SIMILARITY`
    - other tags at most `TAG_MARGIN` below the best one, up to `MAX_TAGS`

    A photo that is not close enough to any prompt gets no tags. Tags are
    meant for searching and naming stacks, not for grouping photos.
    """
    if not photos or not prompts: return {photo: [] for photo in photos}

    tags = list(prompts)

    # One row per photo and one column per tag.
    similarity = np.vstack(list(photos.values())) @ np.vstack([prompts[tag] for tag in tags]).T

    tagged: dict[Photograph, list[tuple[str, float]]] = {}
    for photo, row in zip(photos, similarity):
        best = float(row.max())
        order = np.argsort(-row)[:MAX_TAGS]
        tagged[photo] = [
            (tags[index], float(row[index]))
            for index in order
            if best >= MIN_TAG_SIMILARITY and row[index] >= best - TAG_MARGIN
        ]

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
