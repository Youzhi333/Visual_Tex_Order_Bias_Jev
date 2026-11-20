"""State rendering: text, images, JSON-like dicts, or chat transcripts -> one string."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Tuple

VISUAL_MARK = "<<anyjev-image:{}>>"
VISUAL_MARK_RE = re.compile(r"<<anyjev-image:(.+?)>>")

# Invisible marker for a middle placement: `HFVLMBackend(image_placement="split")`
# inserts the images at the first occurrence and removes the marker. It is not
# whitespace, so the text-line cleanup keeps the line it sits on.
IMAGE_SPLIT = "\u2063ANYJEV_IMAGE_SPLIT\u2063"


@dataclass(frozen=True)
class VisualState:
    """A state with images. Each image is a path the backend can open.

    `render_state` writes the paths into the prompt as `<<anyjev-image:path>>`
    markers; a backend that understands images (e.g. `HFVLMBackend`) replaces
    them with image content parts. Text-only backends would see the markers as
    text, so use a visual backend for these states.

    `text` may contain `IMAGE_SPLIT` to request a middle placement; the backend
    splits there only when constructed with `image_placement="split"`.
    """
    text: str = ""
    images: Tuple[str, ...] = ()


def render_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, VisualState):
        marks = "\n".join(VISUAL_MARK.format(p) for p in state.images)
        return f"{marks}\n{state.text}".strip() if state.text else marks
    if isinstance(state, list) and state and all(
        isinstance(m, dict) and "role" in m and "content" in m for m in state
    ):
        return "\n".join(f"{m['role']}: {m['content']}" for m in state)
    try:
        return json.dumps(state, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(state)
