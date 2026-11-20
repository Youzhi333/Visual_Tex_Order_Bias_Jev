"""transformers backend for vision-language models: the same label readout, with images.

A `VisualState` carries image paths; `render_state` writes them into the prompt as
`<<anyjev-image:path>>` markers. This backend replaces those markers with image content
parts, renders the processor's chat template, and scores the label tokens at the last
position of one prefill. Like `HFBackend`, it generates nothing.

Install with `pip install "anyjev[hf,vlm]"`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anyjev.state import IMAGE_SPLIT, VISUAL_MARK_RE

_ID_MARK = "\x00anyjev-id:"
IMAGE_PLACEMENTS = ("prefix", "suffix", "split", "token", "splice", "splice_token")


def needs_structured_content(template: str) -> bool:
    """Whether a chat template indexes `content` parts instead of accepting a str.

    LLaVA-NeXT's template uses `message['content'][0]['text']` and filters on
    `selectattr('type', ...)`; Qwen's accepts plain strings.
    """
    return ("message['content'][0]['text']" in template
            or "message['content'][0]['type']" in template
            or "selectattr('type'" in template)


def _content_parts(user: str, image_parts: List[Dict[str, Any]],
                   placement: str) -> List[Dict[str, Any]]:
    """Image and text content parts in the requested order.

    `prefix` is the pre-0.4 path (all images, then one text part) and must stay
    byte-identical. `suffix` puts the text first. `split` requires exactly one
    `IMAGE_SPLIT` and inserts the images between the two text parts.
    """
    text_parts = [{"type": "text", "text": user}]
    if placement == "suffix":
        return text_parts + image_parts
    if placement == "split":
        if IMAGE_SPLIT not in user:
            raise ValueError("image_placement='split' needs an "
                             "anyjev.state.IMAGE_SPLIT sentinel in the text")
        before, after = user.split(IMAGE_SPLIT, 1)
        parts: List[Dict[str, Any]] = []
        if before:
            parts.append({"type": "text", "text": before})
        parts.extend(image_parts)
        if after:
            parts.append({"type": "text", "text": after})
        return parts
    return image_parts + text_parts


def _find_subsequence(seq: List[int], sub: List[int]) -> int:
    if not sub:
        return -1
    for index in range(len(seq) - len(sub) + 1):
        if seq[index:index + len(sub)] == sub:
            return index
    return -1


def _move_image_block(ids: List[int], image_id: int, phrase_ids: List[int],
                      after: bool) -> List[int]:
    """Move the contiguous image-token block to a phrase boundary in `ids`.

    `after=True` inserts after the phrase (suffix); `after=False` inserts before
    it (pre-question or pre-options). The token count is unchanged. Raises if
    the image block or the phrase is missing.
    """
    positions = [i for i, token in enumerate(ids) if token == image_id]
    if not positions:
        raise ValueError("no image tokens in the input_ids")
    start, end = positions[0], positions[-1]
    if any(token != image_id for token in ids[start:end + 1]):
        raise ValueError("image tokens are not one contiguous block")
    block = ids[start:end + 1]
    kept = ids[:start] + ids[end + 1:]
    index = _find_subsequence(kept, phrase_ids)
    if index < 0:
        raise ValueError("placement phrase not found in the text tokens")
    insert_at = index + len(phrase_ids) if after else index
    return kept[:insert_at] + block + kept[insert_at:]


def _move_image_block_text(ids: List[int], image_id: int, phrase: str,
                           tokenizer, after: bool,
                           allow_gaps: bool = False) -> List[int]:
    """Character-offset splice: find `phrase` in the decoded text, then insert.

    Tokenizing a phrase in isolation does not always reproduce its in-context
    tokens (LLaVA-NeXT is one example), so map the phrase by its character
    position in the decoded sequence and count the text tokens up to it. With
    `allow_gaps`, a multi-image sequence is moved as one span from the first to
    the last image token, preserving any text between the images.
    """
    positions = [i for i, token in enumerate(ids) if token == image_id]
    if not positions:
        raise ValueError("no image tokens in the input_ids")
    start, end = positions[0], positions[-1]
    if not allow_gaps and any(token != image_id for token in ids[start:end + 1]):
        raise ValueError("image tokens are not one contiguous block")
    block = ids[start:end + 1]
    kept = ids[:start] + ids[end + 1:]
    text = tokenizer.decode(kept)
    char = text.find(phrase)
    if char < 0:
        raise ValueError(f"placement phrase {phrase!r} not found in the decoded text")
    cut = char + len(phrase) if after else char
    insert_at = min(max(len(tokenizer.encode(text[:cut], add_special_tokens=False)), 0), len(kept))
    return kept[:insert_at] + block + kept[insert_at:]


def _move_image_block_fraction(ids: List[int], image_id: int,
                               fraction: float,
                               allow_gaps: bool = False) -> List[int]:
    """Move the image block to floor(fraction * n_text_tokens) after removing it.

    With `allow_gaps`, a multi-image sequence moves as one span from the first
    to the last image token.
    """
    positions = [i for i, token in enumerate(ids) if token == image_id]
    if not positions:
        raise ValueError("no image tokens in the input_ids")
    start, end = positions[0], positions[-1]
    if not allow_gaps and any(token != image_id for token in ids[start:end + 1]):
        raise ValueError("image tokens are not one contiguous block")
    block = ids[start:end + 1]
    kept = ids[:start] + ids[end + 1:]
    index = min(max(int(fraction * len(kept)), 0), len(kept))
    return kept[:index] + block + kept[index:]


def _split_at_token(user: str, fraction: float, tokenizer) -> Tuple[str, str]:
    """Split `user` at floor(fraction * n_text_tokens) using tokenizer offsets.

    This is the token-level placement: the image parts go between the two text
    halves. `fraction=0` is prefix and `fraction=1` is suffix up to the boundary
    tokenization; the exact prompt is otherwise the same.
    """
    try:
        enc = tokenizer(user, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError as exc:
        raise ValueError("image_placement='token' needs a fast tokenizer with "
                         "offset_mapping; use image_placement='split' instead") from exc
    offsets = enc.get("offset_mapping")
    if not offsets:
        raise ValueError("image_placement='token' needs a tokenizer that returns "
                         "offset_mapping; use image_placement='split' instead")
    n = len(offsets)
    index = min(max(int(fraction * n), 0), n)
    char = offsets[index][0] if index < n else len(user)
    return user[:char], user[char:]


class HFVLMBackend:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16",
                 batch_size: int = 4, max_pixels: Optional[int] = None,
                 trust_remote_code: bool = False, revision: Optional[str] = None,
                 image_placement: str = "prefix", image_position: Optional[float] = None,
                 image_splice: Optional[str] = None, **model_kwargs):
        """`max_pixels` caps the image token budget (Qwen3-VL's own processor default is
        large). `image_placement` is "prefix" (default, the pre-0.4 rendering),
        "suffix" (image parts after the text), "split" (insert the images at the
        first `anyjev.state.IMAGE_SPLIT` sentinel in the text), "token" (insert
        them at `image_position` of the text token count), or "splice" /
        "splice_token" (move the image-token block in `input_ids` to an
        `image_splice` boundary or token fraction; a multi-image span moves as
        one block). `model_kwargs` go to `from_pretrained` unchanged, as in
        `HFBackend`."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if image_placement not in IMAGE_PLACEMENTS:
            raise ValueError(f"image_placement must be one of {IMAGE_PLACEMENTS}")
        if image_placement in ("token", "splice_token") and (image_position is None
                                                             or not 0.0 <= image_position <= 1.0):
            raise ValueError("image_placement='token'/'splice_token' needs image_position in [0, 1]")
        if image_placement == "splice" and image_splice not in ("middle", "question", "suffix"):
            raise ValueError("image_placement='splice' needs image_splice in "
                             "{'middle', 'question', 'suffix'}")
        self.name = model_name
        self.image_placement = image_placement
        self.image_position = image_position
        self.image_splice = image_splice
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_pixels = max_pixels
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, revision=revision)
        self.tokenizer = self.processor.tokenizer
        # left padding puts the last real token in the last column of every row
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if max_pixels is not None:
            size = self.processor.image_processor.size
            if hasattr(size, "max_pixels"):
                size.max_pixels = int(max_pixels)
            else:
                size["max_pixels"] = int(max_pixels)

        torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
        kw = dict(model_kwargs)
        kw.setdefault("dtype", torch_dtype)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, revision=revision, **kw)
        if "device_map" not in kw:
            self.model = self.model.to(device)
        self.model.eval()
        self._pending: Dict[str, Tuple[str, Tuple[Any, ...]]] = {}
        self._next_id = 0
        self._image_cache: Dict[str, Any] = {}

    # ---- prompt rendering -------------------------------------------------
    def _load(self, path: str):
        img = self._image_cache.get(path)
        if img is None:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            if len(self._image_cache) > 1024:
                self._image_cache.clear()
            self._image_cache[path] = img
        return img

    def render_chat_parts(self, spec) -> Tuple[str, str]:
        """Render `spec` with the VLM chat template and register its images.

        Returns `(prompt, unique_marker)`. The concatenation is the key the Decider
        uses for this prompt, so the marker keeps two different images with the same
        text from colliding; `next_token_logprobs` strips it before the model sees it.

        `image_placement` controls where the image content parts sit relative to the
        user text: "prefix" (default), "suffix", "split" at `IMAGE_SPLIT`, or
        "token" at `image_position` of the text token count. Only the placement
        changes; the text and the label readout do not.
        """
        placement = getattr(self, "image_placement", "prefix")
        if placement not in IMAGE_PLACEMENTS:
            raise ValueError(f"image_placement must be one of {IMAGE_PLACEMENTS}")
        paths = tuple(VISUAL_MARK_RE.findall(spec.user))
        text = VISUAL_MARK_RE.sub("", spec.user)
        text = "\n".join(line for line in text.split("\n") if line.strip())
        if placement != "split" or not paths:
            text = text.replace(IMAGE_SPLIT, "")
        system, user = spec.system, text
        template = self.processor.chat_template or getattr(self.tokenizer, "chat_template", "") or ""
        # LLaVA-NeXT's template indexes content parts (`content[0]['text']`) and
        # filters on `type`; it needs lists even for text-only messages. Other
        # templates accept plain strings, which is the older path.
        structured = needs_structured_content(template)
        if structured:
            sys_content: Any = [{"type": "text", "text": system}]
            user_content: Any = [{"type": "text", "text": user}]
        else:
            sys_content, user_content = system, user
        if paths:
            image_parts = [{"type": "image", "image": self._load(p)} for p in paths]
            if placement == "token":
                fraction = getattr(self, "image_position", None)
                if fraction is None or not 0.0 <= float(fraction) <= 1.0:
                    raise ValueError("image_placement='token' needs image_position in [0, 1]")
                before, after = _split_at_token(user, float(fraction), self.tokenizer)
                content: List[Dict[str, Any]] = []
                if before:
                    content.append({"type": "text", "text": before})
                content.extend(image_parts)
                if after:
                    content.append({"type": "text", "text": after})
            else:
                content = _content_parts(user, image_parts, placement)
            messages = [{"role": "system", "content": sys_content},
                        {"role": "user", "content": content}]
        else:
            messages = [{"role": "system", "content": sys_content},
                        {"role": "user", "content": user_content}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if "enable_thinking" in template:   # only pass it to templates that declare it
            kwargs["enable_thinking"] = False
        prompt = self.processor.apply_chat_template(messages, **kwargs)
        self._next_id += 1
        key = str(self._next_id)
        self._pending[key] = (prompt, tuple(self._load(p) for p in paths))
        return prompt, _ID_MARK + key

    # ---- scoring ----------------------------------------------------------
    def _inputs(self, texts: Sequence[str], image_lists: Sequence[Sequence[Any]]):
        kwargs: Dict[str, Any] = dict(text=list(texts), return_tensors="pt", padding=True)
        if image_lists:
            if all(len(imgs) == 1 for imgs in image_lists):
                kwargs["images"] = [imgs[0] for imgs in image_lists]
            else:
                kwargs["images"] = [list(imgs) for imgs in image_lists]
        return self.processor(**kwargs)

    # The latter phrases in each tuple are the pointwise readout's boundaries:
    # one option is shown alone, so there is no options list and the answer cue
    # is the yes/no sentence. Main-prompt phrases are tried first.
    _SPLICE_PHRASES = {
        "middle": ("Question:", " Question:", "\nQuestion:", "Question", " Question"),
        "question": ("Options:", " Options:", "\nOptions:", "Options", " Options",
                     "Option:", "\nOption:", " Option:"),
        "suffix": ("Answer with the letter only.", "Answer with the letter only",
                   "Answer with the letter", "Answer yes or no.",
                   "Answer yes or no", "Answer yes"),
    }

    def _splice_encodings(self, enc):
        """Move the image-token block to a text boundary in `input_ids`.

        This is for templates (LLaVA) that force the image first in the content
        parts. The image token count and the pixel values are unchanged; only
        the positions of the image tokens move, so the model merges the same
        visual features at the new location.
        """
        import torch

        ids = enc["input_ids"]
        image_id = getattr(self.processor, "image_token_id", None)
        if image_id is None:
            token = getattr(self.processor, "image_token", None)
            if token is None:
                raise ValueError("splice needs processor.image_token_id or image_token")
            image_id = self.tokenizer.convert_tokens_to_ids(token)
        image_id = int(image_id)
        if self.image_placement == "splice_token":
            fraction = float(self.image_position)
            for row in range(ids.shape[0]):
                moved = _move_image_block_fraction(ids[row].tolist(), image_id,
                                                   fraction, allow_gaps=True)
                ids[row] = torch.tensor(moved, dtype=ids.dtype, device=ids.device)
            enc["input_ids"] = ids
            return enc
        after = self.image_splice == "suffix"
        candidates = self._SPLICE_PHRASES[self.image_splice]
        for row in range(ids.shape[0]):
            row_ids = ids[row].tolist()
            last_error = None
            for phrase in candidates:
                try:
                    moved = _move_image_block_text(row_ids, image_id, phrase,
                                                   self.tokenizer, after,
                                                   allow_gaps=True)
                except ValueError as exc:
                    last_error = exc
                    continue
                ids[row] = torch.tensor(moved, dtype=ids.dtype, device=ids.device)
                break
            else:
                raise ValueError(f"could not splice image at {self.image_splice}: {last_error}")
        enc["input_ids"] = ids
        return enc

    def next_token_logprobs(self, prompts: Sequence[str],
                            token_ids: Sequence[Sequence[int]]) -> List[np.ndarray]:
        import torch

        n = len(prompts)
        out: List[Optional[np.ndarray]] = [None] * n
        items = []
        for i, p in enumerate(prompts):
            if _ID_MARK in p:
                text, key = p.rsplit(_ID_MARK, 1)
                stored = self._pending.pop(key, None)
                prompt, images = stored if stored is not None else (text, ())
            else:
                prompt, images = p, ()
            items.append((i, prompt, images))

        by_count: Dict[int, List[tuple]] = {}
        for item in items:
            by_count.setdefault(len(item[2]), []).append(item)
        for count, group in by_count.items():
            for start in range(0, len(group), self.batch_size):
                chunk = group[start:start + self.batch_size]
                enc = self._inputs([c[1] for c in chunk],
                                   [c[2] for c in chunk] if count else [])
                enc = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in enc.items()}
                if getattr(self, "image_placement", None) in ("splice", "splice_token"):
                    enc = self._splice_encodings(enc)
                with torch.no_grad():
                    try:
                        logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
                    except TypeError:   # forward without logits_to_keep
                        logits = self.model(**enc).logits[:, -1, :].float()
                lp = torch.log_softmax(logits, dim=-1)
                for row, (i, _, _) in enumerate(chunk):
                    ids = torch.as_tensor(list(token_ids[i]), device=lp.device)
                    out[i] = lp[row, ids].cpu().numpy().astype(np.float64)
        return out  # type: ignore[return-value]
