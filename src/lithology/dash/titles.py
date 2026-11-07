"""Short, readable log titles from the dataset's very mixed file names.

``MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft`` -> ``MC765 P011 ST00BP00``,
``60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400`` -> ``Shenzi SB203 ST00BP01``.
The full file name is always shown next to the title, so a rough title is harmless.
"""

from __future__ import annotations

import re

_BW_SERIES = re.compile(r"^0000000009_d-582_\d+-\d+_(log\d+)_(.*)$", re.I)
_API = re.compile(r"^(\d{10,14})(?:\s*-\s*|[_\s]+)")
_DROP = {
    "mudlog", "mud", "log", "masterlog", "master", "ml", "combo", "final", "rev", "depth", "md", "tvd", "show",
    "pdf", "vertical", "to", "td", "mms", "tvdmudlog", "inch", "in", "md5", "tvd5", "1md", "5md", "1tvd",
}  # fmt: skip
_SCALE = re.compile(r"^\d+(in|inch)(md|tvd)?$|^1[-:]\d{2,4}$|^1in(md|tvd)$", re.I)
_RANGE = re.compile(r"^\d+(ft)?-\d+(ft)?$|^\d+ft$|^\d+-rev$", re.I)
_UPPER = {"ocs", "bp", "st", "td"}
_CODE = re.compile(r"[a-z].*\d|\d.*[a-z]", re.I)


def api_number(doc_id: str) -> str | None:
    m = re.match(r"^(\d{10,14})", doc_id)
    return m.group(1) if m and not doc_id.startswith("0000000009") else None


def _word(tok: str) -> str:
    if _CODE.search(tok) or len(tok) <= 2 or tok.lower() in _UPPER or ("-" in tok and len(tok) <= 6):
        return tok.upper()
    return tok[:1].upper() + tok[1:].lower() if tok.islower() or tok.isupper() else tok


def short_title(doc_id: str) -> str:
    name = re.sub(r"\.pdf$", "", doc_id, flags=re.I)
    m = _BW_SERIES.match(name)
    if m:
        return f"D-582 {m.group(1).lower()}"
    api = api_number(name)
    rest = _API.sub("", name, count=1) if api else name
    rest = re.sub(r"^\d{12}[_\s]", "", rest)  # a repeated 12-digit API right after the 14-digit one
    rest = re.sub(r"_+", " ", rest)
    rest = re.sub(r"^mudlog[\-\s]+(md\s+)?", "", rest, flags=re.I)
    rest = re.sub(r"\b\d+\s*inch\b", " ", rest, flags=re.I)  # "5 inch"
    rest = re.sub(r"\b(md|show|in)\s+\d$", "", rest, flags=re.I)  # version suffix "_2"
    rest = re.sub(r"\b\d+\s+in\s+md\b", " ", rest, flags=re.I)  # "5 in_md"
    words: list[str] = []
    for raw in re.split(r"[_\s]+", rest):
        if not raw:
            continue
        parts = raw.split("-")
        # split "shenzi-sb203-st00bp01" but keep "mdn-1", "3-33", "201-1807H", "ocs-g"
        splittable = len(parts) > 1 and all(re.search(r"[a-z]", p, re.I) and len(p) > 2 for p in parts)
        toks = parts if splittable else [raw]
        for tok in toks:
            low = tok.lower()
            if low in _DROP or _SCALE.match(tok) or _RANGE.match(tok):
                continue
            if tok.isdigit() and (len(tok) > 3 or tok.startswith("0")):
                prev = words[-1].lower() if words else ""
                if not prev.endswith("g"):  # keep lease numbers ("OCS-G 26419"), drop depths and padding
                    continue
            words.append(_word(tok))
            if len(words) >= 5:
                break
        if len(words) >= 5:
            break
    if not words:
        return f"Well {api[:10]}" if api else name[:40]
    return " ".join(words)


def scale_label(inches_per_100ft: float | None) -> str | None:
    if not inches_per_100ft:
        return None
    v = round(inches_per_100ft, 1)
    return f"{v:g} in / 100 ft"
