from __future__ import annotations

from dataclasses import dataclass
import sys

try:
    from .config import METHODS_DIR
except ImportError:
    from config import METHODS_DIR

if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

from seek_code import SEEK  # noqa: E402


@dataclass(frozen=True)
class SeekAttribute:
    name: str
    choices: list[str]
    group: str


RIGHT_ATTRIBUTES = {"R_tear_1", "R_hole_1", "R_tear_2", "R_hole_2"}
LEFT_ATTRIBUTES = {"L_tear_1", "L_hole_1", "L_tear_2", "L_hole_2"}


def attribute_group(name: str) -> str:
    if name in RIGHT_ATTRIBUTES:
        return "right ear"
    if name in LEFT_ATTRIBUTES:
        return "left ear"
    return "whole image"


ATTRIBUTES = [
    SeekAttribute(name, list(SEEK.mappings[name]), attribute_group(name))
    for name in SEEK.attribute_names
]


def parse_seek(code):
    return SEEK(str(code))


def normalize_seek(code) -> str:
    return str(parse_seek(code))


def is_valid_seek(code) -> bool:
    try:
        normalize_seek(code)
        return True
    except Exception:
        return False


def seek_to_attrs(code) -> dict[str, str]:
    seek = parse_seek(code)
    return {name: getattr(seek, name) for name in SEEK.attribute_names}


def attrs_to_seek(attrs: dict[str, str]) -> str:
    parts = {name: attrs[name] for name in SEEK.attribute_names}
    code = (
        f"{parts['sex']}{parts['age']}T{parts['right_tusk']}{parts['left_tusk']}"
        f"E{parts['R_tear_1']}{parts['R_hole_1']}{parts['R_tear_2']}{parts['R_hole_2']}-"
        f"{parts['L_tear_1']}{parts['L_hole_1']}{parts['L_tear_2']}{parts['L_hole_2']}"
        f"X{parts['right_extreme']}{parts['left_extreme']}S{parts['ear_special']}{parts['body_special']}"
    )
    return normalize_seek(code)


def seek_to_one_hot(code):
    return parse_seek(code).one_hot_encode()


def one_hot_to_seek(one_hot) -> str:
    return str(SEEK(one_hot))

