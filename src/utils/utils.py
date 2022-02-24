import dataclasses
import enum
import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import List


class Target(enum.Enum):
    VOL = enum.auto()
    ANI = enum.auto()


class DataType(enum.Enum):
    LABELED = enum.auto()
    UNLABELED = enum.auto()
    ALL = enum.auto()


class Split(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()


@dataclasses.dataclass
class SeedWord:
    text: str
    vol_label: int


@dataclasses.dataclass
class Event:
    path: str = ""
    sid: str = ""
    text: str = ""
    tokenized_text: str = ""
    tokenized_sampled_text: List[str] = dataclasses.field(
        default_factory=list
    )  # Used for SOC
    full_text: str = ""
    tokenized_full_text: str = ""
    vol_label: float = -100
    vol_phrase: str = ""
    ani_label: float = -100
    ani_phrase: str = ""
    pred_phrase: str = ""
    pred_rep: str = ""


def read_seed_lex(path: Path) -> List[SeedWord]:
    seed_words = []
    with open(path, "rt", errors="replace") as f:
        for line in f:
            if line.strip():
                label, text = line.strip().split("\t")
                seed_words.append(SeedWord(text, float(label)))
    return seed_words


def read_event_file(path: Path) -> List[Event]:
    ext = path.suffix
    if ext == ".gz":
        open_ = gzip.open
    else:
        open_ = open
    with open_(path, "rt", errors="replace") as f:
        return [Event(**json.loads(line)) for line in f if line.strip()]


def save_event_file(path: Path, events: List[Event], use_gzip: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    file = gzip.open(path, "wt") if use_gzip else open(path, "wt")
    file.write(
        "\n".join(json.dumps(asdict(event), ensure_ascii=False) for event in events)
        + "\n"
    )
    file.close()
