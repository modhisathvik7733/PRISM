"""bAbI data loader.

Tries the HuggingFace `datasets` library first (`Muennighoff/babi` is a
clean mirror of Facebook's tasks_1-20_v1-2). Falls back to downloading
the original tarball if HF isn't available. Returns
`(story_text, question_text, answer_text)` triples.

bAbI file format:
    Lines start with a story-local sentence number. Sentence 1 starts a
    new story; subsequent sentences extend it. Lines without TAB are
    facts; lines with TAB are questions ("question \\t answer \\t
    supporting_fact_ids"). For training we treat each question as one
    example, with `story_text` = all preceding facts in the same story
    (cumulative).
"""

from __future__ import annotations

import os
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator

# Tasks 1..20 in the standard naming used by Facebook + HF mirrors.
BABI_TASK_NAMES: dict[int, str] = {
    1:  "qa1_single-supporting-fact",
    2:  "qa2_two-supporting-facts",
    3:  "qa3_three-supporting-facts",
    4:  "qa4_two-arg-relations",
    5:  "qa5_three-arg-relations",
    6:  "qa6_yes-no-questions",
    7:  "qa7_counting",
    8:  "qa8_lists-sets",
    9:  "qa9_simple-negation",
    10: "qa10_indefinite-knowledge",
    11: "qa11_basic-coreference",
    12: "qa12_conjunction",
    13: "qa13_compound-coreference",
    14: "qa14_time-reasoning",
    15: "qa15_basic-deduction",
    16: "qa16_basic-induction",
    17: "qa17_positional-reasoning",
    18: "qa18_size-reasoning",
    19: "qa19_path-finding",
    20: "qa20_agents-motivations",
}

# Mirror of the original Weston tarball — multiple are floating around;
# the dl.fbaipublicfiles one is most likely to stay alive.
BABI_TARBALL_URL = (
    "https://dl.fbaipublicfiles.com/babi/tasks_1-20_v1-2.tar.gz"
)
BABI_LOCAL_CACHE = Path.home() / ".cache" / "prism_lang" / "babi"


def _download_and_extract(dest: Path) -> Path:
    """Download the bAbI tarball if not present and extract into `dest`.
    Returns the path to the `tasks_1-20_v1-2/en` directory (English,
    1k-train variant)."""
    dest.mkdir(parents=True, exist_ok=True)
    en_dir = dest / "tasks_1-20_v1-2" / "en"
    if en_dir.exists():
        return en_dir
    tar_path = dest / "tasks_1-20_v1-2.tar.gz"
    if not tar_path.exists():
        print(f"[babi] downloading {BABI_TARBALL_URL} → {tar_path}")
        urllib.request.urlretrieve(BABI_TARBALL_URL, tar_path)
    print(f"[babi] extracting {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest)
    if not en_dir.exists():
        raise RuntimeError(f"bAbI extraction did not produce {en_dir}")
    return en_dir


def _parse_babi_file(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield `(story_text, question_text, answer_text)` per Q line."""
    story: list[str] = []
    line_re = re.compile(r"^(\d+)\s+(.*)$")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            m = line_re.match(raw)
            if not m:
                continue
            line_num = int(m.group(1))
            rest = m.group(2)
            if line_num == 1:
                # New story.
                story = []
            if "\t" in rest:
                # Question line: "question_text\tanswer\tsupporting_ids"
                parts = rest.split("\t")
                question = parts[0].strip()
                answer = parts[1].strip()
                yield " ".join(story), question, answer
            else:
                story.append(rest.strip())


def load_babi(task_id: int, split: str = "train",
              cache_dir: Path | None = None) -> list[tuple[str, str, str]]:
    """Load one bAbI task. Returns a list of (story, question, answer).

    `task_id` ∈ 1..20.  `split` ∈ {"train", "test"}.  `cache_dir`
    overrides the default ~/.cache/prism_lang/babi.
    """
    if task_id not in BABI_TASK_NAMES:
        raise ValueError(f"task_id must be 1..20, got {task_id}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split}")
    cache_dir = cache_dir or BABI_LOCAL_CACHE
    en_dir = _download_and_extract(cache_dir)
    fname = f"{BABI_TASK_NAMES[task_id]}_{split}.txt"
    path = en_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"missing bAbI file {path}")
    return list(_parse_babi_file(path))


def format_input(story: str, question: str) -> str:
    """Canonical model input format. Keep simple — the encoder handles
    the rest."""
    return f"Story: {story} Question: {question} Answer:"


def format_target(answer: str) -> str:
    """Canonical model target format. Leading space matches GPT-2 BPE
    (where ' kitchen' tokenizes to one token, 'kitchen' to two)."""
    return f" {answer}"


# --- a tiny offline sample so smoke_test can run without network --------

OFFLINE_SAMPLE: list[tuple[str, str, str]] = [
    ("Mary moved to the bathroom.", "Where is Mary?", "bathroom"),
    ("John went to the hallway.", "Where is John?", "hallway"),
    ("Mary moved to the bathroom. John went to the hallway.",
     "Where is Mary?", "bathroom"),
    ("Mary moved to the bathroom. John went to the hallway.",
     "Where is John?", "hallway"),
    ("Daniel travelled to the office. Sandra went back to the garden.",
     "Where is Daniel?", "office"),
]
