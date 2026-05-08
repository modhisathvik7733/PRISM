"""bAbI data loader.

Tries multiple sources in order:
  1. Cached local extraction (~/.cache/prism_lang/babi/)
  2. The original Facebook tarball mirror (currently returns 403)
  3. HuggingFace `datasets` library (`facebook/babi_qa`) if installed
  4. Procedural synthetic generator — Task 1 only, since its grammar is
     trivial enough that synthetic data is operationally equivalent to
     the real dataset for testing the architecture.

The synthetic fallback unblocks Phase 1 (Task 1) regardless of network
state. Phase 2 (all 20 tasks) requires sources 1-3 to succeed.

bAbI file format (when extracted):
    Lines start with a story-local sentence number. Sentence 1 starts a
    new story; subsequent sentences extend it. Lines without TAB are
    facts; lines with TAB are questions ("question \\t answer \\t
    supporting_fact_ids"). For training we treat each question as one
    example, with `story_text` = all preceding facts in the same story
    (cumulative).
"""

from __future__ import annotations

import random
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


def _try_hf_datasets(task_id: int, split: str
                     ) -> list[tuple[str, str, str]] | None:
    """Try to load via HuggingFace `datasets`. Returns None if the
    package or the dataset entry isn't available."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    # facebook/babi_qa exposes configs like "en-10k-qa1_single-supporting-fact".
    config = f"en-10k-{BABI_TASK_NAMES[task_id]}"
    try:
        ds = load_dataset(
            "facebook/babi_qa", config, split=split, trust_remote_code=True,
        )
    except Exception:
        return None
    examples: list[tuple[str, str, str]] = []
    # The HF schema bundles each story as a list of dicts with
    # 'text' and 'type' (fact vs question). We replay them in order.
    for row in ds:
        story_so_far: list[str] = []
        for sent in row["story"]:
            text = sent["text"].strip()
            if sent.get("type", 0) == 0:  # 0 = fact
                story_so_far.append(text)
            else:
                # question: row also carries answer
                examples.append((
                    " ".join(story_so_far),
                    text,
                    sent["answer"].strip(),
                ))
    return examples


def _gen_synthetic_t1(split: str) -> list[tuple[str, str, str]]:
    """Procedural bAbI Task 1 generator (single supporting fact).

    The real Task 1 has the grammar:
        "<Person> moved/went to the <Location>." × N
        "Where is <Person>?"  →  <last Location seen for that Person>

    Synthetic with the same vocab is operationally equivalent for testing
    the architecture's reasoning capacity. Distinct seeds for train/test
    so they don't leak.

    Sized to the real bAbI 10k variant (10k train / 1k test). The 1k
    variant is too small for a 24M-param model — the model memorizes
    train at ce=0 by step ~2000 and test acc stalls at 40-50%. With 10k
    train the same model generalizes properly because no individual
    (story, question) pattern can be memorized to convergence.
    """
    rng = random.Random({"train": 0, "test": 1}[split])
    n = {"train": 10000, "test": 1000}[split]
    # Same closed vocab the real Task 1 uses (8 locations, 4 people).
    LOCATIONS = ("bathroom", "kitchen", "bedroom", "garden",
                 "hallway", "office", "park", "school")
    PEOPLE = ("Mary", "John", "Daniel", "Sandra")
    VERBS = ("moved to", "went to", "travelled to", "journeyed to")
    examples: list[tuple[str, str, str]] = []
    for _ in range(n):
        n_sentences = rng.randint(2, 6)
        last_loc: dict[str, str] = {}
        sents: list[str] = []
        for _ in range(n_sentences):
            p = rng.choice(PEOPLE)
            l = rng.choice(LOCATIONS)
            v = rng.choice(VERBS)
            sents.append(f"{p} {v} the {l}.")
            last_loc[p] = l
        target = rng.choice(list(last_loc.keys()))
        examples.append((
            " ".join(sents),
            f"Where is {target}?",
            last_loc[target],
        ))
    return examples


def load_babi(task_id: int, split: str = "train",
              cache_dir: Path | None = None) -> list[tuple[str, str, str]]:
    """Load one bAbI task. Returns a list of (story, question, answer).

    Source priority:
      1. Local extracted cache (instant if previously downloaded)
      2. Direct download from Facebook mirror (currently 403 — may come back)
      3. HuggingFace `datasets` library (any task)
      4. Procedural synthetic generator (Task 1 only)

    `task_id` ∈ 1..20.  `split` ∈ {"train", "test"}.  `cache_dir`
    overrides the default ~/.cache/prism_lang/babi.
    """
    if task_id not in BABI_TASK_NAMES:
        raise ValueError(f"task_id must be 1..20, got {task_id}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split}")
    cache_dir = cache_dir or BABI_LOCAL_CACHE

    # 1 & 2: try the cached/downloaded tarball first.
    try:
        en_dir = _download_and_extract(cache_dir)
        fname = f"{BABI_TASK_NAMES[task_id]}_{split}.txt"
        path = en_dir / fname
        if path.exists():
            return list(_parse_babi_file(path))
    except Exception as e:
        print(f"[babi] direct download failed ({e!r}); trying HF datasets…")

    # 3: HuggingFace datasets.
    hf_examples = _try_hf_datasets(task_id, split)
    if hf_examples is not None and len(hf_examples) > 0:
        print(f"[babi] loaded via HF datasets: {len(hf_examples)} {split} examples")
        return hf_examples

    # 4: synthetic fallback (Task 1 only — other tasks have richer grammar
    # we won't fake here).
    if task_id == 1:
        ex = _gen_synthetic_t1(split)
        print(f"[babi] using synthetic Task 1 fallback: "
              f"{len(ex)} {split} examples")
        return ex

    raise RuntimeError(
        f"could not load bAbI task {task_id} {split} from any source. "
        "Try `uv pip install datasets` or place the tarball at "
        f"{cache_dir}/tasks_1-20_v1-2.tar.gz manually."
    )


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
