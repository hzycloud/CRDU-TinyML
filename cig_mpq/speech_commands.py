import random
from pathlib import Path


TARGET_WORDS = (
    "down",
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "yes",
)
WORD_TO_LABEL = {word: index for index, word in enumerate(TARGET_WORDS)}
SILENCE_LABEL = 10
UNKNOWN_LABEL = 11


def _read_split_list(root: Path, filename: str) -> set[str]:
    path = root / filename
    if not path.exists():
        return set()
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _limit(items: list[dict], limit: int | None, rng: random.Random) -> list[dict]:
    items = sorted(items, key=lambda item: item["path"])
    if limit is not None and len(items) > limit:
        return sorted(rng.sample(items, limit), key=lambda item: item["path"])
    return items


def build_speech_commands_manifest(
    root: str | Path,
    split: str,
    seed: int,
    samples_per_class: int | None = None,
) -> list[dict]:
    root = Path(root)
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")

    validation = _read_split_list(root, "validation_list.txt")
    testing = _read_split_list(root, "testing_list.txt")
    excluded_from_train = validation | testing
    word_directories = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name != "_background_noise_"
    )

    all_word_files = {
        path.relative_to(root).as_posix(): path
        for directory in word_directories
        for path in directory.glob("*.wav")
    }
    if split == "validation":
        selected_relative = validation
    elif split == "test":
        selected_relative = testing
    else:
        selected_relative = set(all_word_files) - excluded_from_train

    rng = random.Random(f"{seed}:{split}")
    rows = []
    for word in TARGET_WORDS:
        items = [
            {
                "path": str(all_word_files[relative]),
                "label": WORD_TO_LABEL[word],
                "class_name": word,
                "kind": "word",
            }
            for relative in sorted(selected_relative)
            if relative in all_word_files
            and relative.split("/", 1)[0] == word
        ]
        rows.extend(_limit(items, samples_per_class, rng))

    unknown_items = [
        {
            "path": str(all_word_files[relative]),
            "label": UNKNOWN_LABEL,
            "class_name": "unknown",
            "kind": "word",
        }
        for relative in sorted(selected_relative)
        if relative in all_word_files
        and relative.split("/", 1)[0] not in TARGET_WORDS
    ]
    rows.extend(_limit(unknown_items, samples_per_class, rng))

    background_files = sorted((root / "_background_noise_").glob("*.wav"))
    silence_count = (
        samples_per_class
        if samples_per_class is not None
        else max(1, len(background_files))
    )
    if background_files and silence_count:
        for index in range(silence_count):
            path = background_files[index % len(background_files)]
            rows.append(
                {
                    "path": str(path),
                    "label": SILENCE_LABEL,
                    "class_name": "silence",
                    "kind": "silence",
                    "sample_seed": rng.randrange(0, 2**31),
                }
            )

    return sorted(rows, key=lambda row: (row["label"], row["path"], row.get("sample_seed", 0)))
