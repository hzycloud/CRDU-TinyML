import argparse
import json
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def validate_wav(path: Path) -> tuple[bool, str | None]:
    try:
        with wave.open(str(path), "rb") as stream:
            if stream.getnchannels() != 1:
                return False, "not_mono"
            if stream.getsampwidth() != 2:
                return False, "not_pcm16"
            if stream.getframerate() != 16000:
                return False, f"sample_rate_{stream.getframerate()}"
            if stream.getnframes() <= 0:
                return False, "empty_audio"
    except Exception as exc:
        return False, repr(exc)
    return True, None


def read_list_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    required = [
        args.root / "validation_list.txt",
        args.root / "testing_list.txt",
        args.root / "_background_noise_",
        *[args.root / word for word in TARGET_WORDS],
    ]
    missing_required = [str(path) for path in required if not path.exists()]
    validation_entries = read_list_entries(args.root / "validation_list.txt")
    testing_entries = read_list_entries(args.root / "testing_list.txt")
    missing_listed = [
        entry
        for entry in validation_entries + testing_entries
        if not (args.root / entry).exists()
    ]

    wav_paths = sorted(args.root.rglob("*.wav"))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(validate_wav, wav_paths)
        for path, (valid, error) in tqdm(
            zip(wav_paths, results),
            total=len(wav_paths),
            desc="validating WAV files",
        ):
            if not valid:
                failures.append({"path": str(path), "error": error})
                if len(failures) >= 100:
                    break

    output = {
        "root": str(args.root),
        "wav_files": len(wav_paths),
        "validation_list_entries": len(validation_entries),
        "testing_list_entries": len(testing_entries),
        "missing_required": missing_required,
        "missing_listed": missing_listed[:100],
        "missing_listed_count": len(missing_listed),
        "wav_failures": failures,
        "wav_failure_count_capped": len(failures),
        "passed": (
            not missing_required
            and not missing_listed
            and len(wav_paths) > 100000
            and not failures
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
