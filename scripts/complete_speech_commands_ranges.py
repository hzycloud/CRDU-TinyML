import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.aria2_control import parse_aria2_control


URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
EXPECTED_BYTES = 2428923189
EXPECTED_MD5 = "6b74f3901214cb2c2934e98196829835"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "datasets" / "speech_commands" / "speech_commands_v0.02.tar.gz",
    )
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_piece(
    index: int,
    start: int,
    length: int,
    archive_path: Path,
    temp_dir: Path,
    write_lock: threading.Lock,
) -> int:
    temp_path = temp_dir / f"piece_{index:06d}.bin"
    command = [
        "curl.exe",
        "-4",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "12",
        "--retry-delay",
        "3",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--max-time",
        "180",
        "--range",
        f"{start}-{start + length - 1}",
        "-o",
        str(temp_path),
        URL,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"piece {index} failed: {result.stderr.strip()}"
        )
    actual_length = temp_path.stat().st_size
    if actual_length != length:
        raise RuntimeError(
            f"piece {index} has {actual_length} bytes; expected {length}"
        )
    payload = temp_path.read_bytes()
    with write_lock:
        with archive_path.open("r+b", buffering=0) as stream:
            stream.seek(start)
            stream.write(payload)
    temp_path.unlink()
    return length


def main() -> None:
    args = parse_args()
    control_path = Path(str(args.archive) + ".aria2")
    status_path = args.archive.parent / "download_status.json"
    if not args.archive.exists() or not control_path.exists():
        raise FileNotFoundError("archive or aria2 control file is missing")

    control = parse_aria2_control(control_path.read_bytes())
    if control["total_length"] != EXPECTED_BYTES:
        raise ValueError("aria2 control total length is not the official size")
    missing = control["missing_pieces"]
    piece_length = control["piece_length"]
    total_length = control["total_length"]
    initial = {
        "missing_pieces": len(missing),
        "completed_pieces": len(control["completed_pieces"]),
        "piece_count": control["piece_count"],
        "piece_length": piece_length,
    }
    print(json.dumps(initial, indent=2))
    if not missing:
        actual_md5 = file_md5(args.archive)
        if args.archive.stat().st_size != EXPECTED_BYTES or actual_md5 != EXPECTED_MD5:
            raise RuntimeError("no missing pieces but archive verification failed")
        control_path.unlink()
        status_path.write_text(
            json.dumps(
                {
                    "verified": True,
                    "bytes": args.archive.stat().st_size,
                    "md5": actual_md5,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    temp_dir = args.archive.parent / "range_parts"
    temp_dir.mkdir(exist_ok=True)
    write_lock = threading.Lock()
    tasks = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index in missing:
            start = index * piece_length
            length = min(piece_length, total_length - start)
            future = executor.submit(
                download_piece,
                index,
                start,
                length,
                args.archive,
                temp_dir,
                write_lock,
            )
            tasks[future] = index

        completed_bytes = 0
        completed_pieces = 0
        for future in tqdm(
            as_completed(tasks),
            total=len(tasks),
            desc="missing pieces",
        ):
            completed_bytes += future.result()
            completed_pieces += 1
            if completed_pieces % 25 == 0:
                print(
                    json.dumps(
                        {
                            "repaired_pieces": completed_pieces,
                            "repaired_mib": round(completed_bytes / 1024**2, 2),
                        }
                    ),
                    flush=True,
                )

    with args.archive.open("r+b") as stream:
        stream.truncate(total_length)
    actual_bytes = args.archive.stat().st_size
    actual_md5 = file_md5(args.archive)
    verified = actual_bytes == EXPECTED_BYTES and actual_md5 == EXPECTED_MD5
    status = {
        "verified": verified,
        "bytes": actual_bytes,
        "expected_bytes": EXPECTED_BYTES,
        "md5": actual_md5,
        "expected_md5": EXPECTED_MD5,
        "repaired_pieces": len(missing),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    if not verified:
        raise RuntimeError("archive verification failed after range repair")
    control_path.unlink()


if __name__ == "__main__":
    main()
