"""Download raw corpora only; rerun the same command to resume verified files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import Request, urlopen

from filelock import FileLock
from huggingface_hub import HfApi, RepoFile, hf_hub_download

GB = 1_000_000_000
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "corpus_raw"
STACK_LANGUAGES = (
    "C",
    "CSharp",
    "Cpp",
    "Go",
    "Java",
    "JavaScript",
    "Markdown",
    "PHP",
    "Python",
    "Ruby",
    "Rust",
    "SQL",
    "Shell",
    "Swift",
    "TypeScript",
)


@dataclass(frozen=True)
class Source:
    repo_id: str
    folders: tuple[str, ...]


SOURCES = {
    "fineweb-edu": Source("HuggingFaceFW/fineweb-edu", ("data",)),
    "fineweb": Source("HuggingFaceFW/fineweb", ("data",)),
    "stack-edu": Source("HuggingFaceTB/stack-edu", STACK_LANGUAGES),
    "finemath": Source("HuggingFaceTB/finemath", ("finemath-4plus",)),
    "infiwebmath": Source("HuggingFaceTB/finemath", ("infiwebmath-4plus",)),
    "megamath-web": Source("IFM/MegaMath", ("megamath-web",)),
    "cosmopedia-v2": Source("HuggingFaceTB/smollm-corpus", ("cosmopedia-v2",)),
    "tinystories": Source("roneneldan/TinyStories", ("data",)),
}


def safe_file(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if (
        not relative
        or parts.is_absolute()
        or ".." in parts.parts
        or "\\" in relative
        or ":" in relative
        or str(parts) != relative
    ):
        raise ValueError(f"unsafe dataset path: {relative}")
    current = root
    for part in ("", *parts.parts):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink in dataset path: {relative}")
    destination = root.joinpath(*parts.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"dataset path escapes output: {relative}")
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_space(
    root: Path, additional: int, reserve_bytes: int, max_bytes: int | None
) -> None:
    existing = root
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < additional + reserve_bytes:
        raise OSError(
            f"need {additional / GB:.2f} GB plus {reserve_bytes / GB:.2f} GB "
            f"headroom; only {free / GB:.2f} GB free"
        )
    if max_bytes is not None:
        used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if used + additional > max_bytes:
            raise OSError("download would exceed --max-gb output budget")


def matches_remote(path: Path, entry: RepoFile) -> bool:
    if not path.is_file() or path.stat().st_size != entry.size:
        return False
    if entry.lfs:
        return sha256_file(path) == entry.lfs.sha256
    digest = hashlib.sha1(f"blob {entry.size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == entry.blob_id


def plan_source(
    api: HfApi,
    name: str,
    output: Path,
    max_files: int | None,
    stack_languages: list[str] | None,
) -> tuple[dict, list[RepoFile]]:
    source = SOURCES[name]
    root = output / name
    manifest_path = root / "download-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("purpose") != "raw-download-only"
            or manifest.get("repo_id") != source.repo_id
        ):
            raise ValueError(f"existing manifest has another identity: {name}")
        revision = manifest["revision"]
    else:
        revision = api.dataset_info(source.repo_id).sha
        manifest = {
            "purpose": "raw-download-only",
            "approved_for_training": False,
            "repo_id": source.repo_id,
            "revision": revision,
            "files": {},
        }
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"dataset has no immutable revision: {name}")
    folders = (
        tuple(stack_languages)
        if name == "stack-edu" and stack_languages
        else source.folders
    )

    def parquet_files():
        for folder in folders:
            for entry in api.list_repo_tree(
                source.repo_id,
                path_in_repo=folder,
                recursive=True,
                revision=revision,
                repo_type="dataset",
            ):
                if (
                    isinstance(entry, RepoFile)
                    and entry.path.startswith(folder + "/")
                    and entry.path.endswith(".parquet")
                ):
                    safe_file(root, entry.path)
                    yield entry

    files = list(islice(parquet_files(), max_files))
    if not files:
        raise ValueError(f"no Parquet shards found for {name}")
    cards = api.get_paths_info(
        source.repo_id, ["README.md"], revision=revision, repo_type="dataset"
    )
    cards = [
        entry
        for entry in cards
        if isinstance(entry, RepoFile) and entry.path == "README.md"
    ]
    if len(cards) != 1:
        raise ValueError(f"missing pinned dataset card: {name}")
    files.extend(cards)
    manifest["selection"] = {"folders": folders, "max_files": max_files}
    manifest["source_complete"] = False
    return manifest, files


def download_file(
    source: Source, revision: str, entry: RepoFile, root: Path
) -> dict[str, Any]:
    destination = safe_file(root, entry.path)
    if not matches_remote(destination, entry):
        print(f"Downloading {entry.path} ({entry.size / GB:.3f} GB)", flush=True)
        downloaded = hf_hub_download(
            repo_id=source.repo_id,
            repo_type="dataset",
            filename=entry.path,
            revision=revision,
            local_dir=root,
            force_download=destination.exists(),
        )
        if Path(str(downloaded)).resolve() != destination.resolve():
            raise ValueError("Hub download returned an unexpected destination")
        safe_file(root, entry.path)
        if not matches_remote(destination, entry):
            raise ValueError(f"downloaded file differs from remote hash: {entry.path}")
    return {"size": entry.size, "sha256": sha256_file(destination)}


def download_code(
    blob_id: str,
    root: Path,
    output: Path,
    reserve_bytes: int,
    max_bytes: int | None,
) -> tuple[Path, int]:
    if not re.fullmatch(r"[0-9a-f]{40}", blob_id):
        raise ValueError("Stack-Edu returned an invalid blob ID")
    destination = safe_file(root, f"code/{blob_id[:2]}/{blob_id}.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".gz.part")
    url = f"https://softwareheritage.s3.amazonaws.com/content/{blob_id}"
    request = Request(url, headers={"User-Agent": "kuno-corpus-download/1.0"})
    with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            check_space(output, len(chunk), reserve_bytes, max_bytes)
            handle.write(chunk)
    digest = hashlib.sha1()
    content_bytes = 0
    with gzip.open(temporary, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            content_bytes += len(chunk)
    if digest.hexdigest() != blob_id:
        raise ValueError(f"Software Heritage content hash mismatch: {blob_id}")
    temporary.replace(destination)
    return destination, content_bytes


def download_stack_code(
    files: list[RepoFile],
    root: Path,
    output: Path,
    reserve_bytes: int,
    max_bytes: int | None,
    max_code_files: int | None,
) -> dict:
    import pyarrow.parquet as pq

    count = 0
    mismatches = 0
    with closing(sqlite3.connect(root / "code-downloads.sqlite3")) as database:
        database.execute(
            "CREATE TABLE IF NOT EXISTS downloads ("
            "blob_id TEXT PRIMARY KEY, size INTEGER, sha256 TEXT, "
            "content_bytes INTEGER, metadata_bytes INTEGER)"
        )
        for entry in files:
            if not entry.path.endswith(".parquet"):
                continue
            parquet = pq.ParquetFile(safe_file(root, entry.path))
            for batch in parquet.iter_batches(
                batch_size=1000, columns=["blob_id", "length_bytes"]
            ):
                for row in batch.to_pylist():
                    blob_id = row["blob_id"]
                    if not isinstance(blob_id, str) or not re.fullmatch(
                        r"[0-9a-f]{40}", blob_id
                    ):
                        raise ValueError("Stack-Edu returned an invalid blob ID")
                    destination = safe_file(root, f"code/{blob_id[:2]}/{blob_id}.gz")
                    saved = database.execute(
                        "SELECT size, sha256, content_bytes FROM downloads "
                        "WHERE blob_id = ?",
                        (blob_id,),
                    ).fetchone()
                    if (
                        saved
                        and destination.is_file()
                        and destination.stat().st_size == saved[0]
                        and sha256_file(destination) == saved[1]
                    ):
                        content_bytes = saved[2]
                    else:
                        destination, content_bytes = download_code(
                            blob_id, root, output, reserve_bytes, max_bytes
                        )
                        database.execute(
                            "INSERT OR REPLACE INTO downloads VALUES (?, ?, ?, ?, ?)",
                            (
                                blob_id,
                                destination.stat().st_size,
                                sha256_file(destination),
                                content_bytes,
                                row["length_bytes"],
                            ),
                        )
                        database.commit()
                    count += 1
                    mismatches += content_bytes != row["length_bytes"]
                    if max_code_files is not None and count >= max_code_files:
                        return {
                            "records_checked": count,
                            "metadata_length_mismatches": mismatches,
                            "complete": False,
                        }
    return {
        "records_checked": count,
        "metadata_length_mismatches": mismatches,
        "complete": True,
    }


def run(args: argparse.Namespace) -> list[dict]:
    names = list(SOURCES) if args.all else list(dict.fromkeys(args.sources))
    output = args.output.resolve()
    reserve_bytes = int(args.reserve_gb * GB)
    max_bytes = int(args.max_gb * GB) if args.max_gb is not None else None
    api = HfApi()
    results = []
    for name in names:
        root = output / name
        manifest, files = plan_source(
            api, name, output, args.max_files, args.stack_languages
        )
        total = sum(entry.size for entry in files)
        needed = sum(
            entry.size
            for entry in files
            if not matches_remote(safe_file(root, entry.path), entry)
        )
        print(
            f"{name}: {len(files) - 1} Parquet shards, {total / GB:.3f} GB; "
            f"{needed / GB:.3f} GB remaining; revision {manifest['revision']}",
            flush=True,
        )
        if name == "stack-edu":
            print(
                "Stack-Edu size excludes separately retrieved code bodies.", flush=True
            )
        if args.dry_run:
            results.append({"source": name, "bytes": total, "remaining_bytes": needed})
            continue
        check_space(output, needed, reserve_bytes, max_bytes)
        root.mkdir(parents=True, exist_ok=True)
        manifest["status"] = "downloading"
        manifest["source_complete"] = False
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        path = root / "download-manifest.json"
        write_manifest(path, manifest)
        try:
            for entry in files:
                destination = safe_file(root, entry.path)
                if not matches_remote(destination, entry):
                    check_space(output, entry.size, reserve_bytes, max_bytes)
                manifest["files"][entry.path] = download_file(
                    SOURCES[name], manifest["revision"], entry, root
                )
                write_manifest(path, manifest)
            code_complete = True
            if name == "stack-edu":
                manifest["code"] = download_stack_code(
                    files,
                    root,
                    output,
                    reserve_bytes,
                    max_bytes,
                    args.max_code_files,
                )
                code_complete = manifest["code"]["complete"]
            manifest["status"] = "downloaded"
            manifest["source_complete"] = (
                args.max_files is None
                and (name != "stack-edu" or not args.stack_languages)
                and code_complete
            )
            manifest.pop("error", None)
        except BaseException as error:
            manifest["status"] = "interrupted"
            manifest["error"] = str(error)
            write_manifest(path, manifest)
            raise
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        write_manifest(path, manifest)
        results.append(
            {
                "source": name,
                "status": "downloaded",
                "manifest": str(path),
                "bytes": total,
            }
        )
    return results


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_gb(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples: --sources tinystories; --sources fineweb finemath "
            "--max-files 1; --all --max-files 1 --max-code-files 100. "
            "Install dependencies with uv sync --extra data."
        ),
    )
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--sources", nargs="+", choices=SOURCES)
    selected.add_argument(
        "--all", action="store_true", help="select every listed source"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-files", type=positive_int, help="Parquet shards per source"
    )
    parser.add_argument(
        "--max-code-files", type=positive_int, help="Stack-Edu code records"
    )
    parser.add_argument("--stack-languages", nargs="+", choices=STACK_LANGUAGES)
    parser.add_argument("--reserve-gb", type=positive_gb, default=50.0)
    parser.add_argument(
        "--max-gb", type=positive_gb, help="total output storage budget"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list sizes without downloads"
    )
    args = parser.parse_args()
    try:
        if args.dry_run:
            results = run(args)
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            with FileLock(args.output / ".download.lock", timeout=0):
                results = run(args)
        print(json.dumps(results, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
