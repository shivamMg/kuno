"""Prepare packed next-token data; Azure upload never approves production training."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import closing
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pyarrow.parquet as pq
from dotenv import dotenv_values
from filelock import FileLock

from kuno.data.dataset import PreparedDataset, read_corpus
from kuno.data.prepare import (
    load_preparation_config,
    pack_records,
    validate_corpus,
    validate_provenance,
)
from kuno.model.tokenizer import SPECIAL_TOKENS, encode_ordinary, train_byte_level_bpe
from kuno.storage import (
    AzureBatchStore,
    inventory,
    object_hash,
    safe_path,
    sha256_file,
    write_local_json,
)
from training.data.download import GB, SOURCES, check_space, positive_gb, positive_int

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent
CATEGORIES = {
    "fineweb-edu": "web",
    "fineweb": "web",
    "stack-edu": "code",
    "finemath": "math",
    "infiwebmath": "math",
    "megamath-web": "math",
    "cosmopedia-v2": "synthetic",
    "tinystories": "synthetic",
}


def raw_inventory(raw: Path, sources: list[str]) -> tuple[dict, list[tuple]]:
    identities, files = {}, []
    for source in sources:
        root = safe_path(raw, source)
        manifest = json.loads((root / "download-manifest.json").read_bytes())
        if (
            manifest.get("purpose") != "raw-download-only"
            or manifest.get("status") != "downloaded"
            or manifest.get("repo_id") != SOURCES[source].repo_id
        ):
            raise ValueError(f"raw source is not a verified download: {source}")
        identities[source] = {
            key: manifest[key] for key in ("repo_id", "revision", "files")
        }
        for relative, expected in manifest["files"].items():
            path = safe_path(root, relative)
            if (
                path.stat().st_size != expected["size"]
                or sha256_file(path) != expected["sha256"]
            ):
                raise ValueError(f"raw download changed: {source}/{relative}")
            if path.suffix == ".parquet":
                heldout = path.name.startswith(("validation-", "test-", "dev-"))
                files.append((heldout, source, relative, path))
    if not files:
        raise ValueError("no raw Parquet files selected")
    return identities, sorted(files, key=lambda item: not item[0])


def rows(path: Path, source: str, raw: Path):
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2000):
        for row in batch.to_pylist():
            row_index = offset
            offset += 1
            text = row.get("text")
            if source == "stack-edu":
                if row.get("license_type") != "permissive":
                    yield row_index, None, row
                    continue
                blob_id = row.get("blob_id", "")
                if not re.fullmatch(r"[0-9a-f]{40}", blob_id):
                    raise ValueError("invalid Stack-Edu blob ID")
                code = safe_path(raw / source, f"code/{blob_id[:2]}/{blob_id}.gz")
                with gzip.open(code, "rb") as handle:
                    body = handle.read()
                if sha1(body).hexdigest() != blob_id:
                    raise ValueError("Stack-Edu code identity differs")
                text = body.decode(row.get("src_encoding") or "utf-8", errors="strict")
            if text is not None and not isinstance(text, str):
                raise ValueError(f"text is not a string: {source}/{path.name}")
            yield row_index, text, row


def stage_documents(
    database: sqlite3.Connection,
    files: list[tuple],
    raw: Path,
    fit_bytes: int,
    validation_fraction: float,
    tokenizer_sample: Path | None = None,
) -> dict:
    database.executescript(
        "CREATE TABLE groups (id TEXT PRIMARY KEY, split TEXT);"
        "CREATE TABLE documents (hash TEXT PRIMARY KEY, group_id TEXT, "
        "source_id TEXT, category TEXT, split TEXT, text TEXT, attribution TEXT);"
        "CREATE TABLE fitting_documents (hash TEXT PRIMARY KEY);"
    )
    if tokenizer_sample:
        with tokenizer_sample.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                database.execute(
                    "INSERT OR IGNORE INTO fitting_documents VALUES (?)",
                    (record["document_hash"],),
                )
                database.execute(
                    "INSERT OR IGNORE INTO groups VALUES (?, 'tokenizer_fit')",
                    (record["group_id"],),
                )
    recipe = load_preparation_config(
        ROOT / "training" / "configs" / "preparation.yaml"
    ).source_percentages
    weights = {source: recipe.get(source, 1) for _, source, _, _ in files}
    budgets = {
        source: fit_bytes * weight / sum(weights.values())
        for source, weight in weights.items()
    }
    totals = Counter()
    sample_bytes = Counter()
    for heldout, source, relative, path in files:
        print(f"Reading {source}/{relative}", flush=True)
        for row_index, text, row in rows(path, source, raw):
            totals["raw_records"] += 1
            if text is None or not text.strip():
                totals["excluded_empty_or_unlicensed"] += 1
                continue
            body = text.encode("utf-8")
            document_hash = sha256(body).hexdigest()
            if database.execute(
                "SELECT 1 FROM documents WHERE hash=?", (document_hash,)
            ).fetchone():
                totals["exact_duplicates_removed"] += 1
                continue
            group = row.get("repo_name") or row.get("url") or document_hash
            group_id = sha256(str(group).encode("utf-8")).hexdigest()
            saved = database.execute(
                "SELECT split FROM groups WHERE id=?", (group_id,)
            ).fetchone()
            in_sample = database.execute(
                "SELECT 1 FROM fitting_documents WHERE hash=?", (document_hash,)
            ).fetchone()
            if heldout and (in_sample or (saved and saved[0] == "tokenizer_fit")):
                raise ValueError("frozen tokenizer overlaps native held-out data")
            if in_sample:
                split = "tokenizer_fit"
            elif saved:
                split = saved[0]
            elif heldout or (
                source != "tinystories"
                and int(group_id[:16], 16) / 2**64 < validation_fraction
            ):
                split = "development"
            elif not tokenizer_sample and sample_bytes[source] < budgets[source]:
                split = "tokenizer_fit"
            else:
                split = "train"
            if split == "tokenizer_fit":
                sample_bytes[source] += len(body)
            attribution = {
                "document_hash": document_hash,
                "source_id": source,
                "raw_file": relative,
                "row": row_index,
                **{
                    key: row[key]
                    for key in (
                        "url",
                        "repo_name",
                        "path",
                        "blob_id",
                        "detected_licenses",
                        "license_type",
                    )
                    if key in row
                },
            }
            database.execute(
                "INSERT OR IGNORE INTO groups VALUES (?, ?)", (group_id, split)
            )
            database.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    document_hash,
                    group_id,
                    source,
                    CATEGORIES[source],
                    split,
                    text,
                    json.dumps(attribution),
                ),
            )
            totals[split] += 1
            if totals["raw_records"] % 10000 == 0:
                database.commit()
        database.commit()
    if (not tokenizer_sample and not totals["tokenizer_fit"]) or not totals["train"]:
        raise ValueError("need separate tokenizer-fitting and model-training documents")
    return dict(totals)


def fit_tokenizer(database, directory: Path, vocab_size: int, sequence_length: int):
    directory.mkdir()
    database_path = database.execute("PRAGMA database_list").fetchone()[2]
    if not database_path:
        raise ValueError("tokenizer fitting requires a file-backed document index")

    def sample():
        position = 0
        with (directory / "sample.jsonl").open("w", encoding="utf-8") as handle:
            while True:
                # Tokenizers can request successive batches on different threads.
                with closing(sqlite3.connect(database_path)) as reader:
                    batch = reader.execute(
                        "SELECT rowid, hash, group_id, source_id, text FROM documents "
                        "WHERE split='tokenizer_fit' AND rowid>? "
                        "ORDER BY rowid LIMIT 1000",
                        (position,),
                    ).fetchall()
                if not batch:
                    return
                for row_id, digest, group, source, text in batch:
                    position = row_id
                    handle.write(
                        json.dumps(
                            {
                                "document_hash": digest,
                                "group_id": group,
                                "source_id": source,
                            }
                        )
                        + "\n"
                    )
                    yield text

    documents = sample()
    try:
        return train_byte_level_bpe(
            documents,
            output_dir=directory,
            min_frequency=1,
            vocab_size=vocab_size,
            model_max_length=sequence_length,
        )
    finally:
        documents.close()


def training_records(database):
    for count, (digest, group, source, category, split, text) in enumerate(
        database.execute(
            "SELECT hash, group_id, source_id, category, split, text FROM documents "
            "WHERE split!='tokenizer_fit' ORDER BY rowid"
        ),
        start=1,
    ):
        yield {
            "document_hash": digest,
            "group_id": group,
            "source_id": source,
            "category": category,
            "split": split,
            "text": text,
        }
        if count % 10000 == 0:
            print(f"Packed {count:,} documents", flush=True)


def validate_prepared(root: Path, *, scratch: Path | None = None) -> dict:
    corpus = read_corpus(root, sha256_file(root / "corpus.json"))
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    if (
        len(tokenizer) != corpus["vocab_size"]
        or tokenizer.eos_token_id != corpus["eos_token_id"]
        or object_hash(inventory(root / "tokenizer")) != corpus["tokenizer_sha256"]
    ):
        raise ValueError("tokenizer/corpus identities differ")
    for text in ("  code\n\t12345", "\u00e9 e\u0301", " ".join(SPECIAL_TOKENS)):
        encoded = encode_ordinary(tokenizer, text)
        if (
            set(encoded).intersection(tokenizer.all_special_ids)
            or tokenizer.decode(encoded, clean_up_tokenization_spaces=False) != text
        ):
            raise ValueError("ordinary text/control serialization differs")
    if corpus["vocab_size"] == 65536 and corpus["sequence_length"] == 8192:
        return validate_corpus(root, scratch=scratch)
    validate_provenance(root, corpus, scratch=scratch)
    emitted = 0
    streams = Counter()
    for shard in corpus["shards"]:
        path = safe_path(root, shard["path"])
        if path.stat().st_size != shard["size"] or sha256_file(path) != shard["sha256"]:
            raise ValueError("prepared token shard changed")
        tokens = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if tokens.dtype != np.dtype("uint32") or tokens.shape != (
                shard["sequences"],
                corpus["sequence_length"],
            ):
                raise ValueError("prepared shard shape/dtype differs")
            if np.any(tokens >= len(tokenizer)) or np.any(
                np.isin(
                    tokens,
                    [
                        value
                        for value in tokenizer.all_special_ids
                        if value != tokenizer.eos_token_id
                    ],
                )
            ):
                raise ValueError("invalid/reserved token in ordinary corpus")
            key = (shard["split"], shard["category"], shard["source_id"])
            if streams[key] != shard["stream_start"]:
                raise ValueError("prepared stream offset differs")
            streams[key] += tokens.size
            emitted += tokens.size
        finally:
            tokens._mmap.close()
    for stream in corpus["streams"]:
        key = (stream["split"], stream["category"], stream["source_id"])
        if (
            not 0 <= stream["discarded_tail_tokens"] < corpus["sequence_length"]
            or streams[key] + stream["discarded_tail_tokens"]
            != stream["serialized_tokens"]
        ):
            raise ValueError("token tail accounting differs")
    return {
        "corpus_sha256": sha256_file(root / "corpus.json"),
        "emitted_tokens": emitted,
        "shards": len(corpus["shards"]),
    }


def verify_prepared_files(root: Path) -> dict:
    marker = json.loads((root / "PREPARED.json").read_bytes())
    actual = inventory(root)
    actual.pop("PREPARED.json")
    if actual != marker["files"]:
        raise ValueError("prepared file inventory changed")
    return marker


def prepare(args) -> Path:
    sources = args.sources or [
        name
        for name in SOURCES
        if (args.raw / name / "download-manifest.json").is_file()
    ]
    if not sources:
        raise ValueError("no raw corpora downloaded")
    sources = list(dict.fromkeys(sources))
    name = args.name or "_".join(sources)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("invalid prepared dataset name")
    identities, files = raw_inventory(args.raw.resolve(), sources)
    identity = {
        "sources": identities,
        "vocab_size": args.vocab_size,
        "sequence_length": args.sequence_length,
        "shard_sequences": args.shard_sequences,
        "tokenizer_fit_bytes": int(args.tokenizer_fit_mb * 1_000_000),
        "validation_fraction": args.validation_fraction,
        "frozen_tokenizer": inventory(args.tokenizer) if args.tokenizer else None,
    }
    destination = safe_path(args.output.resolve(), name)
    if destination.exists():
        marker = verify_prepared_files(destination)
        if marker["identity"] != identity:
            raise ValueError("preparation request changed; use a new --name")
        return destination
    check_space(args.output.resolve(), 10 * GB, 50 * GB, None)
    work_root = args.output / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=work_root) as temporary:
        work = Path(temporary)
        with closing(sqlite3.connect(work / "documents.sqlite3")) as database:
            print("Staging unique documents and disjoint splits", flush=True)
            statistics = stage_documents(
                database,
                files,
                args.raw,
                identity["tokenizer_fit_bytes"],
                args.validation_fraction,
                args.tokenizer / "sample.jsonl" if args.tokenizer else None,
            )
            print(f"Fitting {args.vocab_size}-token byte-level BPE", flush=True)
            tokenizer_dir = work / "tokenizer"
            if args.tokenizer:
                from transformers import PreTrainedTokenizerFast

                shutil.copytree(args.tokenizer, tokenizer_dir)
                tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    tokenizer_dir, local_files_only=True
                )
                if len(tokenizer) != args.vocab_size:
                    raise ValueError("--vocab-size differs from frozen tokenizer")
            else:
                tokenizer = fit_tokenizer(
                    database, tokenizer_dir, args.vocab_size, args.sequence_length
                )
            print("Packing next-token sequences", flush=True)
            release = work / "release"
            corpus = pack_records(
                training_records(database),
                tokenizer,
                release,
                sequence_length=args.sequence_length,
                shard_sequences=args.shard_sequences,
            )
            shutil.move(str(tokenizer_dir), str(release / "tokenizer"))
            for source in sources:
                card = safe_path(release, f"source_cards/{source}.md")
                card.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.raw / source / "README.md", card)
            corpus["tokenizer_sha256"] = object_hash(inventory(release / "tokenizer"))
            write_local_json(release / "corpus.json", corpus)
            with (release / "attribution.jsonl").open("w", encoding="utf-8") as handle:
                for (record,) in database.execute(
                    "SELECT attribution FROM documents ORDER BY rowid"
                ):
                    handle.write(record + "\n")
            write_local_json(
                release / "preparation.json",
                {
                    "identity": identity,
                    "statistics": statistics,
                    "production_compatible": args.vocab_size == 65536
                    and args.sequence_length == 8192,
                    "production_approved": False,
                },
            )
            print("Validating token shards and split boundaries", flush=True)
            validation = validate_prepared(release, scratch=work)
            dataset = PreparedDataset(
                release, validation["corpus_sha256"], work / "loader-check"
            )
            try:
                dataset[0]
            finally:
                dataset.close()
            write_local_json(
                release / "PREPARED.json",
                {
                    "schema_version": 1,
                    "purpose": "prepared-next-token-data",
                    "identity": identity,
                    "validation": validation,
                    "production_approved": False,
                    "files": inventory(release),
                },
            )
            release.rename(destination)
    return destination


def find_env(explicit: Path | None) -> Path | None:
    if explicit:
        if not explicit.is_file():
            raise ValueError("--env file does not exist")
        return explicit
    return next(
        (
            path
            for path in (
                DATA.parent / ".env",
                DATA / ".env",
                ROOT / ".env",
            )
            if path.is_file()
        ),
        None,
    )


def blob_store(args, prefix: str) -> tuple[AzureBatchStore, str, str]:
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobServiceClient

    env = find_env(args.env)
    settings = dotenv_values(env) if env else {}
    resource = settings.get("STORAGE_ACCOUNT_RESOURCE_ID") or ""
    account = (
        args.storage_account
        or settings.get("AZURE_STORAGE_ACCOUNT")
        or resource.rsplit("/", 1)[-1]
    )
    container = (
        args.container
        or settings.get("AZURE_STORAGE_CONTAINER")
        or settings.get("STORAGE_CONTAINER")
        or "training"
    )
    if not account or not re.fullmatch(r"[a-z0-9]{3,24}", account):
        raise ValueError("set AZURE_STORAGE_ACCOUNT or --storage-account")
    connection = settings.get("AZURE_STORAGE_CONNECTION_STRING")
    options: dict[str, Any] = {
        "retry_total": 0,
        "connection_timeout": 10,
        "read_timeout": 60,
        "max_block_size": 8 * 1024 * 1024,
        "max_single_put_size": 8 * 1024 * 1024,
    }
    if connection:
        fields = dict(
            item.split("=", 1) for item in connection.split(";") if "=" in item
        )
        if fields.get("AccountName") != account:
            raise ValueError("connection string belongs to another storage account")
        service = BlobServiceClient.from_connection_string(connection, **options)
    else:
        service = BlobServiceClient(
            f"https://{account}.blob.core.windows.net",
            credential=AzureCliCredential(process_timeout=15),
            **options,
        )
    client = service.get_container_client(container)
    client.get_container_properties(timeout=30)
    return AzureBatchStore(client, prefix), account, container


def upload_prepared(root: Path, args) -> Path:
    marker = verify_prepared_files(root)
    digest = object_hash(marker["files"])
    env = find_env(args.env)
    settings = dotenv_values(env) if env else {}
    base = (
        args.blob_prefix
        or settings.get("AZURE_STORAGE_BLOB_PREFIX")
        or "data/corpus_prepared"
    )
    prefix = f"{base.rstrip('/')}/{root.name}/{digest[:16]}"
    store, account, container = blob_store(args, prefix)
    files = {
        **marker["files"],
        "PREPARED.json": {
            "size": (root / "PREPARED.json").stat().st_size,
            "sha256": sha256_file(root / "PREPARED.json"),
        },
    }
    for relative, expected in files.items():
        print(f"Uploading and verifying {relative}", flush=True)
        store.upload_file(safe_path(root, relative), relative, expected)
    verify_prepared_files(root)
    receipt = args.output / "upload-receipts" / f"{root.name}-{digest[:16]}.json"
    write_local_json(
        receipt,
        {
            "account": account,
            "container": container,
            "prefix": prefix,
            "verified": True,
            "production_approved": False,
            "files": files,
        },
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Default 16,384-token vocabulary is for pilot models. "
            "Use --vocab-size 65536 with sufficiently diverse tokenizer data "
            "for Kuno's production contract. Install with uv sync --extra data."
        ),
    )
    parser.add_argument("--sources", nargs="+", choices=SOURCES)
    parser.add_argument("--raw", type=Path, default=DATA / "corpus_raw")
    parser.add_argument("--output", type=Path, default=DATA / "corpus_prepared")
    parser.add_argument("--name")
    parser.add_argument("--vocab-size", type=positive_int, default=16384)
    parser.add_argument(
        "--tokenizer", type=Path, help="reuse a frozen tokenizer directory"
    )
    parser.add_argument("--sequence-length", type=positive_int, default=8192)
    parser.add_argument("--shard-sequences", type=positive_int, default=256)
    parser.add_argument("--tokenizer-fit-mb", type=positive_gb, default=64.0)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--env", type=Path)
    parser.add_argument("--storage-account")
    parser.add_argument("--container")
    parser.add_argument("--blob-prefix")
    args = parser.parse_args()
    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be between zero and one")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        with FileLock(args.output / ".prepare.lock", timeout=0):
            root = prepare(args)
            result: dict[str, Any] = {"prepared": str(root)}
            if args.upload:
                result["upload_receipt"] = str(upload_prepared(root, args))
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; raw data retained.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Preparation/upload failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
