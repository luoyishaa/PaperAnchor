"""Prepare an auditable, resumable PDF corpus from the official QASPER splits."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import tarfile
import time
import unicodedata

import httpx
import pymupdf

from paperanchor.pdf import parse_pdf


HF_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
SOURCE_ROOT = "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
ARCHIVES = {
    "qasper-train-dev-v0.3.tgz": {
        "train": "qasper-train-v0.3.json", "validation": "qasper-dev-v0.3.json",
    },
    "qasper-test-and-evaluator-v0.3.tgz": {"test": "qasper-test-v0.3.json"},
}
DEFAULT_TARGETS = {"train": 200, "validation": 80, "test": 120}
ARCHIVE_SHA256 = {
    "qasper-train-dev-v0.3.tgz": "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a",
    "qasper-test-and-evaluator-v0.3.tgz": "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72",
}
ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5})(v[1-9]\d*)?$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class DownloadFailure(Exception):
    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


def download(url: str, path: Path, *, retries: int = 1, timeout: float = 20,
             wall_timeout: float = 90, client=None) -> list[dict]:
    """Keep a partial transfer and resume only when the server honors its range."""
    attempts = []
    partial = path.with_suffix(path.suffix + ".part")
    owns_client = client is None
    if owns_client:
        client = httpx.Client(follow_redirects=True, timeout=timeout,
                             headers={"User-Agent": "PaperAnchor-QASPER/0.1", "Accept-Encoding": "identity"})
    try:
        for retry in range(retries + 1):
            attempt = {"started_at": now(), "url": url}
            started = time.monotonic()
            offset = partial.stat().st_size if partial.exists() else 0
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with client.stream("GET", url, headers=headers, timeout=timeout) as response:
                    attempt.update(http_status=response.status_code, final_url=str(response.url))
                    if response.status_code == 416:
                        # A completed partial cannot be trusted without a full response.
                        partial.unlink(missing_ok=True)
                    response.raise_for_status()
                    append = response.status_code == 206 and offset > 0
                    if response.status_code == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("content-range", ""))
                        if not match or int(match[1]) != offset:
                            raise ValueError("Unexpected Content-Range")
                    with partial.open("ab" if append else "wb") as output:
                        for chunk in response.iter_bytes(64 * 1024):
                            if time.monotonic() - started > wall_timeout:
                                raise TimeoutError("Download wall timeout exceeded")
                            output.write(chunk)
                    if response.status_code == 206 and partial.stat().st_size != int(match[3]):
                        raise ValueError("Incomplete ranged response")
                partial.replace(path)
                attempt.update(status="downloaded", bytes=path.stat().st_size)
                attempts.append(attempt)
                return attempts
            except (httpx.HTTPError, OSError, ValueError, TimeoutError) as error:
                attempt.update(status="failed", error=f"{type(error).__name__}: {error}")
                attempts.append(attempt)
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (400, 401, 403, 404):
                    break
                if retry < retries:
                    time.sleep(min(2 ** retry, 8))
        raise DownloadFailure(attempts[-1]["error"], attempts)
    finally:
        if owns_client:
            client.close()


def load_sources(data_dir: Path, source: dict, **download_options) -> dict:
    cache, raw = data_dir / "cache", data_dir / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    source.update(
        dataset="allenai/qasper", hf_revision=HF_REVISION,
        hf_script_url=f"https://huggingface.co/datasets/allenai/qasper/blob/{HF_REVISION}/qasper.py",
        license="CC-BY-4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        license_scope="QASPER annotations; source PDFs retain their respective licenses",
        citation="Dasigi et al. (2021), A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers",
        citation_url="https://aclanthology.org/2021.naacl-main.365/",
    )
    archives = source.setdefault("archives", {})
    raw_files = source.setdefault("raw_files", {})
    splits = {}
    for filename, members in ARCHIVES.items():
        archive = cache / filename
        record = archives.setdefault(filename, {"url": SOURCE_ROOT + filename, "attempts": []})
        expected_sha256 = record.get("sha256")
        if not archive.exists():
            try:
                record["attempts"].extend(download(record["url"], archive, **download_options))
            except DownloadFailure as error:
                record["attempts"].extend(error.attempts)
                raise
        actual_sha256 = sha256(archive)
        if actual_sha256 != ARCHIVE_SHA256[filename] or (expected_sha256 and actual_sha256 != expected_sha256):
            raise ValueError(f"Archive checksum changed: {archive}")
        record.update(sha256=actual_sha256, path=str(archive), status="cached")
        with tarfile.open(archive, "r:gz") as bundle:
            for split, member_name in members.items():
                matches = [member for member in bundle.getmembers()
                           if member.isfile() and Path(member.name).name == member_name]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one archive member {member_name}")
                with bundle.extractfile(matches[0]) as stream:
                    content = stream.read()
                papers = json.loads(content)
                if not isinstance(papers, dict):
                    raise ValueError(f"Expected paper map in {member_name}")
                destination = raw / member_name
                destination.write_bytes(content)
                raw_files[split] = str(destination)
                splits[split] = papers
    return splits


def make_candidates(splits: dict, data_dir: Path, seed: int) -> list[dict]:
    candidates = []
    seen = set()
    # Sorted input makes the order independent of JSON object insertion order.
    for split in DEFAULT_TARGETS:
        ids = sorted(splits[split])
        random.Random(f"{seed}:{split}").shuffle(ids)
        for paper_id in ids:
            paper = splits[split][paper_id]
            match = ARXIV_ID.fullmatch(paper_id)
            work_id = match[1] if match else paper_id
            version = (match[2] or "v1") if match else None
            reason = "unsupported_arxiv_id" if not match else "duplicate_work_id" if work_id in seen else None
            seen.add(work_id)
            versioned_id = work_id + version if match else None
            candidates.append({
                "paper_id": paper_id, "work_id": work_id, "split": split,
                "title": paper["title"], "url": f"https://arxiv.org/pdf/{versioned_id}" if match else None,
                "version": version, "version_policy": "explicit source ID version, otherwise v1",
                "pdf_path": str(data_dir / "pdfs" / f"{versioned_id}.pdf") if match else None,
                "status": "excluded" if reason else "pending", "exclusion_reason": reason,
                "sha256": None, "parse_version": None, "parse_status": "pending",
                "identity_status": "pending", "version_evidence": None,
                "fulltext_alignment_status": "pending", "attempts": [],
            })
    return candidates


def title_key(text: str) -> str:
    return "".join(character for character in unicodedata.normalize("NFKC", text).casefold()
                   if character.isalnum())


def validate_pdf(candidate: dict) -> None:
    path = Path(candidate["pdf_path"])
    candidate["sha256"] = sha256(path)
    import paperanchor.pdf as parser_module
    candidate["parse_version"] = {
        "implementation": "paperanchor.pdf.parse_pdf", "source_sha256": sha256(Path(parser_module.__file__)),
        "pymupdf": pymupdf.VersionBind,
    }
    parsed = parse_pdf(path)
    text = " ".join(passage.text for passage in parsed.passages)
    candidate.update(parse_status="parsed", page_count=parsed.page_count, text_characters=len(text))
    if len(text) < 500:
        raise ValueError("insufficient_usable_text")
    first_page = " ".join(passage.text for passage in parsed.passages if passage.page_number == 1)
    expected_title = title_key(candidate["title"])
    if not expected_title or (expected_title not in title_key(first_page)
                              and expected_title != title_key(parsed.title)):
        candidate["identity_status"] = "title_mismatch"
        raise ValueError("title_mismatch")
    candidate["identity_status"] = "title_verified"
    stamp = candidate["work_id"] + candidate["version"]
    if not re.search(r"(?<![\w.])" + re.escape(stamp) + r"(?!\d)", text):
        raise ValueError("explicit_version_not_verified_in_pdf")
    candidate["version_evidence"] = {
        "status": "verified", "method": "explicit_version_url_and_pdf_arxiv_stamp", "stamp": stamp,
        "qasper_fulltext_version": "unknown",
    }
    candidate.update(status="ready", exclusion_reason=None)


def acquire(candidate: dict, **options) -> dict:
    path = Path(candidate["pdf_path"])
    candidate["attempts"].append({"started_at": now(), "action": "prepare"})
    if not path.exists():
        try:
            candidate["attempts"].extend(download(candidate["url"], path, **options))
        except DownloadFailure as error:
            candidate["attempts"].extend(error.attempts)
            candidate.update(status="download_failed", exclusion_reason=str(error))
            return candidate
    try:
        validate_pdf(candidate)
    except Exception as error:
        candidate.update(status="excluded", exclusion_reason=f"{type(error).__name__}: {error}")
        if candidate["parse_status"] != "parsed":
            candidate["parse_status"] = "failed"
    return candidate


def prepare(data_dir: Path, *, targets: dict | None = None, seed: int = 20261005,
            concurrency: int = 1, retries: int = 1, timeout: float = 20,
            wall_timeout: float = 90, failure_limit: int = 3, delay: float = 3) -> dict:
    data_dir = data_dir.resolve()
    targets = dict(DEFAULT_TARGETS if targets is None else targets)
    if set(targets) != set(DEFAULT_TARGETS) or any(value < 0 for value in targets.values()):
        raise ValueError("Targets must contain nonnegative train, validation and test counts")
    if concurrency < 1 or failure_limit < 1 or retries < 0 or min(timeout, wall_timeout) <= 0 or delay < 0:
        raise ValueError("Invalid download limits")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "pdfs").mkdir(exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "schema_version": 1, "source": {}, "candidates": [], "created_at": now(),
    }
    config = {"seed": seed, "targets": targets, "hf_revision": HF_REVISION}
    if manifest.get("config", config) != config:
        raise ValueError("Existing manifest configuration differs; choose another data directory")
    manifest.update(config=config, status="preparing", updated_at=now(),
                    download_config={"concurrency": concurrency, "retries": retries, "timeout": timeout,
                                     "wall_timeout": wall_timeout, "failure_limit": failure_limit, "delay": delay})
    options = {"retries": retries, "timeout": timeout, "wall_timeout": wall_timeout}
    try:
        splits = load_sources(data_dir, manifest["source"], **options)
    except (DownloadFailure, ValueError, OSError, tarfile.TarError) as error:
        manifest.update(status="preparation_incomplete", stop_reason=f"source_unavailable: {error}")
        save_json(manifest_path, manifest)
        return manifest
    if not manifest["candidates"]:
        manifest["candidates"] = make_candidates(splits, data_dir, seed)
    counts = dict.fromkeys(DEFAULT_TARGETS, 0)
    for candidate in manifest["candidates"]:
        if candidate["status"] == "ready":
            path = Path(candidate["pdf_path"])
            if not path.exists() or sha256(path) != candidate["sha256"]:
                raise ValueError(f"Prepared PDF missing or checksum changed: {path}")
            counts[candidate["split"]] += 1
    failures = 0
    manifest.pop("stop_reason", None)
    save_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for split in DEFAULT_TARGETS:
            pending = [candidate for candidate in manifest["candidates"]
                       if candidate["split"] == split and candidate["status"] in ("pending", "download_failed")]
            cursor = 0
            while counts[split] < targets[split] and cursor < len(pending):
                size = min(concurrency, targets[split] - counts[split], failure_limit - failures)
                batch = pending[cursor:cursor + size]
                futures = [executor.submit(acquire, candidate, **options) for candidate in batch]
                for future in futures:
                    candidate = future.result()
                    if candidate["status"] == "ready":
                        counts[split] += 1
                    failures = failures + 1 if candidate["status"] == "download_failed" else 0
                    print(f"{split} {candidate['paper_id']} {candidate['status']} ({counts[split]}/{targets[split]})", flush=True)
                cursor += len(batch)
                manifest.update(summary={"ready": counts.copy(), "total_ready": sum(counts.values())}, updated_at=now())
                save_json(manifest_path, manifest)
                if failures >= failure_limit:
                    manifest["stop_reason"] = f"network_wall: {failures} consecutive distinct candidate download failures"
                    break
                if counts[split] < targets[split] and cursor < len(pending) and delay:
                    time.sleep(delay)
            if failures >= failure_limit:
                break
    complete = all(counts[split] == targets[split] for split in targets)
    manifest.update(status="prepared" if complete else "preparation_incomplete", updated_at=now(),
                    summary={"ready": counts, "total_ready": sum(counts.values()),
                             "excluded": sum(c["status"] == "excluded" for c in manifest["candidates"]),
                             "download_failed": sum(c["status"] == "download_failed" for c in manifest["candidates"])})
    if not complete and "stop_reason" not in manifest:
        manifest["stop_reason"] = "eligible_candidates_exhausted"
    save_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("prepare")
    command.add_argument("--data-dir", type=Path, default=Path("data/qasper"))
    command.add_argument("--seed", type=int, default=20261005)
    for split, count in DEFAULT_TARGETS.items():
        command.add_argument(f"--{split}", type=int, default=count)
    command.add_argument("--concurrency", type=int, default=1)
    command.add_argument("--retries", type=int, default=1)
    command.add_argument("--timeout", type=float, default=20)
    command.add_argument("--wall-timeout", type=float, default=90)
    command.add_argument("--failure-limit", type=int, default=3)
    command.add_argument("--delay", type=float, default=3)
    args = vars(parser.parse_args(argv))
    args.pop("command")
    args["targets"] = {split: args.pop(split) for split in DEFAULT_TARGETS}
    manifest = prepare(**args)
    print(json.dumps({key: manifest[key] for key in ("status", "summary", "stop_reason") if key in manifest}), flush=True)
    return 0 if manifest["status"] == "prepared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
