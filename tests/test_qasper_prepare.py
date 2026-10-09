import io
import json
from pathlib import Path
import tarfile

import httpx
import pymupdf
import pytest

from evals.qasper_benchmark import prepare as prep


def write_pdf(path, title="A Study of Language", stamp="1801.00001v1"):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 50), title)
        page.insert_text((50, 80), stamp)
        page.insert_textbox((50, 110, 550, 750), ("Usable research text and experimental results. " * 30), fontsize=10)
        document.save(path)


def splits():
    return {"train": {"1801.00001": {"title": "A Study of Language", "qas": []},
                      "1801.00002": {"title": "Second Paper", "qas": []}},
            "validation": {"1801.00001v2": {"title": "A Study of Language", "qas": []}},
            "test": {"1801.00003": {"title": "Third Paper", "qas": []}}}


def test_candidates_deterministic_and_deduplicate_versions(tmp_path):
    first = prep.make_candidates(splits(), tmp_path, 20261005)
    reordered = {split: dict(reversed(list(papers.items()))) for split, papers in splits().items()}
    assert first == prep.make_candidates(reordered, tmp_path, 20261005)
    duplicate = next(candidate for candidate in first if candidate["split"] == "validation")
    assert duplicate["status"] == "excluded"
    assert duplicate["exclusion_reason"] == "duplicate_work_id"
    candidate = first[0]
    assert candidate["url"].endswith(candidate["work_id"] + "v1")
    assert Path(candidate["pdf_path"]).name == candidate["work_id"] + "v1.pdf"
    assert candidate["fulltext_alignment_status"] == "pending"


def test_official_sources_preserve_raw_questions_and_cache_hashes(tmp_path, monkeypatch):
    calls = []
    checksums = {}
    monkeypatch.setattr(prep, "ARCHIVE_SHA256", checksums)

    def download(url, destination, **options):
        calls.append(url)
        with tarfile.open(destination, "w:gz") as bundle:
            for split, filename in prep.ARCHIVES[destination.name].items():
                content = json.dumps(splits()[split]).encode()
                member = tarfile.TarInfo("nested/" + filename)
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
        checksums[destination.name] = prep.sha256(destination)
        return [{"status": "downloaded"}]

    monkeypatch.setattr(prep, "download", download)
    source = {}
    assert prep.load_sources(tmp_path, source) == splits()
    assert len(calls) == 2
    assert prep.load_sources(tmp_path, source) == splits()
    assert len(calls) == 2
    assert source["hf_revision"] == prep.HF_REVISION
    assert source["license"] == "CC-BY-4.0"
    for split, path in source["raw_files"].items():
        assert json.loads(Path(path).read_text()) == splits()[split]
    for record in source["archives"].values():
        assert record["sha256"] == prep.sha256(Path(record["path"]))


def test_removed_archive_changed_redownload_preserves_expected_checksum(tmp_path, monkeypatch):
    filename = next(iter(prep.ARCHIVES))
    archive = tmp_path / "cache" / filename
    archive.parent.mkdir()
    archive.write_bytes(b"original archive")
    expected = prep.sha256(archive)
    source = {"archives": {filename: {"url": prep.SOURCE_ROOT + filename,
                                      "attempts": [], "sha256": expected}}}
    monkeypatch.setattr(prep, "ARCHIVE_SHA256", {filename: expected})
    archive.unlink()
    raw = tmp_path / "raw" / "qasper-train-v0.3.json"
    raw.parent.mkdir()
    raw.write_bytes(b"original raw data")

    def download(url, destination, **options):
        destination.write_bytes(b"changed archive")
        return [{"status": "downloaded"}]

    monkeypatch.setattr(prep, "download", download)
    with pytest.raises(ValueError, match="Archive checksum changed"):
        prep.load_sources(tmp_path, source)
    assert source["archives"][filename]["sha256"] == expected
    assert raw.read_bytes() == b"original raw data"


@pytest.mark.parametrize("honor_range", [True, False])
def test_download_resumes_or_restarts_partial(tmp_path, honor_range):
    destination = tmp_path / "paper.pdf"
    destination.with_suffix(".pdf.part").write_bytes(b"abc")

    def response(request):
        assert request.headers["range"] == "bytes=3-"
        if honor_range:
            return httpx.Response(206, headers={"Content-Range": "bytes 3-5/6"}, content=b"def")
        return httpx.Response(200, content=b"abcdef")

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        attempts = prep.download("https://example.org/paper.pdf", destination, client=client)
    assert destination.read_bytes() == b"abcdef"
    assert attempts[-1]["status"] == "downloaded"


def test_download_bounded_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(prep.time, "sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client:
        with pytest.raises(prep.DownloadFailure) as caught:
            prep.download("https://example.org/paper.pdf", tmp_path / "paper.pdf", retries=2, client=client)
    assert len(caught.value.attempts) == 3


def test_product_parse_title_and_explicit_version(tmp_path):
    candidate = next(c for c in prep.make_candidates(splits(), tmp_path, 1) if c["paper_id"] == "1801.00001")
    Path(candidate["pdf_path"]).parent.mkdir()
    write_pdf(candidate["pdf_path"])
    prep.validate_pdf(candidate)
    assert candidate["status"] == "ready"
    assert candidate["parse_status"] == "parsed"
    assert candidate["version_evidence"]["qasper_fulltext_version"] == "unknown"
    assert candidate["fulltext_alignment_status"] == "pending"
    candidate["title"] = "An Unrelated Publication"
    with pytest.raises(ValueError, match="title_mismatch"):
        prep.validate_pdf(candidate)
    candidate["title"] = "A Study of Language"
    candidate["version"] = "v2"
    with pytest.raises(ValueError, match="explicit_version_not_verified"):
        prep.validate_pdf(candidate)


def test_network_wall_retains_all_candidates_and_resume(tmp_path, monkeypatch):
    data = {"train": {f"1801.{index:05}": {"title": "Paper"} for index in range(10)},
            "validation": {}, "test": {}}
    monkeypatch.setattr(prep, "load_sources", lambda *args, **kwargs: data)
    calls = []

    def fail(candidate, **options):
        calls.append(candidate["paper_id"])
        candidate.update(status="download_failed", exclusion_reason="blocked")
        candidate["attempts"].append({"status": "failed"})
        return candidate

    monkeypatch.setattr(prep, "acquire", fail)
    targets = {"train": 4, "validation": 0, "test": 0}
    manifest = prep.prepare(tmp_path, targets=targets, delay=0)
    assert manifest["status"] == "preparation_incomplete"
    assert manifest["stop_reason"].startswith("network_wall")
    assert len(calls) == len(set(calls)) == 3
    assert len(manifest["candidates"]) == 10
    assert manifest["summary"]["total_ready"] == 0
    initial_order = [candidate["paper_id"] for candidate in manifest["candidates"]]

    def succeed(candidate, **options):
        path = Path(candidate["pdf_path"])
        path.write_bytes(b"fixture")
        candidate.update(status="ready", sha256=prep.sha256(path))
        return candidate

    monkeypatch.setattr(prep, "acquire", succeed)
    manifest = prep.prepare(tmp_path, targets=targets, delay=0)
    assert manifest["status"] == "prepared"
    assert manifest["summary"]["total_ready"] == 4
    assert [candidate["paper_id"] for candidate in manifest["candidates"]] == initial_order
    monkeypatch.setattr(prep, "acquire", lambda *args, **kwargs: pytest.fail("Already prepared"))
    assert prep.prepare(tmp_path, targets=targets, delay=0)["status"] == "prepared"
    with pytest.raises(ValueError, match="configuration differs"):
        prep.prepare(tmp_path, targets=targets, seed=2)
