"""Associate original QASPER questions with prepared, indexed PDF evidence."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path
import random
import re
import sqlite3

from .align import align_annotations, build_document
from .identity import corpus_fingerprint, public_cases_fingerprint


SEED = 20261005
QUOTAS = {"dev": {"aligned": 32, "unanswerable": 8},
          "test": {"aligned": 64, "unanswerable": 16}}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_cases(manifest_path: Path, build_path: Path, database: Path,
                  output_dir: Path, *, seed: int = SEED,
                  diagnostic_dev: bool = False, resume: bool = False) -> dict:
    """Write public cases and separate scoring data, without querying retrieval."""
    manifest, build = _load(manifest_path), _load(build_path)
    sources = {split: _load(path) for split, path in manifest["source"]["raw_files"].items()}
    if set(sources) != {"train", "validation", "test"}:
        raise ValueError("Original train, validation and test sources are required")
    candidates = {(item["split"], item["paper_id"]): item for item in manifest["candidates"]}
    if len(candidates) != len(manifest["candidates"]):
        raise ValueError("Duplicate source candidate IDs")

    mappings, index_ids, paths, works = {}, set(), set(), set()
    for item in build["mappings"]:
        key = item["split"], item["paper_id"]
        source = candidates.get(key)
        if source is None or source["status"] != "ready":
            raise ValueError("Indexed mapping must have a ready source candidate")
        if item["paper_id"] not in sources[item["split"]]:
            raise ValueError("Indexed paper is absent from its original split")
        work = re.sub(r"v[1-9]\d*$", "", item["paper_id"])
        path = Path(item["pdf_path"]).resolve(strict=True)
        if path != Path(source["pdf_path"]).resolve(strict=True):
            raise ValueError("Prepared and indexed PDF paths differ")
        if (key in mappings or item["index_paper_id"] in index_ids
                or path in paths or work in works):
            raise ValueError("Duplicate paper or overlapping works across splits")
        if item["title"] != sources[item["split"]][item["paper_id"]]["title"]:
            raise ValueError("Mapping title differs from original QASPER title")
        mappings[key] = item
        index_ids.add(item["index_paper_id"])
        paths.add(path)
        works.add(work)

    documents = {}
    connection = sqlite3.connect(f"{Path(database).resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        indexed = {row["id"]: dict(row) for row in connection.execute(
            "SELECT id,file_path,file_hash FROM papers")}
        if set(indexed) != index_ids:
            raise ValueError("SQLite paper IDs differ from build mappings")
        for key, item in mappings.items():
            paper_id = item["index_paper_id"]
            if Path(indexed[paper_id]["file_path"]).resolve() != Path(item["pdf_path"]).resolve():
                raise ValueError("SQLite PDF path differs from build mapping")
            checksum = candidates[key].get("sha256")
            if not checksum or checksum != indexed[paper_id]["file_hash"]:
                raise ValueError("SQLite PDF revision differs from prepared source")
            blocks = [{"paper_id": paper_id, "passage_id": row["id"],
                       "page_number": row["page_number"], "block_number": row["block_number"],
                       "text": row["text"], "bbox": [row[k] for k in ("x0", "y0", "x1", "y1")]}
                      for row in connection.execute(
                          "SELECT * FROM passages WHERE paper_id=? ORDER BY page_number,block_number,id",
                          (paper_id,))]
            if not blocks:
                raise ValueError("Indexed paper contains no source passages")
            documents[str(paper_id)] = build_document(paper_id, blocks)
        corpus_identity = corpus_fingerprint(connection)
    finally:
        connection.close()

    gold_cases, decisions, eligible = [], [], []
    for original_split in ("train", "validation", "test"):
        for source_id, paper in sorted(sources[original_split].items()):
            key = original_split, source_id
            source, mapping = candidates.get(key), mappings.get(key)
            for question in paper["qas"]:
                case_id = f"q{len(gold_cases) + 1:06d}"
                split = "dev" if original_split == "validation" else original_split
                gold = {"case_id": case_id, "question_id": question["question_id"],
                        "source_paper_id": source_id, "original_split": original_split,
                        "split": split, "original_question": question,
                        "annotations": question["answers"], "label_origin": "QASPER human annotations"}
                decision = {"case_id": case_id, "source_paper_id": source_id,
                            "question_id": question["question_id"], "split": split,
                            "decision": "excluded"}
                if original_split == "train":
                    reason = "training_split"
                elif diagnostic_dev and original_split == "test":
                    reason = "test_not_evaluated"
                elif source is None or source["status"] != "ready":
                    reason = "source_not_ready"
                    decision["source_status"] = source["status"] if source else "not_candidate"
                    decision["source_exclusion_reason"] = source.get("exclusion_reason") if source else None
                elif mapping is None:
                    reason = "not_indexed"
                else:
                    document = documents[str(mapping["index_paper_id"])]
                    alignment = align_annotations(question["answers"], document)
                    gold.update(index_paper_id=mapping["index_paper_id"], alignment=alignment)
                    decision["alignment_status"] = alignment["status"]
                    if alignment["status"] == "excluded":
                        reason = alignment["reason"]
                    else:
                        reason = "quota_not_selected"
                        eligible.append((gold, decision, paper["title"]))
                decision["reason"] = reason
                gold_cases.append(gold)
                decisions.append(decision)

    random.Random(seed).shuffle(eligible)
    public_cases, selected_gold = [], []
    selected = {split: Counter() for split in QUOTAS}
    for gold, decision, title in eligible:
        split, kind = gold["split"], gold["alignment"]["status"]
        if diagnostic_dev:
            if len(public_cases) == 10:
                decision["reason"] = "diagnostic_limit"
                continue
        elif selected[split][kind] >= QUOTAS[split][kind]:
            continue
        question = gold["original_question"]["question"]
        case = {"case_id": gold["case_id"], "split": split, "question": question,
                "title": title, "display_question": f"{question}\nPaper: {title}",
                "scope": "full_library" if kind == "aligned" else "single_paper_auxiliary"}
        if kind == "unanswerable":
            case["paper_id"] = gold["index_paper_id"]
        public_cases.append(case)
        selected_gold.append(gold)
        selected[split][kind] += 1
        decision.update(decision="selected", reason="eligible")

    blockers = []
    if not 300 <= len(mappings) <= 500:
        blockers.append("corpus_size_outside_300_500")
    if build["status"] != "ready" or not build.get("validation", {}).get("valid"):
        blockers.append("build_not_ready")
    if any(selected[split][kind] != count for split, kinds in QUOTAS.items()
           for kind, count in kinds.items()):
        blockers.append("question_quotas_incomplete")
    status = "diagnostic" if diagnostic_dev else "incomplete" if blockers else "frozen"
    counters = {"corpus": len(mappings), "candidates": len(decisions),
                "eligible": len(eligible), "selected": len(public_cases),
                "selected_by_split": {s: {k: selected[s][k] for k in QUOTAS[s]} for s in QUOTAS},
                "decisions": dict(Counter(item["reason"] for item in decisions))}
    for key, item in mappings.items():
        with Path(item["pdf_path"]).open("rb") as stream:
            actual_checksum = file_digest(stream, "sha256").hexdigest()
        if actual_checksum != candidates[key]["sha256"]:
            raise ValueError("PDF bytes differ from prepared/indexed revision")
    public_identity = public_cases_fingerprint(public_cases)
    config = {"schema_version": 2, "seed": seed, "diagnostic_dev": diagnostic_dev,
              "quotas": QUOTAS, "question_rule": "question_then_paper_title"}
    fingerprint = sha256(json.dumps(
        {"config": config, "manifest": manifest, "build": build,
         "sources": sources, "documents": documents, "decisions": decisions,
         "gold_cases": gold_cases, "corpus_fingerprint": corpus_identity,
         "public_cases_fingerprint": public_identity}, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    metadata = {"status": status, "fingerprint": fingerprint, "counters": counters,
                "corpus_fingerprint": corpus_identity, "public_cases_fingerprint": public_identity,
                "freeze_blockers": blockers, "config": config}
    payloads = {
        "cases.json": {**metadata, "cases": public_cases},
        "gold.json": {**metadata, "cases": selected_gold, "candidates": gold_cases,
                      "documents": documents},
        "alignment.json": {**metadata, "decisions": decisions},
    }
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not resume:
            raise ValueError("Output directory already exists; use a new directory or --resume")
        for name, payload in payloads.items():
            path = output_dir / name
            if not path.is_file() or _load(path) != payload:
                raise ValueError("Resume inputs or saved outputs differ; choose a new output directory")
        return payloads["cases.json"]
    output_dir.mkdir(parents=True)
    for name, payload in payloads.items():
        with (output_dir / name).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return payloads["cases.json"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/qasper/manifest.json"))
    parser.add_argument("--build", type=Path, default=Path(".paperanchor/qasper-staging.build.json"))
    parser.add_argument("--database", type=Path, default=Path(".paperanchor/qasper-staging.sqlite3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--diagnostic-dev", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_cases(args.manifest, args.build, args.database, args.output,
                           seed=args.seed, diagnostic_dev=args.diagnostic_dev, resume=args.resume)
    print(json.dumps({k: result[k] for k in ("status", "counters", "freeze_blockers", "fingerprint")},
                     indent=2))
    return 0 if result["status"] == "frozen" or (
        result["status"] == "diagnostic" and result["cases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
