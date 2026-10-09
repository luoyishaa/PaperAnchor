"""Evaluate one installed retrieval profile on the frozen QASPER 120 cases."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import statistics
import time

from paperanchor.evidence_profile import EvidenceProfile
from paperanchor.evidence_policies import BGE, MINILM
from paperanchor.evidence_retrieval import EvidenceRetriever

from .identity import corpus_fingerprint, public_cases_fingerprint
from .retrieve import save_checkpoint
from .score import score_context


PROFILES = {"minilm": "evidence_minilm", "bge": "evidence_v2",
            "optimized": "evidence_optimized"}


class UncachedProfile(EvidenceProfile):
    """Keep the product models resident without query or score result reuse."""

    def _current_engine(self):
        engine = super()._current_engine()
        engine.cache_enabled = False
        return engine


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_identity(model):
    identity = {"name": EvidenceRetriever._identity(model)}
    native = getattr(getattr(model, "_model", None), "model", None)
    directory = getattr(native, "_model_dir", None)
    description = getattr(native, "model_description", None)
    if directory is not None and description is not None:
        path = Path(directory) / description.model_file
        with path.open("rb") as stream:
            identity.update(file=str(path.resolve()), sha256=file_digest(stream, "sha256").hexdigest())
    return identity


def validate_inputs(public, gold, database):
    if public.get("status") != "frozen" or gold.get("status") != "frozen":
        raise ValueError("Both cases and gold must be frozen")
    cases = sorted(public["cases"], key=lambda case: case["case_id"])
    identifiers = {case["case_id"] for case in cases}
    if len(cases) != 120 or len(identifiers) != 120:
        raise ValueError("Expected exactly 120 distinct cases")
    main = sum(case["scope"] == "full_library" for case in cases)
    auxiliary = sum(case["scope"] == "single_paper_auxiliary" for case in cases)
    if (main, auxiliary) != (96, 24):
        raise ValueError("Expected 96 main and 24 auxiliary cases")
    if len(gold["cases"]) != 120 or {case["case_id"] for case in gold["cases"]} != identifiers:
        raise ValueError("Cases and gold contain different question sets")
    for key in ("fingerprint", "corpus_fingerprint", "public_cases_fingerprint"):
        if not public.get(key) or public[key] != gold.get(key):
            raise ValueError(f"Cases and gold {key} differ")
    if public_cases_fingerprint(public["cases"]) != public["public_cases_fingerprint"]:
        raise ValueError("Public case content changed")
    if corpus_fingerprint(database) != public["corpus_fingerprint"]:
        raise ValueError("Indexed corpus differs from the frozen cases")
    references = {case["case_id"]: case for case in gold["cases"]}
    for case in cases:
        reference = references[case["case_id"]]
        expected = "aligned" if case["scope"] == "full_library" else "unanswerable"
        if reference["alignment"]["status"] != expected:
            raise ValueError("Case scope differs from its gold annotation")
        if case["scope"] == "single_paper_auxiliary" and case["paper_id"] != reference["index_paper_id"]:
            raise ValueError("Auxiliary paper scope differs from its gold annotation")
        if not isinstance(case["question"], str) or not isinstance(case["title"], str):
            raise ValueError("Question and title must be strings")
    return cases, references


def prepare_case(profile, case, profile_name):
    return profile.prepare(
        case["question"], title=case["title"], profile_name=profile_name,
        paper_ids=(case["paper_id"],) if case["scope"] == "single_paper_auxiliary" else None,
    )


def summarize(records):
    main = [row for row in records if row["scope"] == "full_library"]
    durations = sorted(row["seconds"] for row in records)
    complete = sum(row["metrics"]["complete_support"] for row in main)
    return {
        "status": "complete" if all(row["status"] == "ok" for row in records) else "incomplete",
        "questions": len(records), "main": len(main), "auxiliary": len(records) - len(main),
        "failures": sum(row["status"] != "ok" for row in records),
        "complete": complete, "ContextCompleteSupport": complete / len(main),
        "ContextEvidenceRecall": statistics.mean(row["metrics"]["evidence_recall"] for row in main),
        "mean_reference_tokens": statistics.mean(row["context"]["reference_tokens"] for row in records),
        "seconds": {"mean": statistics.mean(durations), "p50": statistics.median(durations),
                    "p95": durations[min(len(durations) - 1, int(len(durations) * .95))]},
        "timing": "Warm models; query and score result caches disabled; warmup excluded.",
        "generation": "not_evaluated",
    }


def run(database, cases_path, gold_path, output, *, profile_name="minilm"):
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}")
    database, output = Path(database).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory for each evaluation")
    public, gold = read(cases_path), read(gold_path)
    cases, references = validate_inputs(public, gold, database)
    output.mkdir(parents=True)
    profile = UncachedProfile(database)
    selected = PROFILES[profile_name]
    try:
        spec = profile._optimized_spec() if profile_name == "optimized" else None
        config = spec["config"] if spec else {
            **profile.config, "rerank_model": MINILM if profile_name == "minilm" else BGE,
        }
        source_root = Path(__file__).resolve().parents[2]
        source_files = [
            "evals/qasper_benchmark/run.py", "evals/qasper_benchmark/score.py",
            "evals/qasper_benchmark/align.py", "src/paperanchor/evidence_profile.py",
            "src/paperanchor/evidence_retrieval.py", "src/paperanchor/evidence_context.py",
            "src/paperanchor/evidence_policies.py",
        ]
        protocol = {
            "profile": selected, "config": config, "optimized_spec": spec,
            "database": str(database), "cases": str(Path(cases_path).resolve()),
            "gold": str(Path(gold_path).resolve()), "budget": 4096,
            "result_cache": False, "questions": 120, "main": 96, "auxiliary": 24,
            "cases_fingerprint": public["fingerprint"],
            "public_cases_fingerprint": public["public_cases_fingerprint"],
            "corpus_fingerprint": public["corpus_fingerprint"],
            "tokenizer_sha256": sha256(profile.tokenizer_path.read_bytes()).hexdigest(),
            "source_sha256": {path: sha256((source_root / path).read_bytes()).hexdigest()
                              for path in source_files},
        }
        save_checkpoint(output / "protocol.json", protocol)
        prepare_case(profile, cases[0], selected)
        selected_models = (BGE, MINILM) if spec else (config["rerank_model"],)
        protocol["runtime_models"] = {
            "embedding": model_identity(profile.provider),
            "rerankers": [model_identity(profile.models[name]) for name in selected_models],
        }
        save_checkpoint(output / "protocol.json", protocol)
        revision = profile._revision
        records = []
        for index, case in enumerate(cases, 1):
            started = time.perf_counter()
            row = {"case_id": case["case_id"], "scope": case["scope"], "split": case["split"],
                   "question": case["question"], "title": case["title"], "profile": selected}
            try:
                context = prepare_case(profile, case, selected)
                row["seconds"] = time.perf_counter() - started
                if context["budget"] != 4096 or context["reference_tokens"] > 4096:
                    raise ValueError("Product context exceeds the fixed 4096-token budget")
                row.update(status="ok", context=context)
            except Exception as error:
                row.update(status="failed", seconds=time.perf_counter() - started,
                           error=f"{type(error).__name__}: {error}",
                           context={"chunks": [], "text": "", "budget": 4096, "reference_tokens": 0})
            if profile._revision != revision:
                raise ValueError("Indexed content changed during evaluation; use a stable library")
            if case["scope"] == "full_library":
                reference = references[case["case_id"]]
                row["metrics"] = score_context(reference["alignment"]["support_groups"],
                                               row["context"]["chunks"],
                                               gold["documents"][str(reference["index_paper_id"])])
            # Product contexts can contain Paths; normalize only for JSON storage.
            row = json.loads(json.dumps(row, default=lambda value: str(value) if isinstance(value, Path) else value))
            save_checkpoint(output / f'{case["case_id"]}.json', row)
            records.append(row)
            print(f'{profile_name}: {index}/120 {case["case_id"]} {row["status"]}', flush=True)
        result = summarize(records)
        save_checkpoint(output / "records.json", {"protocol": protocol, "records": records})
        save_checkpoint(output / "summary.json", result)
        return result
    finally:
        profile.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="minilm")
    parser.add_argument("--database", type=Path, default=Path(".paperanchor/library.sqlite3"))
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.database, args.cases, args.gold, args.output, profile_name=args.profile)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
