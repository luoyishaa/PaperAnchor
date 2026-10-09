"""Fixed-pool, retrieval-only QASPER experiments over an existing PDF index."""

import argparse
from contextlib import closing
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time

from paperanchor.library import PaperLibrary, PassageHit
from paperanchor.semantic import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider, SemanticIndex
from paperanchor.unit_index import UnitSemanticIndex, UnitStore
from paperanchor import library, representation, semantic, unit_index

from . import identity
from .identity import corpus_fingerprint, public_cases_fingerprint


METHODS = ("bm25", "dense", "hybrid", "rerank")


class CachedEmbeddingProvider(FastEmbedProvider):
    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.name, cache_dir=str(self.cache_dir) if self.cache_dir else None,
                local_files_only=True,
            )
        return self._model


class CachedReranker:
    def __init__(self, model_name, cache_dir=None):
        self.model_name, self.cache_dir = model_name, cache_dir
        self._model = None

    def score(self, query, texts):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                local_files_only=True,
            )
        return list(self._model.rerank(query, texts))


class ReferenceTokenizer:
    """An explicit cached tokenizer.json; never resolves a remote model name."""

    def __init__(self, path):
        from tokenizers import Tokenizer

        self.path = Path(path).resolve()
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


class FrozenUnitStore(UnitStore):
    def refresh(self, **kwargs):
        # Evaluation reads the prepared representation without rebuilding it.
        return 0


def serialize_hit(hit):
    chunk = asdict(hit)
    chunk["paper_path"] = str(hit.paper_path)
    return chunk


def assemble_context(ranked, tokenizer, budget=4096):
    """Append whole chunks, then the longest fitting sentence prefix and stop."""
    if budget < 1:
        raise ValueError("context budget must be positive")
    chunks, rendered = [], ""
    for hit in ranked:
        chunk = serialize_hit(hit)
        evidence_id = f"E{len(chunks) + 1}"
        header = f"[{evidence_id}] {hit.paper_title} (page {hit.page_number})\n"
        separator = "\n\n" if rendered else ""
        end = len(hit.text)
        candidate = rendered + separator + header + hit.text
        if tokenizer.count(candidate) > budget:
            end = 0
            for boundary in re.finditer(r"[.!?](?=\s|$)", hit.text):
                trial = boundary.end()
                if tokenizer.count(rendered + separator + header + hit.text[:trial]) <= budget:
                    end = trial
            if not end:
                break
            candidate = rendered + separator + header + hit.text[:end]
        chunk.update(
            evidence_id=evidence_id, text=hit.text[:end],
            retrieved_char_range=[0, end], original_char_count=len(hit.text),
            truncated=end < len(hit.text), reference_tokens=tokenizer.count(hit.text[:end]),
        )
        chunks.append(chunk)
        rendered = candidate
        if end < len(hit.text):
            break
    return {"chunks": chunks, "text": rendered, "reference_tokens": tokenizer.count(rendered),
            "budget": budget, "truncation_rule": "sentence_prefix_then_stop"}


def fixed_candidates(query, lexical, semantic, provider, *, depth=40, rrf_k=60,
                     reranker=None, paper_id=None, dense_error=None):
    """Call each channel once; rerank the entire identical RRF union."""
    if depth < 1 or rrf_k < 1:
        raise ValueError("candidate depth and RRF constant must be positive")
    scope = {} if paper_id is None else {"paper_id": paper_id}
    channels, failures, timings = {}, {}, {}
    for name in ("bm25", "dense"):
        started = time.perf_counter()
        try:
            if name == "bm25":
                channels[name] = tuple(lexical.search(query, depth, **scope))
            else:
                if dense_error:
                    raise RuntimeError(dense_error)
                if semantic.count(provider.name) == 0:
                    raise RuntimeError("No indexed vectors for requested embedding model")
                channels[name] = tuple(semantic.search(query, provider, limit=depth, **scope))
                if not channels[name] and paper_id is None:
                    del channels[name]
                    raise RuntimeError("Indexed vectors returned no candidates; check embedding dimensions")
        except Exception as exc:
            failures[name] = {"type": type(exc).__name__, "message": str(exc)}
        timings[name] = time.perf_counter() - started
    rankings = dict(channels)
    union = ()
    if not failures:
        started = time.perf_counter()
        by_id, scores = {}, {}
        for hits in channels.values():
            for rank, hit in enumerate(hits, 1):
                by_id[hit.passage_id] = hit
                scores[hit.passage_id] = scores.get(hit.passage_id, 0) + 1 / (rrf_k + rank)
        union = tuple(replace(by_id[key], score=scores[key]) for key in
                      sorted(scores, key=lambda key: (-scores[key], key)))
        rankings["hybrid"] = union
        timings["hybrid"] = sum(timings.values()) + time.perf_counter() - started
        if reranker is not None:
            started = time.perf_counter()
            try:
                values = reranker.score(query, [hit.text for hit in union]) if union else []
                if len(values) != len(union) or not all(math.isfinite(v) for v in values):
                    raise ValueError("Reranker returned invalid scores")
                rankings["rerank"] = tuple(replace(union[i], score=float(values[i])) for i in
                                           sorted(range(len(union)), key=lambda i: (-values[i], i)))
            except Exception as exc:
                failures["rerank"] = {"type": type(exc).__name__, "message": str(exc)}
            timings["rerank"] = timings["hybrid"] + time.perf_counter() - started
    else:
        failures["hybrid"] = {"type": "CandidateFailure", "message": "A candidate channel failed"}
        if reranker is not None:
            failures["rerank"] = failures["hybrid"].copy()
    return {"rankings": rankings, "channels": channels, "union": union,
            "failures": failures, "seconds": timings}


def completed_groups(records, cases, representations, split, methods):
    case_ids = [case.get("case_id", case.get("question_id")) for case in cases if case["split"] == split]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case IDs must be unique within the selected split")
    expected = {(case_id, representation) for case_id in case_ids for representation in representations}
    groups, seen = {}, set()
    for record in records:
        group = (record["case_id"], record["representation"])
        key = (*group, record["method"])
        if key in seen:
            raise ValueError("Checkpoint contains duplicate records")
        if (group not in expected or record["split"] != split or record["method"] not in methods
                or record["status"] not in ("ok", "failed", "context_failed")):
            raise ValueError("Checkpoint contains an unexpected record")
        seen.add(key)
        groups.setdefault(group, set()).add(record["method"])
    if any(value != set(methods) for value in groups.values()):
        raise ValueError("Checkpoint contains an incomplete candidate-pool group")
    return set(groups)


def run_cases(cases, backends, provider, tokenizer, *, split="dev", allow_test=False,
              depth=40, top_k=20, budget=4096, rrf_k=60, reranker=None,
              tokenizer_error=None, index_states=None, previous_records=(), checkpoint=None,
              methods=None):
    if split not in ("dev", "test") or (split == "test" and not allow_test):
        raise ValueError("test requires explicit allow_test")
    if top_k < 1 or top_k > depth:
        raise ValueError("top_k must be between 1 and candidate depth")
    records = list(previous_records)
    available = METHODS if reranker is not None else METHODS[:-1]
    methods = tuple(methods) if methods is not None else available
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(available):
        raise ValueError("Select unique available retrieval methods; rerank requires a reranker")
    completed = completed_groups(records, cases, backends, split, methods)
    for case in cases:
        if case["split"] != split:
            continue
        query = case.get("display_question") or f'{case["question"]}\nPaper: {case["title"]}'
        scope = case.get("scope", "full_library")
        if scope not in ("full_library", "single_paper_auxiliary"):
            raise ValueError(f"Unknown scope: {scope}")
        paper_id = case["paper_id"] if scope == "single_paper_auxiliary" else None
        for representation, (lexical, semantic) in backends.items():
            if (case.get("case_id", case.get("question_id")), representation) in completed:
                continue
            dense_error = None
            if index_states is not None:
                state = index_states.get(representation, {})
                if not state.get("vector_coverage_complete") or state.get("unprepared_papers"):
                    dense_error = (f"Incomplete {representation} embedding coverage: "
                                   f'{state.get("vectors", 0)}/{state.get("total", 0)}; '
                                   f'unprepared papers: {state.get("unprepared_papers", 0)}')
            result = fixed_candidates(query, lexical, semantic, provider, depth=depth,
                                      rrf_k=rrf_k, reranker=reranker, paper_id=paper_id,
                                      dense_error=dense_error)
            trace = {"channels": {name: [serialize_hit(h) for h in hits]
                                  for name, hits in result["channels"].items()},
                     "union": [serialize_hit(h) for h in result["union"]]}
            group_records = []
            for method in methods:
                record = {"case_id": case.get("case_id", case.get("question_id")),
                          "split": split, "scope": scope, "query": query,
                          "representation": representation, "method": method,
                          "candidate_depth": depth, "rrf_k": rrf_k, "top_k": top_k,
                          "candidates": trace, "seconds": result["seconds"].get(method),
                          "status": "failed" if method in result["failures"] else "ok"}
                if method in result["failures"]:
                    record["failure"] = result["failures"][method]
                else:
                    ranked = result["rankings"][method][:top_k]
                    record["ranked"] = [serialize_hit(h) for h in ranked]
                    try:
                        if tokenizer is None:
                            raise RuntimeError(tokenizer_error or "Reference tokenizer cache unavailable")
                        record["context"] = assemble_context(ranked, tokenizer, budget)
                    except Exception as exc:
                        record["context_failure"] = {"type": type(exc).__name__, "message": str(exc)}
                        record["status"] = "context_failed"
                group_records.append(record)
            records.extend(group_records)
            if checkpoint is not None:
                checkpoint(list(records))
    return records


def index_metadata(path, representation, model):
    table, vectors = (("passages", "passage_embeddings") if representation == "blocks"
                      else ("retrieval_units", "unit_embeddings"))
    with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as connection:
        total = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        foreign_key = "passage_id" if representation == "blocks" else "unit_id"
        counts = connection.execute(
            f"SELECT e.dimension,count(*) FROM {vectors} e JOIN {table} t "
            f"ON t.id=e.{foreign_key} WHERE e.model=? GROUP BY e.dimension", (model,)
        ).fetchall()
        metadata = {"total": total, "vectors": sum(row[1] for row in counts),
                    "dimensions": [row[0] for row in counts], "model": model}
        metadata["vector_coverage_complete"] = (
            total > 0 and metadata["vectors"] == total and len(counts) == 1
        )
        if representation == "units":
            metadata["unit_configurations"] = connection.execute(
                "SELECT DISTINCT representation_version,tokenizer_model,config_hash FROM unit_index_state"
            ).fetchall()
            metadata["unprepared_papers"] = connection.execute(
                "SELECT count(*) FROM papers p LEFT JOIN unit_index_state s ON s.paper_id=p.id "
                "WHERE s.paper_id IS NULL OR s.file_hash!=p.file_hash"
            ).fetchone()[0]
        return metadata


def database_fingerprint(path):
    """Hash a logical read snapshot, including committed WAL contents."""
    digest = sha256()
    with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        for statement in connection.iterdump():
            digest.update(statement.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def implementation_fingerprint():
    digest = sha256()
    for path in (Path(__file__), Path(identity.__file__), Path(library.__file__),
                 Path(semantic.__file__), Path(unit_index.__file__), Path(representation.__file__)):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_fingerprints(config, database, tokenizer_path, cases_text):
    configuration = {key: value for key, value in config.items() if key not in ("output", "resume")}
    configuration.update(protocol_version=1, rrf_k=60,
                         truncation_rule="sentence_prefix_then_stop",
                         reference_add_special_tokens=False,
                         implementation_fingerprint=implementation_fingerprint())
    fingerprints = {
        "config_fingerprint": sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
        "database_fingerprint": database_fingerprint(database),
        "tokenizer_fingerprint": (sha256(tokenizer_path.read_bytes()).hexdigest()
                                  if tokenizer_path.is_file() else None),
        "cases_content_fingerprint": sha256(cases_text.encode("utf-8")).hexdigest(),
    }
    fingerprints["run_fingerprint"] = sha256(
        json.dumps(fingerprints, sort_keys=True).encode()
    ).hexdigest()
    return fingerprints


def save_checkpoint(path, payload):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Resume reuses all stored attempts, including failures, only with matching input fingerprints.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-tokenizer", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rerank-model", default="")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--representations", nargs="+", choices=("blocks", "units"), default=["blocks", "units"])
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--budget", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.methods and (len(set(args.methods)) != len(args.methods)
                         or ("rerank" in args.methods and not args.rerank_model)):
        parser.error("Select unique methods; rerank requires --rerank-model")
    if args.split == "test" and not args.allow_test:
        parser.error("test requires --allow-test")
    if not args.database.is_file():
        parser.error("database must already exist")
    if args.output.exists() and not args.resume:
        parser.error("output must be a new file unless --resume is supplied")
    if args.resume and not args.output.is_file():
        parser.error("--resume requires an existing checkpoint")
    raw = args.cases.read_text(encoding="utf-8")
    cases = ([json.loads(line) for line in raw.splitlines() if line.strip()]
             if args.cases.suffix == ".jsonl" else json.loads(raw))
    cases_metadata = {key: value for key, value in cases.items() if key != "cases"} if isinstance(cases, dict) else {}
    if isinstance(cases, dict):
        cases = cases["cases"]
    if args.split == "test" and cases_metadata.get("status") != "frozen":
        parser.error("test requires frozen cases")
    actual_identities = {"corpus_fingerprint": corpus_fingerprint(args.database),
                         "public_cases_fingerprint": public_cases_fingerprint(cases)}
    for key, value in actual_identities.items():
        if (cases_metadata.get("status") == "frozen" or key in cases_metadata) and cases_metadata.get(key) != value:
            parser.error(f"Cases {key} does not match the supplied inputs")
    tokenizer, tokenizer_error = None, None
    try:
        tokenizer = ReferenceTokenizer(args.reference_tokenizer)
    except Exception as exc:
        tokenizer_error = f"{type(exc).__name__}: {exc}"
    provider = CachedEmbeddingProvider(args.embedding_model, args.cache_dir)
    reranker = CachedReranker(args.rerank_model, args.cache_dir) if args.rerank_model else None
    library = PaperLibrary(args.database)
    backends, indexes, failures = {}, {}, {}
    for representation in args.representations:
        try:
            if representation == "blocks":
                backends[representation] = (library, SemanticIndex(args.database))
            else:
                store = FrozenUnitStore(library)
                backends[representation] = (store, UnitSemanticIndex(store))
            indexes[representation] = index_metadata(args.database, representation, provider.name)
        except Exception as exc:
            failures[representation] = {"type": type(exc).__name__, "message": str(exc)}
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    fingerprints = run_fingerprints(config, args.database, args.reference_tokenizer, raw)
    prepared_corpus = corpus_fingerprint(args.database)
    if cases_metadata.get("status") == "frozen" and prepared_corpus != actual_identities["corpus_fingerprint"]:
        parser.error("Frozen corpus changed during index initialization")
    actual_identities["corpus_fingerprint"] = prepared_corpus
    methods = tuple(args.methods) if args.methods else (METHODS if reranker is not None else METHODS[:-1])
    expected_groups = sum(case["split"] == args.split for case in cases) * len(args.representations)
    expected_records = expected_groups * len(methods)
    records = []
    if args.resume:
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        required = {**fingerprints, **actual_identities, "cases_fingerprint": cases_metadata.get("fingerprint"),
                    "cases_status": cases_metadata.get("status"),
                    "expected_records": expected_records, "expected_groups": expected_groups}
        if any(previous.get(key) != value for key, value in required.items()):
            parser.error("Checkpoint fingerprints or expected counts do not match this run")
        records = previous["records"]
        failures = {**previous.get("failures", {}), **failures}
    try:
        completed_groups(records, cases, args.representations, args.split, methods)
    except ValueError as exc:
        parser.error(str(exc))
    payload = {"status": "incomplete", "records": records,
               "expected_records": expected_records, "expected_groups": expected_groups,
               "index_metadata": indexes, "failures": failures,
               "cases_metadata": cases_metadata,
               "cases_fingerprint": cases_metadata.get("fingerprint"),
               "cases_status": cases_metadata.get("status"),
               **fingerprints, **actual_identities, "resume_supported": True,
               "config": config,
               "reference_tokenizer_error": tokenizer_error,
               "reference_count_is_generator_native": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(current):
        payload["records"] = current
        payload["completed_groups"] = len(current) // len(methods)
        save_checkpoint(args.output, payload)

    checkpoint(records)
    records = run_cases(cases, backends, provider, tokenizer, split=args.split,
                        allow_test=args.allow_test, depth=args.depth, top_k=args.top_k,
                        budget=args.budget, reranker=reranker, tokenizer_error=tokenizer_error,
                        index_states=indexes, previous_records=records, checkpoint=checkpoint,
                        methods=methods)
    if database_fingerprint(args.database) != fingerprints["database_fingerprint"]:
        failures["database"] = {"type": "DatabaseChanged", "message": "Database changed during retrieval"}
    complete = (bool(records) and len(records) == expected_records and not failures
                and all(r["status"] == "ok" for r in records)
                and all(index["vector_coverage_complete"] and not index.get("unprepared_papers")
                        for index in indexes.values()))
    payload["status"] = "complete" if complete else "incomplete"
    checkpoint(records)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
