"""Question-only LitQA2 evaluation using Europe PMC full text and product retrieval.

This adapter tests text retrieval/QA, not PDF extraction or page coordinates.
Gold sources and key passages are never passed to the retriever or answer model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import time
import unicodedata
import xml.etree.ElementTree as ET

import httpx

DATASET = "futurehouse/lab-bench"
ROOT = Path("data/litqa2")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def get(url, **kwargs):
    response = httpx.get(url, timeout=httpx.Timeout(35, connect=10), follow_redirects=True, **kwargs)
    response.raise_for_status()
    return response


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "questions.json"
    if not path.exists():
        info = get(f"https://huggingface.co/api/datasets/{DATASET}").json()
        rows = []
        offset = 0
        while True:
            data = get("https://datasets-server.huggingface.co/rows", params={
                "dataset": DATASET, "config": "LitQA2", "split": "train",
                "offset": offset, "length": 100}).json()
            rows.extend(item["row"] for item in data["rows"])
            offset += len(data["rows"])
            if offset >= data["num_rows_total"]:
                break
            if not data["rows"]:
                raise ValueError("Dataset pagination returned an empty page")
        after = get(f"https://huggingface.co/api/datasets/{DATASET}").json()
        if info["sha"] != after["sha"]:
            raise ValueError("Dataset changed during download")
        save(path, {"dataset": DATASET, "revision": info["sha"], "rows": rows})
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = sorted({url for row in data["rows"] for url in row["sources"]})
    old_path = ROOT / "manifest.json"
    manifest = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}

    def fetch(source):
        previous = manifest.get(source)
        if previous and previous.get("status") == "ready":
            return source, previous
        try:
            doi = re.sub(r"^https?://(?:dx\.)?doi.org/", "", source)
            result = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={
                "query": f'DOI:"{doi}"', "format": "json", "resultType": "core"}).json()
            matches = [p for p in result["resultList"]["result"]
                       if p.get("doi", "").lower() == doi.lower() and p.get("pmcid")]
            if not matches:
                return source, {"status": "unavailable", "reason": "No matching PMC full text", "doi": doi}
            paper = matches[0]
            url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{paper["pmcid"]}/fullTextXML'
            raw = get(url).content
            article = ET.fromstring(raw)
            if article.find("body") is None:
                raise ValueError("Downloaded XML has no article body")
            out = ROOT / "xml" / f'{paper["pmcid"]}.xml'
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(raw)
            return source, {"status": "ready", "doi": doi, "pmcid": paper["pmcid"],
                            "title": paper["title"], "path": str(out.resolve()),
                            "url": url, "sha256": sha256(raw).hexdigest()}
        except Exception as error:
            return source, {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, source) for source in sources]
        for i, future in enumerate(as_completed(futures), 1):
            source, record = future.result()
            manifest[source] = record
            save(old_path, manifest)
            print(f"Sources {i}/{len(sources)} ready={sum(v['status']=='ready' for v in manifest.values())}", flush=True)
    print(f"Questions: {len(data['rows'])}; unique sources: {len(sources)}", flush=True)


def article_blocks(raw):
    """Keep complete paragraphs, tables and captions; synthetic sections are not PDF pages."""
    root = ET.fromstring(raw)
    title = " ".join(root.findtext("./front/article-meta/title-group/article-title", "").split())
    title_node = root.find("./front/article-meta/title-group/article-title")
    if title_node is not None:
        title = " ".join("".join(title_node.itertext()).split())
    blocks = []
    section = 1

    def walk(node):
        nonlocal section
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "sec":
            section += 1
        if tag in {"title", "p", "table-wrap", "fig", "disp-formula", "list"}:
            text = " ".join("".join(node.itertext()).split())
            if text:
                blocks.append((section, text))
            return
        for child in node:
            walk(child)

    for node in root.findall("./front/article-meta/abstract") + root.findall("body"):
        walk(node)
    if not blocks:
        raise ValueError("No full-text blocks")
    return title, blocks


def build(database, manifest_path=ROOT / "manifest.json"):
    from paperanchor.library import PaperLibrary
    from paperanchor.semantic import FastEmbedProvider
    from paperanchor.unit_index import UnitStore, UnitSemanticIndex

    if database.name == "library.sqlite3":
        raise ValueError("Use an isolated evaluation database, not library.sqlite3")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    library = PaperLibrary(database)
    mapping = {}
    for source, paper in sorted(manifest.items()):
        if paper["status"] != "ready":
            continue
        raw = Path(paper["path"]).read_bytes()
        if sha256(raw).hexdigest() != paper["sha256"]:
            raise ValueError("Source file changed")
        title, blocks = article_blocks(raw)
        with sqlite3.connect(database) as connection:
            row = connection.execute("SELECT id,file_hash FROM papers WHERE file_path=?", (paper["path"],)).fetchone()
            if row:
                if row[1] != paper["sha256"]:
                    raise ValueError("Existing indexed source changed; use a fresh database")
                paper_id = row[0]
            else:
                paper_id = connection.execute(
                    "INSERT INTO papers(file_path,file_hash,title,page_count,indexed_at) VALUES(?,?,?,?,?)",
                    (paper["path"], paper["sha256"], title, max(p for p, _ in blocks),
                     datetime.now(timezone.utc).isoformat())).lastrowid
                for i, (section, text) in enumerate(blocks):
                    pid = connection.execute(
                        "INSERT INTO passages(paper_id,page_number,block_number,text,x0,y0,x1,y1) VALUES(?,?,?,?,0,0,0,0)",
                        (paper_id, section, i, text)).lastrowid
                    connection.execute("INSERT INTO passage_search(rowid,text) VALUES(?,?)", (pid, text))
            mapping[source] = paper_id
    save(database.with_suffix(".sources.json"), mapping)
    provider = FastEmbedProvider(cache_dir=Path(".paperanchor/models"), local_files_only=True)
    units = UnitStore(library)
    print("Creating reading units", flush=True)
    units.refresh(token_count=provider.count_untruncated_tokens, tokenizer_model=provider.name)
    with sqlite3.connect(database) as connection:
        unit_count = connection.execute("SELECT COUNT(*) FROM retrieval_units").fetchone()[0]
    print(f"Embedding {unit_count} units", flush=True)
    index = UnitSemanticIndex(units)
    index.build(provider, progress=lambda n, total: print(f"Vectors {n}/{total}", flush=True)
                if n % 512 == 0 or n == total else None)
    print(f"Indexed sources: {len(set(mapping.values()))}", flush=True)


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", "", text)


def score_context(context, gold_ids, key_passage):
    by_paper = {}
    for chunk in context["chunks"]:
        by_paper.setdefault(chunk["paper_id"], []).append(chunk["text"])
    retrieved = set(by_paper)
    key = normalize(key_passage or "")
    # Strict phrase presence is diagnostic, not semantic answer accuracy.
    exact = bool(key) and any(key in normalize(" ".join(by_paper.get(pid, []))) for pid in gold_ids)
    # Order-independent character coverage allows evidence split across emitted slices.
    intervals = []
    if key:
        from difflib import SequenceMatcher
        for pid in gold_ids:
            for part in by_paper.get(pid, []):
                for match in SequenceMatcher(None, key, normalize(part), autojunk=False).get_matching_blocks():
                    if match.size >= 20:
                        intervals.append((match.a, match.a + match.size))
    covered = set()
    for start, end in intervals:
        covered.update(range(start, end))
    return {"source_hit": bool(retrieved & gold_ids),
            "all_sources": bool(gold_ids) and gold_ids <= retrieved,
            "source_recall": len(retrieved & gold_ids) / len(gold_ids) if gold_ids else None,
            "key_passage_exact": exact if key else None,
            "key_passage_character_coverage": len(covered) / len(key) if key else None}


def run(database, output, answer=False, answer_limit=30, source_policy="all"):
    from paperanchor.evidence_profile import EvidenceProfile
    from paperanchor.model_config import load_model_config, generation_config, get_setting
    from paperanchor.rag import answer_from_context
    from paperanchor.llm import generate_with_config
    import random

    data = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    mapping = json.loads(database.with_suffix(".sources.json").read_text(encoding="utf-8"))
    rows = data["rows"]
    output.mkdir(parents=True, exist_ok=True)
    eligible = [row for row in rows if (any if source_policy == "any" else all)(source in mapping for source in row["sources"])]
    excluded = [{"id": row["id"], "missing_sources": [s for s in row["sources"] if s not in mapping]}
                for row in rows if row not in eligible]
    answer_ids = {row["id"] for row in sorted(eligible, key=lambda r: sha256(("20261009:" + r["id"]).encode()).hexdigest())[:answer_limit]}
    protocol = {"dataset": DATASET, "revision": data["revision"], "questions": len(rows),
                "eligible": len(eligible), "excluded": excluded, "source_format": "article XML wrappers from available full-text sources",
                "source_policy": source_policy,
                "partial_source_questions": [row["id"] for row in eligible if not all(s in mapping for s in row["sources"])],
                "input": "original question only; no title, DOI, gold evidence or source filter",
                "profile": "evidence_minilm", "reference_budget": 4096,
                "corpus_papers": len(set(mapping.values())), "result_cache": False,
                "pdf_parsing_evaluated": False, "tuning_on_this_set": False}
    with sqlite3.connect(database) as connection:
        corpus = connection.execute("SELECT id,file_hash FROM papers ORDER BY id").fetchall()
    protocol["corpus_fingerprint"] = sha256(json.dumps(corpus).encode()).hexdigest()
    protocol["questions_sha256"] = sha256((ROOT / "questions.json").read_bytes()).hexdigest()
    protocol["implementation"] = {
        name: sha256(Path("src/paperanchor", name + ".py").read_bytes()).hexdigest()
        for name in ("evidence_profile", "evidence_retrieval", "evidence_context", "representation",
                     "unit_index", "semantic", "rag", "ollama")}
    protocol["adapter_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    previous_path = output / "protocol.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        for key in ("corpus_fingerprint", "questions_sha256", "profile", "input", "implementation", "adapter_sha256", "source_policy"):
            if previous.get(key) != protocol[key]:
                raise ValueError(f"Resume protocol mismatch: {key}; use a fresh output directory")
    if answer:
        config = load_model_config()
        limits = generation_config(config)
        protocol["answer_model"] = {"provider": config.provider, "model": config.model,
                                    "thinking": get_setting("PAPERANCHOR_OLLAMA_THINKING") or "auto",
                                    "context_tokens": limits.context_tokens, "max_tokens": limits.max_tokens}
        protocol["answer_sample"] = {"method": "first N sorted SHA256(20261009:id), before seeing retrieval outcomes",
                                     "ids": sorted(answer_ids)}
        if previous_path.exists() and previous.get("answer_model", protocol["answer_model"]) != protocol["answer_model"]:
            raise ValueError("Answer model changed; use a fresh output directory")
    save(output / "protocol.json", protocol)
    profile = EvidenceProfile(database)
    profile.prepare("What experimental methods were used?", profile_name="evidence_minilm")
    profile._engine.cache_enabled = False
    trace = {}
    original_retrieve = profile._engine.retrieve

    def observed_retrieve(question, **kwargs):
        if kwargs.get("title") or kwargs.get("paper_ids") is not None or kwargs.get("scope_paper_ids") is not None:
            raise ValueError("Question-only evaluation received a source hint")
        result = original_retrieve(question, **kwargs)
        trace.clear()
        trace.update(queries=result["queries"], selected_papers=result["selected_papers"],
                     candidate_papers=sorted({hit.paper_id for hit in result["candidates"]["union"]}),
                     ranked=[{"paper_id": hit.paper_id, "passage_id": hit.passage_id, "text": hit.text}
                             for hit in result["ranked"]])
        return result

    profile._engine.retrieve = observed_retrieve
    results = []
    for i, row in enumerate(eligible, 1):
        record_path = output / f'{row["id"]}.json'
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if not answer or row["id"] not in answer_ids or "answer" in record:
                results.append(record)
                continue
        else:
            start = time.perf_counter()
            try:
                context = profile.prepare(row["question"], profile_name="evidence_minilm")
                gold_ids = {mapping.get(s, "missing:" + s) for s in row["sources"]}
                record = {"id": row["id"], "question": row["question"], "context": context,
                          "retrieval_seconds": time.perf_counter() - start,
                          "trace": dict(trace),
                          "metrics": score_context(context, gold_ids, row.get("key-passage"))}
                record["metrics"]["candidate_source_hit"] = bool(gold_ids & set(trace["candidate_papers"]))
                ranked_papers = [item["paper_id"] for item in trace["ranked"]]
                record["metrics"]["top20_source_hit"] = bool(gold_ids & set(ranked_papers))
                record["metrics"]["source_mrr_at20_units"] = next(
                    (1 / (j+1) for j, pid in enumerate(ranked_papers) if pid in gold_ids), 0)
            except Exception as error:
                record = {"id": row["id"], "question": row["question"], "error": str(error)}
        if answer and row["id"] in answer_ids and "context" in record:
            choices = [(row["ideal"], "correct")] + [(s, "distractor") for s in row["distractors"]]
            choices.append(("Insufficient information to answer the question", "abstain"))
            random.Random(row["id"]).shuffle(choices)
            prompt = row["question"] + "\n\n" + "\n".join(f"{chr(65+j)}. {s}" for j,(s,_) in enumerate(choices))
            prompt += "\nChoose based only on the provided evidence. Explain briefly with evidence citations, then end with [ANSWER]letter[/ANSWER]. Select insufficient information if the evidence does not support a choice."
            started = time.perf_counter()
            try:
                result = answer_from_context(prompt, record["context"], generate_with_config)
                matches = re.findall(r"\[ANSWER\]\s*([A-Z])\s*\[/ANSWER\]", result.text)
                idx = ord(matches[-1]) - 65 if matches else -1
                category = choices[idx][1] if 0 <= idx < len(choices) else "invalid"
                record["answer"] = {"text": result.text, "status": result.status, "category": category,
                                    "seconds": time.perf_counter()-started, "choices": [s for s,_ in choices]}
                record.pop("answer_error", None)
            except Exception as error:
                record["answer_error"] = str(error)
        save(record_path, record)
        results.append(record)
        print(f"Evaluated {i}/{len(eligible)} source_hit={record.get('metrics',{}).get('source_hit')} answer={record.get('answer',{}).get('category')}", flush=True)
        summarize(results, protocol, output)
    profile.close()
    summarize(results, protocol, output)


def summarize(results, protocol, output):
    n = len(results)
    good = [r for r in results if "metrics" in r]
    summary = {"evaluated": n, "eligible": protocol["eligible"], "total_questions": protocol["questions"],
               "retrieval_errors": n-len(good), "corpus_papers": protocol["corpus_papers"],
               "source_hits": sum(r["metrics"]["source_hit"] for r in good),
               "all_source_hits": sum(r["metrics"]["all_sources"] for r in good),
               "candidate_source_hits": sum(r["metrics"]["candidate_source_hit"] for r in good),
               "top20_source_hits": sum(r["metrics"]["top20_source_hit"] for r in good),
               "key_passage_exact": sum(r["metrics"]["key_passage_exact"] is True for r in good),
               "mean_retrieval_seconds": sum(r["retrieval_seconds"] for r in good)/max(1,len(good)),
               "answers": {k: sum(r.get("answer",{}).get("category")==k for r in results)
                           for k in ("correct", "distractor", "abstain", "invalid")},
               "answer_errors": sum("answer_error" in r for r in results)}
    summary["answer_sample_size"] = len(protocol.get("answer_sample", {}).get("ids", []))
    summary["status"] = "complete" if n == protocol["eligible"] and len(good) == n else "incomplete"
    answers = summary["answers"]
    answer_n = summary["answer_sample_size"]
    summary["answer_status"] = ("not_requested" if not answer_n else "complete"
                                if sum(answers.values()) == answer_n and not summary["answer_errors"]
                                else "incomplete")
    if answer_n:
        summary["answer_accuracy"] = answers["correct"] / answer_n
        summary["answer_coverage"] = (answers["correct"] + answers["distractor"]) / answer_n
        summary["answer_precision"] = answers["correct"] / max(1, answers["correct"] + answers["distractor"])
    save(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "build", "run"])
    parser.add_argument("--database", type=Path, default=Path(".paperanchor/litqa2.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path(".study/litqa2-question-only-20261009"))
    parser.add_argument("--answer", action="store_true")
    parser.add_argument("--answer-limit", type=int, default=30)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    parser.add_argument("--source-policy", choices=["all", "any"], default="all")
    args = parser.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "build": build(args.database, args.manifest)
    else: run(args.database, args.output, args.answer, args.answer_limit, args.source_policy)


if __name__ == "__main__":
    main()
