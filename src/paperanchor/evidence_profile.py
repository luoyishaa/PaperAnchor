"""Product adapter for the shared, budgeted evidence retrieval pipeline."""

from contextlib import closing
from pathlib import Path
import json
import sqlite3
from threading import Lock

from .semantic import FastEmbedProvider, FastEmbedReranker
from .evidence_revision import ensure_revision_tracking
from .evidence_policies import BGE, MINILM, POLICIES, PolicyScorer


# S2_3: selected by the QASPER 120 comparison (80/96 complete evidence coverage).
EVIDENCE_V2_CONFIG = {
    "route_top": 1,
    "depth": 40,
    "query_mode": "question",
    "rerank_mode": "question",
    "pool": "hybrid",
    "strategy": "ranked_expansion",
    "protect": 8,
    "rerank_model": "BAAI/bge-reranker-base",
}
OPTIMIZED_PROFILE_PATH = Path(__file__).with_name("evidence_optimized.json")
PROFILE_NAMES = ("legacy", "evidence_v2", "evidence_minilm", "evidence_optimized")
REFERENCE_TOKENIZER = Path(
    "models--qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q/snapshots/"
    "faf4aa4225822f3bc6376869cb1164e8e3feedd0/tokenizer.json"
)


class EvidenceProfileUnavailable(RuntimeError):
    """The optional profile is missing a required local model or index."""


class ReferenceTokenizer:
    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        if not path.is_file():
            raise EvidenceProfileUnavailable(
                "Expanded evidence needs its locally cached reference tokenizer. "
                "Use Standard search until the local model cache is available.")
        self.path = path.resolve()
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


class EvidenceProfile:
    """Keep models and corpus warm; refresh only after indexed content changes."""

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.config = dict(EVIDENCE_V2_CONFIG)
        cache_dir = self.database_path.parent / "models"
        self.provider = FastEmbedProvider(cache_dir=cache_dir, local_files_only=True)
        self.reranker = FastEmbedReranker(
            self.config["rerank_model"], cache_dir=cache_dir, local_files_only=True)
        self.models = {BGE: self.reranker,
                       MINILM: FastEmbedReranker(MINILM, cache_dir=cache_dir, local_files_only=True)}
        self.optimized_path = OPTIMIZED_PROFILE_PATH
        self.tokenizer_path = cache_dir / REFERENCE_TOKENIZER
        self._tokenizer = None
        self._engine = None
        self._revision = None
        self._lock = Lock()

    def _optimized_spec(self) -> dict:
        if not self.optimized_path.is_file():
            raise EvidenceProfileUnavailable("An optimized search profile has not been installed.")
        try:
            spec = json.loads(self.optimized_path.read_text(encoding="utf-8"))
            config = spec["config"]
            required = {"route_top", "depth", "query_mode", "rerank_mode", "pool",
                        "candidate_policy", "expansion_policy", "strategy", "protect",
                        "supplement_depth"}
            if spec.get("schema_version") != 1 or not required <= config.keys():
                raise ValueError("Unsupported optimized profile format")
            if (config["candidate_policy"] not in POLICIES
                    or config["expansion_policy"] not in POLICIES):
                raise ValueError("Unknown optimized scoring policy")
            for key in ("route_top", "depth", "protect", "supplement_depth"):
                minimum = 0 if key == "supplement_depth" else 1
                if type(config[key]) is not int or config[key] < minimum:
                    raise ValueError(f"Invalid optimized {key}")
            if (config["query_mode"] not in ("question", "display", "dual")
                    or config["rerank_mode"] not in ("question", "display")
                    or config["pool"] not in ("hybrid", "all_units", "all_blocks")
                    or config["strategy"] not in ("seed_first", "dedup_parent", "ranked_expansion")):
                raise ValueError("Unknown optimized retrieval or assembly strategy")
            aliases = spec.get("title_aliases", [])
            if not isinstance(aliases, list):
                raise ValueError("Title aliases must be a list")
            for item in aliases:
                if (not isinstance(item["paper_id"], int)
                        or not isinstance(item["file_hash"], str) or len(item["file_hash"]) != 64
                        or not isinstance(item["title"], str) or not item["title"].strip()):
                    raise ValueError("Invalid title alias")
            if len({item["paper_id"] for item in aliases}) != len(aliases):
                raise ValueError("Duplicate title aliases")
            return spec
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise EvidenceProfileUnavailable(f"The optimized search profile is invalid: {error}") from error

    @staticmethod
    def _bound_aliases(spec: dict, paper_hashes: dict[int, str]) -> dict[int, str]:
        aliases = {}
        for item in spec.get("title_aliases", []):
            if paper_hashes.get(item["paper_id"]) != item["file_hash"]:
                raise EvidenceProfileUnavailable(
                    f"Optimized search is unavailable: the indexed PDF for paper {item['paper_id']} "
                    "does not match its saved title metadata.")
            aliases[item["paper_id"]] = item["title"]
        return aliases

    def availability(self) -> dict:
        profiles = ["legacy", "evidence_v2", "evidence_minilm"]
        errors = {}
        try:
            spec = self._optimized_spec()
            with closing(sqlite3.connect(
                    f"{self.database_path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
                hashes = dict(connection.execute("SELECT id,file_hash FROM papers"))
            self._bound_aliases(spec, hashes)
            profiles.append("evidence_optimized")
        except (EvidenceProfileUnavailable, sqlite3.Error) as error:
            errors["evidence_optimized"] = str(error)
        return {"profiles": profiles, "errors": errors}

    def _current_engine(self):
        from .evidence_retrieval import EvidenceRetriever

        if self._engine is not None:
            self._engine.begin_snapshot()
            if self._engine.snapshot_revision() == self._revision:
                return self._engine
            self._engine.close()
            self._engine = None
        # This also installs triggers for relevant tables created since the
        # preceding snapshot; schema_version is part of snapshot_revision.
        ensure_revision_tracking(self.database_path)
        self._engine = EvidenceRetriever(self.database_path, self.provider, self.reranker)
        self._revision = self._engine.snapshot_revision()
        return self._engine

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
                self._revision = None

    def prepare(self, question: str, *, title: str = "",
                paper_ids: tuple[int, ...] | None = None,
                scope_paper_ids: tuple[int, ...] | None = None,
                profile_name: str = "evidence_v2") -> dict:
        from .evidence_context import assemble_evidence

        with self._lock:
            engine = None
            try:
                if profile_name not in PROFILE_NAMES or profile_name == "legacy":
                    raise EvidenceProfileUnavailable("Unknown expanded evidence profile")
                spec = self._optimized_spec() if profile_name == "evidence_optimized" else None
                config = dict(spec["config"] if spec else self.config)
                if self._tokenizer is None:
                    self._tokenizer = ReferenceTokenizer(self.tokenizer_path)
                engine = self._current_engine()
                models = {**self.models, BGE: self.reranker}
                options = {}
                expansion_ranker = engine
                if spec is not None:
                    options = {
                        "title_aliases": self._bound_aliases(spec, engine.paper_hashes),
                        "supplement_depth": config["supplement_depth"],
                        "ranker": PolicyScorer(engine, models, config["candidate_policy"]),
                    }
                    expansion_ranker = PolicyScorer(engine, models, config["expansion_policy"])
                else:
                    engine.reranker = models[MINILM if profile_name == "evidence_minilm" else BGE]
                result = engine.retrieve(
                    question, title=title, paper_ids=paper_ids,
                    scope_paper_ids=scope_paper_ids, **options,
                    **{key: config[key] for key in (
                        "route_top", "depth", "query_mode", "rerank_mode", "pool")})
                return assemble_evidence(
                    result["ranked"], engine.blocks, self._tokenizer,
                    strategy=config["strategy"], budget=4096,
                    reranker=expansion_ranker, query=question, protect=config["protect"])
            except EvidenceProfileUnavailable:
                raise
            except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
                raise EvidenceProfileUnavailable(
                    f"Expanded evidence is unavailable: {error}") from error
            finally:
                if self._engine is not None:
                    # An idle read transaction blocks DELETE-journal writers
                    # and pins WAL frames. Keep memory, release the DB snapshot.
                    self._engine.release_snapshot()
