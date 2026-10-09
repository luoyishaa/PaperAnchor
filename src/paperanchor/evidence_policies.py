"""Model-aware ranking policies shared by evaluation and product requests."""

from copy import deepcopy
import math


BGE = "BAAI/bge-reranker-base"
MINILM = "Xenova/ms-marco-MiniLM-L-6-v2"
POLICIES = ("bge", "minilm", "rrf_equal", "rrf_minilm2", "rrf_bge2", "balanced")


def ranking(values):
    return sorted(range(len(values)), key=lambda index: (-values[index], index))


def balanced_order(minilm_order, bge_order):
    """Protect four from each model, then alternate unseen candidates."""
    result, seen = [], set()

    def add(index):
        if index not in seen:
            seen.add(index)
            result.append(index)

    for order in (minilm_order, bge_order):
        for index in order[:4]:
            add(index)
    cursors = [4, 4]
    orders = (minilm_order, bge_order)
    while any(cursors[i] < len(order) for i, order in enumerate(orders)):
        for i, order in enumerate(orders):
            while cursors[i] < len(order) and order[cursors[i]] in seen:
                cursors[i] += 1
            if cursors[i] < len(order):
                add(order[cursors[i]])
                cursors[i] += 1
    return result


class PolicyScorer:
    """Fuse ranks on one identical input list; cache only native model scores."""

    def __init__(self, engine, models, policy):
        if policy not in POLICIES:
            raise ValueError(f"Unknown ranking policy: {policy}")
        self.engine, self.models, self.policy = engine, models, policy
        self.last_score_trace = {}

    def score(self, query, documents):
        documents = list(documents)
        model_names = ([BGE] if self.policy == "bge" else [MINILM]
                       if self.policy == "minilm" else [MINILM, BGE])
        raw, ranks, orders, traces = {}, {}, {}, {}
        for name in model_names:
            values = list(self.engine.score_with(self.models[name], query, documents))
            if len(values) != len(documents) or not all(math.isfinite(float(v)) for v in values):
                raise ValueError("Ranking policy received invalid model scores")
            raw[name] = [float(v) for v in values]
            orders[name] = ranking(values)
            ranks[name] = [0] * len(values)
            for rank, index in enumerate(orders[name], 1):
                ranks[name][index] = rank
            traces[name] = deepcopy(self.engine.last_score_trace)
        weights = None
        if len(model_names) == 1:
            result = raw[model_names[0]]
        elif self.policy == "balanced":
            order = balanced_order(orders[MINILM], orders[BGE])
            result = [0.0] * len(documents)
            for position, index in enumerate(order):
                result[index] = float(len(documents) - position)
        else:
            weights = {MINILM: 2 if self.policy == "rrf_minilm2" else 1,
                       BGE: 2 if self.policy == "rrf_bge2" else 1}
            result = [sum(weights[name] / (60 + ranks[name][index]) for name in model_names)
                      for index in range(len(documents))]
        self.last_score_trace = {
            "policy": self.policy, "rrf_k": 60 if weights else None,
            "weights": weights, "documents": len(documents), "raw_scores": raw,
            "ranks": ranks, "scores": result, "order": ranking(result), "models": traces,
        }
        return result
