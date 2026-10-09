import pytest

from paperanchor.evidence_policies import BGE, MINILM, PolicyScorer, balanced_order, ranking


class Engine:
    def __init__(self):
        self.calls = []
        self.last_score_trace = {}

    def score_with(self, model, query, texts):
        self.calls.append((model, query, list(texts)))
        self.last_score_trace = {"model": model, "documents": len(texts)}
        return {BGE: [1000, 100, -20], MINILM: [0.1, 0.3, 0.2]}[model][:len(texts)]


def test_rrf_uses_ranks_on_identical_inputs_despite_score_scale():
    engine = Engine()
    scorer = PolicyScorer(engine, {BGE: BGE, MINILM: MINILM}, "rrf_equal")
    result = scorer.score("question", ["a", "b", "c"])
    assert result == pytest.approx([1/61 + 1/63, 1/62 + 1/61, 1/63 + 1/62])
    assert ranking(result) == [1, 0, 2]
    assert [call[2] for call in engine.calls] == [["a", "b", "c"]] * 2
    assert scorer.last_score_trace["models"][MINILM]["model"] == MINILM


def test_single_model_never_calls_the_other_model():
    engine = Engine()
    scorer = PolicyScorer(engine, {MINILM: MINILM}, "minilm")
    assert scorer.score("q", ["a", "b", "c"]) == [0.1, 0.3, 0.2]
    assert len(engine.calls) == 1


def test_weighting_and_stable_ties():
    scorer = PolicyScorer(Engine(), {BGE: BGE, MINILM: MINILM}, "rrf_minilm2")
    assert scorer.score("q", ["a", "b", "c"]) == pytest.approx(
        [1/61 + 2/63, 1/62 + 2/61, 1/63 + 2/62])
    assert ranking([0.2, 0.2, 0.2]) == [0, 1, 2]


def test_balanced_keeps_both_top_four_and_fills_duplicates():
    order = balanced_order(list(range(24)), list(range(23, -1, -1)))
    assert order[:8] == [0, 1, 2, 3, 23, 22, 21, 20]
    assert order[8:12] == [4, 19, 5, 18]
    assert len(order) == len(set(order)) == 24
    assert balanced_order(list(range(24)), list(range(24))) == list(range(24))
    assert balanced_order([], []) == []


def test_bad_policy_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        PolicyScorer(Engine(), {}, "unknown")
