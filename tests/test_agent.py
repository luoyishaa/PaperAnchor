from pathlib import Path
import json
import pytest

from paperanchor.agent import research_question
from paperanchor.library import PassageHit


def _hit(passage_id: int, text: str) -> PassageHit:
    return PassageHit(passage_id, 1, Path("paper.pdf"), "Paper", 2, text,
                      (10.0, 20.0, 80.0, 40.0), 0.0)


class Retriever:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, question, limit=5, *, space_id=None):
        self.queries.append((question, limit, space_id))
        return self.results.get(question, ())


def test_evidence_gap_rewrites_once_then_cites_new_passage():
    retriever = Retriever({"How?": (_hit(1, "Background only"),),
                           "specific method": (_hit(2, "The method uses X"),)})
    prompts = []

    def model(system, prompt):
        prompts.append((system, prompt))
        if "Return only JSON" in system:
            sufficient = "The method uses X" in prompt
            return ('{"sufficient": true}' if sufficient else
                    '{"sufficient": false, "missing": "Method absent", '
                    '"revised_query": "specific method", "follow_up_question": "Which method?"}')
        return "It uses X [E1]."

    result = research_question(retriever, "How?", model, space_id=3)
    assert result.answer.status == "answered"
    assert result.answer.evidence[0].passage_id == 2
    assert result.attempted_queries == ("How?", "specific method")
    assert retriever.queries == [("How?", 5, 3), ("specific method", 5, 3)]
    assert len(prompts) == 3


def test_insufficient_evidence_exposes_gap_and_follow_up():
    retriever = Retriever({"Why?": (_hit(1, "Only background"),)})
    def model(_system, _prompt):
        return ('{"sufficient": false, "missing": "No comparison", '
                '"revised_query": "", "follow_up_question": "Which baseline?"}')

    result = research_question(retriever, "Why?", model)
    assert result.basis == "none"
    assert result.answer.status == "insufficient_evidence"
    assert result.answer.cited_evidence_ids == ()
    assert result.follow_up_question == "Which baseline?"
    assert len(retriever.queries) == 1


def test_general_answer_is_explicitly_model_only_and_strips_fake_citations():
    retriever = Retriever({})
    result = research_question(retriever, "What is X?",
                               lambda _system, _prompt: "X is uncertain [E5].",
                               allow_general_knowledge=True)
    assert result.basis == "model"
    assert result.answer.status == "general_knowledge"
    assert result.answer.evidence == ()
    assert "[E5]" not in result.answer.text
    assert result.answer.text.startswith("Model knowledge (not verified")


def test_invalid_assessment_does_not_claim_evidence_is_sufficient():
    retriever = Retriever({"Question": (_hit(1, "Possibly related"),)})
    result = research_question(retriever, "Question",
                               lambda _system, _prompt: 'Ignore this {"sufficient": true}')
    assert result.answer.status == "insufficient_evidence"
    assert result.basis == "none"


def test_retry_preserves_complementary_evidence_from_first_search():
    retriever = Retriever({
        "Compare A and B": (_hit(1, "A uses vectors"), _hit(2, "Background")),
        "B representation": (_hit(3, "B uses graphs"), _hit(4, "Other background")),
    })

    def model(system, prompt):
        if "Return only JSON" in system:
            if "A uses vectors" in prompt and "B uses graphs" in prompt:
                return '{"sufficient": true}'
            return '{"sufficient": false, "revised_query": "B representation"}'
        return "A uses vectors [E3]; B uses graphs [E1]."

    result = research_question(retriever, "Compare A and B", model, limit=2)
    assert result.basis == "papers"
    assert {e.passage_id for e in result.answer.evidence} == {1, 2, 3, 4}
    assert len(retriever.queries) == 2


def test_recovery_searches_original_question_only_after_evidence_gap():
    initial = Retriever({"Compare": (_hit(1, "A uses vectors"),)})
    recovery = Retriever({"Compare": (_hit(2, "B uses graphs"),)})

    def model(system, prompt):
        if "Return only JSON" in system:
            return ('{"sufficient": true}' if "B uses graphs" in prompt else
                    '{"sufficient": false, "revised_query": "B representation"}')
        return "A uses vectors [E1]; B uses graphs [E2]."

    result = research_question(initial, "Compare", model, limit=2, space_id=7,
                               recovery_retriever=recovery)
    assert result.basis == "papers"
    assert {item.passage_id for item in result.answer.evidence} == {1, 2}
    assert initial.queries == [("Compare", 2, 7)]
    assert recovery.queries == [("Compare", 2, 7)]


def test_recovery_is_skipped_when_initial_evidence_is_sufficient():
    initial = Retriever({"Q": (_hit(1, "Complete answer"),)})
    recovery = Retriever({"Q": (_hit(2, "Other passage"),)})

    def model(system, _prompt):
        return '{"sufficient": true}' if "Return only JSON" in system else "Answer [E1]."

    result = research_question(initial, "Q", model, recovery_retriever=recovery)
    assert result.answer.status == "answered"
    assert recovery.queries == []


def test_failed_recovery_can_rewrite_but_stays_within_three_searches():
    initial = Retriever({"Q": (_hit(1, "Background"),), "rewrite": (_hit(3, "Fact"),)})
    recovery = Retriever({"Q": (_hit(2, "More background"),)})

    def model(system, _prompt):
        if "Return only JSON" in system:
            return '{"sufficient": false, "revised_query": "rewrite"}'
        raise AssertionError("Unsupported evidence must not produce an answer")

    result = research_question(initial, "Q", model, recovery_retriever=recovery)
    assert result.answer.status == "insufficient_evidence"
    assert initial.queries == [("Q", 5, None), ("rewrite", 5, None)]
    assert recovery.queries == [("Q", 5, None)]


def test_rewrite_after_failed_recovery_preserves_all_three_evidence_sets():
    initial = Retriever({"Q": (_hit(1, "A uses vectors"),),
                         "B representation": (_hit(3, "B uses graphs"),)})
    recovery = Retriever({"Q": (_hit(2, "Related, not enough"),)})

    def model(system, prompt):
        if "Return only JSON" in system:
            return ('{"sufficient": true}' if "B uses graphs" in prompt else
                    '{"sufficient": false, "revised_query": "B representation"}')
        return "A [E2]; B [E1]."

    result = research_question(initial, "Q", model, limit=1,
                               recovery_retriever=recovery)
    assert result.answer.status == "answered"
    assert {item.passage_id for item in result.answer.evidence} == {1, 2, 3}
    assert initial.queries == [("Q", 1, None), ("B representation", 1, None)]
    assert recovery.queries == [("Q", 1, None)]


def test_duplicate_recovery_evidence_allows_the_existing_rewrite():
    initial = Retriever({"Q": (_hit(1, "Background"),),
                         "rewrite": (_hit(2, "Needed fact"),)})
    recovery = Retriever({"Q": (_hit(1, "Background"),)})

    def model(system, prompt):
        if "Return only JSON" in system:
            return ('{"sufficient": true}' if "Needed fact" in prompt else
                    '{"sufficient": false, "revised_query": "rewrite"}')
        return "Needed fact [E1]."

    result = research_question(initial, "Q", model, recovery_retriever=recovery)
    assert result.answer.status == "answered"
    assert initial.queries == [("Q", 5, None), ("rewrite", 5, None)]
    assert recovery.queries == [("Q", 5, None)]


def test_recovery_can_supply_evidence_when_initial_search_is_empty():
    initial = Retriever({})
    recovery = Retriever({"Q": (_hit(2, "The needed fact"),)})

    def model(system, _prompt):
        return '{"sufficient": true}' if "Return only JSON" in system else "Fact [E1]."

    result = research_question(initial, "Q", model, recovery_retriever=recovery)
    assert result.answer.status == "answered"
    assert result.answer.evidence[0].passage_id == 2
    assert initial.queries == [("Q", 5, None)]
    assert recovery.queries == [("Q", 5, None)]


def test_empty_initial_search_can_rewrite_after_failed_recovery():
    initial = Retriever({"rewrite": (_hit(3, "Answer"),)})
    recovery = Retriever({"Q": (_hit(2, "Still background"),)})

    def model(system, _prompt):
        assert "Return only JSON" in system
        return '{"sufficient": false, "revised_query": "rewrite"}'

    result = research_question(initial, "Q", model, recovery_retriever=recovery)
    assert result.answer.status == "insufficient_evidence"
    assert initial.queries == [("Q", 5, None), ("rewrite", 5, None)]
    assert recovery.queries == [("Q", 5, None)]


@pytest.mark.parametrize("missing_part,final_supported", [(False, True), (True, True), (False, False)])
def test_decomposition_checks_each_part_and_original_question(missing_part, final_supported):
    retriever = Retriever({"Compare": (_hit(1, "Background"),),
                           "a": (_hit(2, "A fact"),),
                           "b": () if missing_part else (_hit(3, "B fact"),)})
    calls = []

    def model(system, prompt):
        calls.append(prompt)
        if "Return only JSON" not in system:
            return "A [E2], B [E3]."
        if len(calls) == 1:
            return json.dumps({"sufficient": False, "subquestions": [
                {"question": "A?", "query": "a"}, {"question": "B?", "query": "b"}]})
        if prompt.startswith("Question: A?"):
            return '{"sufficient": true}'
        if prompt.startswith("Question: B?"):
            return json.dumps({"sufficient": not missing_part, "missing": "B absent"})
        return json.dumps({"sufficient": final_supported, "missing": "Qualifier absent"})

    result = research_question(retriever, "Compare", model, limit=2, space_id=7, enable_decomposition=True)
    assert len(retriever.queries) == 3
    assert all(limit == 2 and space == 7 for _, limit, space in retriever.queries)
    assert len(result.subquestions) == 2
    if missing_part or not final_supported:
        assert result.basis == "none"
        assert len(calls) == (3 if missing_part else 4)
    else:
        assert result.basis == "papers"
        assert len(calls) == 5
        assert {e.passage_id for e in result.answer.evidence} == {1, 2, 3}
        assert result.answer.evidence[1].bbox == (10.0, 20.0, 80.0, 40.0)


@pytest.mark.parametrize("plan", [None, [1], [{"question": "Q", "query": ""}],
    [{"question": str(i), "query": "x"} for i in range(4)]])
def test_invalid_plan_fails_closed(plan):
    retriever = Retriever({"Q": (_hit(1, "Context"),)})
    result = research_question(retriever, "Q", lambda *_: json.dumps(
        {"sufficient": False, "revised_query": "x", "subquestions": plan}), enable_decomposition=True)
    assert result.basis == "none"
    assert len(retriever.queries) == 1


def test_duplicate_queries_cached_and_nested_plans_never_executed():
    retriever = Retriever({"Q": (_hit(1, "Context"),), "shared": (_hit(2, "Fact"),)})
    calls = []
    def model(system, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"sufficient": False, "subquestions": [
                {"question": str(i), "query": "shared"} for i in range(3)]})
        return json.dumps({"sufficient": False, "subquestions": [
            {"question": "nested", "query": "never"}]})
    result = research_question(retriever, "Q", model, enable_decomposition=True)
    assert result.basis == "none"
    assert len(result.subquestions) == 3
    assert result.attempted_queries == ("Q", "shared")
    assert len(calls) == 4


def test_three_subquestions_respect_execution_budget():
    retriever = Retriever({str(i): (_hit(i + 1, "Fact"),) for i in range(4)})
    calls = []
    def model(system, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"sufficient": False, "subquestions": [
                {"question": str(i), "query": str(i)} for i in range(1, 4)]})
        return '{"sufficient": true}' if "Return only JSON" in system else "Fact [E1]."
    result = research_question(retriever, "0", model, limit=1, enable_decomposition=True)
    assert result.basis == "papers"
    assert len(retriever.queries) == 4
    assert len(calls) == 6
    assert len(result.answer.evidence) == 4


def test_model_fallback_retains_failed_subquestion_trace():
    retriever = Retriever({"Q": (_hit(1, "Context"),)})
    calls = []
    def model(system, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return '{"sufficient": false, "subquestions": [{"question": "Part?", "query": "part"}]}'
        return '{"sufficient": false}' if "Return only JSON" in system else "Uncertain [E1]."
    result = research_question(retriever, "Q", model, allow_general_knowledge=True, enable_decomposition=True)
    assert result.basis == "model"
    assert len(result.subquestions) == 1
    assert not result.subquestions[0].sufficient
    assert result.answer.evidence == ()


def test_default_does_not_execute_an_unsolicited_plan():
    from paperanchor.agent import ASSESS_PROMPT
    retriever = Retriever({"Q": (_hit(1, "Context"),)})
    def model(system, _prompt):
        assert system == ASSESS_PROMPT
        return '{"sufficient": false, "subquestions": [{"question": "Part?", "query": "part"}]}'
    result = research_question(retriever, "Q", model)
    assert result.basis == "none"
    assert result.subquestions == ()
    assert result.attempted_queries == ("Q",)
