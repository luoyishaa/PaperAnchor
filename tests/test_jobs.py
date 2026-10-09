from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier

import pytest

from paperanchor.jobs import JobStore


@pytest.mark.parametrize("round_number", range(5))
def test_concurrent_enqueue_deduplicates_active_kind(tmp_path: Path, round_number: int) -> None:
    path = tmp_path / "jobs.sqlite3"
    stores = [JobStore(path) for _ in range(8)]
    barrier = Barrier(len(stores))

    def enqueue(store):
        barrier.wait(timeout=10)
        return store.enqueue("semantic")

    with ThreadPoolExecutor(max_workers=len(stores)) as executor:
        results = list(executor.map(enqueue, stores))

    assert len({job.id for job in results}) == 1
    assert all(job.state == "queued" for job in results)
    with closing(stores[0]._connect()) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1

    claimed = stores[0].claim_next()
    assert claimed.id == results[0].id
    assert stores[1].enqueue("semantic") == claimed
    stores[0].finish(claimed.id)
    assert stores[1].enqueue("semantic").id != claimed.id


@pytest.mark.parametrize("round_number", range(5))
def test_concurrent_claims_are_unique(tmp_path: Path, round_number: int) -> None:
    path = tmp_path / "jobs.sqlite3"
    stores = [JobStore(path) for _ in range(8)]
    queued = [stores[0].enqueue(f"kind-{number}") for number in range(20)]
    barrier = Barrier(len(stores))

    def drain(store):
        barrier.wait(timeout=10)
        claimed = []
        while (job := store.claim_next()) is not None:
            assert job.state == "running"
            claimed.append(job.id)
        return claimed

    with ThreadPoolExecutor(max_workers=len(stores)) as executor:
        results = list(executor.map(drain, stores))

    claimed_ids = [job_id for result in results for job_id in result]
    assert len(claimed_ids) == len(set(claimed_ids)) == len(queued)
    assert set(claimed_ids) == {job.id for job in queued}
    assert stores[0].claim_next() is None


def test_fifo_progress_and_terminal_states(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    first = store.enqueue("semantic")
    second = store.enqueue("other")
    assert store.latest("semantic") == first
    assert store.latest("missing") is None
    assert store.get(-1) is None
    store.progress(first.id, 1, 3)
    assert store.get(first.id).completed == 0
    assert store.claim_next().id == first.id
    store.progress(first.id, 2, 3)
    assert (store.get(first.id).completed, store.get(first.id).total) == (2, 3)
    store.finish(first.id)
    store.progress(first.id, 3, 3)
    assert store.get(first.id).state == "succeeded"
    assert store.get(first.id).completed == 2
    assert store.claim_next().id == second.id
    store.finish(second.id, error="embedding failed")
    assert store.get(second.id).state == "failed"
    assert store.get(second.id).error == "embedding failed"
    assert store.enqueue("other").id != second.id


def test_reopen_and_recover_interrupted_job(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite3"
    store = JobStore(path)
    interrupted = store.enqueue("semantic")
    store.claim_next()
    store.progress(interrupted.id, 2, 5)
    succeeded = store.enqueue("succeeded")
    store.finish(succeeded.id)
    succeeded = store.get(succeeded.id)
    failed = store.enqueue("failed")
    store.finish(failed.id, error="previous failure")
    failed = store.get(failed.id)
    queued = store.enqueue("queued")
    before = store.get(interrupted.id)

    reopened = JobStore(path)
    assert reopened.get(interrupted.id) == before
    reopened.recover()
    recovered = reopened.get(interrupted.id)
    assert recovered.state == "queued"
    assert (recovered.completed, recovered.total) == (2, 5)
    assert recovered.created_at == before.created_at
    assert recovered.updated_at >= before.updated_at
    assert reopened.get(succeeded.id) == succeeded
    assert reopened.get(failed.id) == failed
    assert reopened.get(queued.id) == queued
    reopened.recover()
    assert reopened.get(interrupted.id) == recovered
    assert reopened.enqueue("semantic") == recovered
    assert reopened.claim_next().id == interrupted.id
    reopened.finish(interrupted.id)
    assert reopened.claim_next().id == queued.id
    assert reopened.claim_next() is None


@pytest.mark.parametrize("operation", ["enqueue", "claim_next"])
def test_failed_transaction_releases_writer_lock(tmp_path: Path, monkeypatch, operation: str) -> None:
    path = tmp_path / "jobs.sqlite3"
    store = JobStore(path)
    queued = store.enqueue("semantic") if operation == "claim_next" else None

    def fail(row):
        raise RuntimeError("job decoding failed")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_job", fail)
        with pytest.raises(RuntimeError, match="job decoding failed"):
            if operation == "enqueue":
                store.enqueue("semantic")
            else:
                store.claim_next()

    reopened = JobStore(path)
    if queued is None:
        assert reopened.latest("semantic") is None
        queued = reopened.enqueue("semantic")
    else:
        assert reopened.get(queued.id) == queued
    assert reopened.claim_next().id == queued.id
