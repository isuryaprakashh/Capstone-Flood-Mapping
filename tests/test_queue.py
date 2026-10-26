"""
Tests for the experiment queue's failure handling and the EMA warm-up.
"""

import json

import pytest
import torch

import training.queue as queue
from training.engine import EMA


def _runs(n):
    return [{"id": f"run_{i}", "model": "m", "protocol": "P1", "seed": 0, "priority": i} for i in range(n)]


def _status(results, run_id):
    return json.loads((results / run_id / "status.json").read_text())


def test_failed_run_is_not_retried_in_the_same_session(tmp_path, monkeypatch):
    calls = []

    def flaky(cfg, *a, **k):
        calls.append(cfg["id"])
        if calls == ["run_0"]:  # only the very first call fails
            raise RuntimeError("download blip")
        return "completed"

    monkeypatch.setattr(queue, "train_run", flaky)
    queue.run_worker(_runs(3), tmp_path, tmp_path, session="s1")
    assert calls == ["run_0", "run_1", "run_2"]  # run_0 not retried in s1
    assert _status(tmp_path, "run_0")["attempts"] == 1
    queue.run_worker(_runs(3), tmp_path, tmp_path, session="s2")
    assert calls[-1] == "run_0" and _status(tmp_path, "run_0")["state"] == "completed"


def test_breaker_stops_a_worker_after_consecutive_failures(tmp_path, monkeypatch):
    def broken(cfg, *a, **k):
        raise RuntimeError("CUDA error: device-side assert")

    monkeypatch.setattr(queue, "train_run", broken)
    with pytest.raises(queue.QueueAborted):
        queue.run_worker(_runs(10), tmp_path, tmp_path, session="s1")
    attempted = [r["id"] for r in _runs(10) if (tmp_path / r["id"] / "status.json").exists()]
    assert len(attempted) == queue.MAX_CONSECUTIVE_FAILURES  # the rest of the queue is untouched


def test_run_left_running_by_a_dead_worker_counts_an_attempt_and_resumes(tmp_path, monkeypatch):
    queue.write_status(tmp_path, "run_0", {"id": "run_0", "state": "running", "attempts": 0, "session": "old"})
    monkeypatch.setattr(queue, "train_run", lambda cfg, *a, **k: "completed")
    cfg = queue.claim_next(_runs(1), tmp_path, "new", "0")
    assert cfg["id"] == "run_0"
    st = _status(tmp_path, "run_0")
    assert st["attempts"] == 1 and "worker died" in st["error"] and st["state"] == "running"


def test_run_paused_in_this_session_is_not_reclaimed(tmp_path):
    queue.write_status(tmp_path, "run_0", {"id": "run_0", "state": "paused", "attempts": 0, "session": "s1"})
    assert queue.claim_next(_runs(1), tmp_path, "s1", "0") is None
    assert queue.claim_next(_runs(1), tmp_path, "s2", "0")["id"] == "run_0"


def test_reset_failed_gives_fresh_attempts(tmp_path):
    queue.write_status(tmp_path, "run_0", {"id": "run_0", "state": "failed", "attempts": 3, "session": "s"})
    (tmp_path / "run_0" / "checkpoints").mkdir()
    (tmp_path / "run_0" / "checkpoints" / "last.pt").write_text("x")
    queue.write_status(tmp_path, "run_1", {"id": "run_1", "state": "failed", "attempts": 3, "session": "s"})
    assert queue.reset_failed(_runs(2), tmp_path) == ["run_0", "run_1"]
    assert _status(tmp_path, "run_0")["state"] == "paused" and _status(tmp_path, "run_1")["state"] == "pending"
    assert _status(tmp_path, "run_0")["attempts"] == 0


def _finished_run(results, run_id, stress_selection, keep_best=True):
    run = results / run_id
    (run / "checkpoints").mkdir(parents=True)
    if keep_best:
        (run / "checkpoints" / "best.pt").write_text("weights")
    (run / "metrics_val.json").write_text(json.dumps({"thresholds": {"building": 0.5}}))
    (run / "metrics_test.json").write_text("{}")
    (run / "timing.json").write_text("{}")
    (run / "stress_test.json").write_text(json.dumps({"shift_px": {}, "selection": stress_selection}))
    queue.write_status(results, run_id, {"id": run_id, "state": "completed", "attempts": 1, "session": "s"})


def test_stale_stress_tests_reopen_only_runs_that_kept_their_weights(tmp_path):
    from training.evaluate_run import STRESS_SELECTION

    _finished_run(tmp_path, "run_0", stress_selection=None)  # made under the old tile rule
    _finished_run(tmp_path, "run_1", stress_selection=STRESS_SELECTION)
    _finished_run(tmp_path, "run_2", stress_selection=None, keep_best=False)  # re-opening would retrain
    assert queue.reopen_stale_evaluations(_runs(3), tmp_path) == ["run_0"]
    st = _status(tmp_path, "run_0")
    assert st["state"] == "paused" and st["attempts"] == 1 and st["session"] is None
    assert _status(tmp_path, "run_1")["state"] == "completed" and _status(tmp_path, "run_2")["state"] == "completed"
    assert queue.claim_next(_runs(3), tmp_path, "s2", "0")["id"] == "run_0"


def test_progress_page_never_kills_a_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_write_progress", lambda *a: (_ for _ in ()).throw(FileNotFoundError("race")))
    queue.write_progress(_runs(1), tmp_path)  # logs a warning, does not raise


def test_ema_warmup_and_state_roundtrip():
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.zero_()
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)  # first update uses decay 2/11: mostly the new weights, not the initial ones
    assert torch.allclose(ema.shadow["weight"], torch.full_like(model.weight, 1 - 2 / 11))
    for _ in range(9999):
        ema.update(model)
    assert min(0.999, (1 + ema.updates) / (10 + ema.updates)) == 0.999

    restored = EMA(model, decay=0.999)
    restored.load_state_dict(ema.state_dict())
    assert restored.updates == ema.updates and torch.equal(restored.shadow["weight"], ema.shadow["weight"])
