"""
Tests for the Kaggle orchestrator state machine (with a fake Kaggle).
"""

import json
from pathlib import Path

import pytest
import yaml

import orchestration.kaggle_orchestrator as orch_mod
from orchestration.kaggle_orchestrator import (
    DirResultsStore,
    JOB_PLACEHOLDER,
    Orchestrator,
    normalize_status,
)


class FakeKaggle:
    def __init__(self):
        self.username = "tester"
        self.pushes = []
        self.statuses = {}
        self.outputs = {}  # slug -> {relative path: text}

    def push(self, slug, script, gpu, sources):
        self.pushes.append({"slug": slug, "script": script, "gpu": gpu, "sources": sources})
        self.statuses[slug] = "running"

    def status(self, slug):
        return self.statuses.get(slug, "unknown")

    def download(self, slug, dest):
        for rel, text in self.outputs.get(slug, {}).items():
            path = Path(dest) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


def _job(push):
    """Execute the pushed script's JOB line exactly as Python on Kaggle would."""
    line = next(l for l in push["script"].splitlines() if l.startswith("JOB = "))
    namespace = {}
    exec(line, namespace)
    return namespace["JOB"]


def _status(state, attempts=0):
    return json.dumps({"state": state, "attempts": attempts})


@pytest.fixture
def setup(tmp_path):
    runs_file = tmp_path / "runs.yaml"
    runs_file.write_text(yaml.safe_dump({
        "defaults": {"protocol": "P1", "seed": 0, "queue": "kaggle", "model": "ra_fusion", "epochs": 1},
        "runs": [{"id": "run_a"}, {"id": "run_b"}],
    }))
    kaggle = FakeKaggle()
    store = DirResultsStore(tmp_path / "results")
    orch = Orchestrator(kaggle, store, "https://example.com/repo.git", "a" * 40, runs_file=runs_file)
    return orch, kaggle, store


def _smoke_output(ok=True, fits=True):
    out = {f"results/smoke/smoke_{m}/status.json": _status("completed") for m in ("ra_fusion", "snunet")}
    if not ok:
        out["results/smoke/smoke_snunet/status.json"] = _status("failed", 1)
    out["results/smoke/gpu_probe.json"] = json.dumps(
        {"models": {"ra_fusion": {"fits_configured_batch": True}, "snunet": {"fits_configured_batch": fits}}}
    )
    out["sn8-smoke.log"] = "smoke log"
    return out


def test_normalize_status_variants():
    assert normalize_status('tester/sn8 has status "KernelWorkerStatus.RUNNING"') == "running"
    assert normalize_status("status: complete") == "complete"
    assert normalize_status("KernelWorkerStatus.ERROR") == "error"
    assert normalize_status("cancelAcknowledged") == "cancelled"
    assert normalize_status("queued") == "queued"
    assert normalize_status('u/k has status "KernelWorkerStatus.COMPLETE"\nFailure message: "error in cell"') == "complete"


def test_first_tick_launches_smoke_with_pinned_commit(setup):
    orch, kaggle, store = setup
    st = orch.tick()
    assert st["phase"] == "smoke" and st["kernel"] == "sn8-smoke"
    push = kaggle.pushes[0]
    assert push["gpu"] is True and push["sources"] == []
    assert JOB_PLACEHOLDER not in push["script"]
    job = _job(push)
    assert job["mode"] == "smoke" and job["commit"] == "a" * 40 and job["previous"] is None
    assert (store.root / "STATUS.md").exists() and store.commits


def test_running_session_is_left_alone(setup):
    orch, kaggle, _ = setup
    orch.tick()
    orch.tick()
    assert len(kaggle.pushes) == 1


def test_smoke_pass_moves_to_cpu_prepare(setup):
    orch, kaggle, _ = setup
    orch.tick()
    kaggle.statuses["sn8-smoke"] = "complete"
    kaggle.outputs["sn8-smoke"] = _smoke_output()
    st = orch.tick()
    assert st["phase"] == "prepare"
    assert kaggle.pushes[-1]["slug"] == "sn8-prepare" and kaggle.pushes[-1]["gpu"] is False


def test_smoke_failure_retries_then_stops(setup):
    orch, kaggle, _ = setup
    orch.tick()
    for _ in range(orch_mod.MAX_PHASE_RETRIES + 1):
        kaggle.statuses["sn8-smoke"] = "complete"
        kaggle.outputs["sn8-smoke"] = _smoke_output(ok=False)
        st = orch.tick()
    assert st["phase"] == "stopped" and "not completed" in st["stop_reason"]


def test_model_not_fitting_memory_blocks_training(setup):
    orch, kaggle, _ = setup
    orch.tick()
    kaggle.statuses["sn8-smoke"] = "complete"
    kaggle.outputs["sn8-smoke"] = _smoke_output(fits=False)
    st = orch.tick()
    assert st["phase"] == "smoke" and st["phase_retries"] == 1  # retried, not training


def _to_training(orch, kaggle):
    orch.tick()
    kaggle.statuses["sn8-smoke"] = "complete"
    kaggle.outputs["sn8-smoke"] = _smoke_output()
    orch.tick()
    kaggle.statuses["sn8-prepare"] = "complete"
    return orch.tick()


def test_training_sessions_alternate_and_attach_previous_output(setup):
    orch, kaggle, store = setup
    st = _to_training(orch, kaggle)
    assert st["phase"] == "train" and st["kernel"] == "sn8-train-a"
    assert kaggle.pushes[-1]["sources"] == ["tester/sn8-prepare"]
    assert _job(kaggle.pushes[-1])["previous"] is None

    kaggle.statuses["sn8-train-a"] = "complete"
    kaggle.outputs["sn8-train-a"] = {
        "results/run_a/status.json": _status("completed"),
        "results/run_a/checkpoints/best.pt": "weights",
        "results/run_b/status.json": _status("paused"),
        "results/PROGRESS.md": "progress",
    }
    st = orch.tick()
    assert st["kernel"] == "sn8-train-b"
    assert kaggle.pushes[-1]["sources"] == ["tester/sn8-prepare", "tester/sn8-train-a"]
    assert _job(kaggle.pushes[-1])["previous"] == "sn8-train-a"
    assert (store.root / "run_a" / "status.json").exists()
    assert not (store.root / "run_a" / "checkpoints").exists()  # checkpoints never reach git


def test_quota_exhaustion_waits(setup, monkeypatch):
    orch, kaggle, _ = setup
    _to_training(orch, kaggle)
    kaggle.statuses["sn8-train-a"] = "error"
    kaggle.outputs["sn8-train-a"] = {"sn8-train-a.log": "Your GPU quota has been exceeded"}
    st = orch.tick()
    assert st["wait_until"] is not None and st["kernel"] is None
    pushes = len(kaggle.pushes)
    orch.tick()
    assert len(kaggle.pushes) == pushes  # still waiting


def test_all_runs_finished_marks_done(setup):
    orch, kaggle, _ = setup
    _to_training(orch, kaggle)
    kaggle.statuses["sn8-train-a"] = "complete"
    kaggle.outputs["sn8-train-a"] = {
        "results/run_a/status.json": _status("completed"),
        "results/run_b/status.json": _status("failed", orch_mod.MAX_RUN_ATTEMPTS),
    }
    st = orch.tick()
    assert st["phase"] == "done"
    pushes = len(kaggle.pushes)
    orch.tick()
    assert len(kaggle.pushes) == pushes


def test_failed_session_without_output_keeps_last_good(setup):
    orch, kaggle, _ = setup
    _to_training(orch, kaggle)
    kaggle.statuses["sn8-train-a"] = "complete"
    kaggle.outputs["sn8-train-a"] = {"results/run_a/status.json": _status("paused")}
    orch.tick()  # sn8-train-b resumes from a
    kaggle.statuses["sn8-train-b"] = "error"
    kaggle.outputs["sn8-train-b"] = {"sn8-train-b.log": "Traceback: CUDA error"}
    st = orch.tick()
    # b produced nothing: retried on b, still resuming from a's output
    assert st["last_good_train"] == "sn8-train-a" and st["kernel"] == "sn8-train-b"
    assert _job(kaggle.pushes[-1])["previous"] == "sn8-train-a"


def test_resume_leaves_stopped(setup):
    orch, kaggle, _ = setup
    st = orch.load()
    st.update({"phase": "stopped", "stop_reason": "x"})
    orch.store.open()
    orch.save(st)
    st = orch.resume("train")
    assert st["phase"] == "train" and st["stop_reason"] is None


def test_session_restores_from_the_named_kernel_output(tmp_path, monkeypatch):
    import orchestration.kaggle_session as session

    for source in ("sn8-prepare", "sn8-train-a"):
        (tmp_path / source / "results").mkdir(parents=True)
    (tmp_path / "sn8-prepare" / "processed").mkdir()
    (tmp_path / "sn8-prepare" / "processed" / "manifest.json").write_text("{}")
    monkeypatch.setattr(session, "INPUT", tmp_path)
    assert session.find_input("results", "sn8-train-a") == tmp_path / "sn8-train-a" / "results"
    assert session.find_input("processed/manifest.json", "sn8-prepare").parent.name == "processed"
    assert session.find_input("results", "sn8-train-b") is None
