"""
Tests for the stress-test tile subset and the engine's re-evaluation of
finished runs (only best.pt kept) without retraining them.
"""

import json

import pytest
import torch

import training.engine as engine
from training.evaluate_run import STRESS_SELECTION, evaluation_complete, stress_tile_ids


def test_stress_subset_puts_flooded_tiles_first():
    tiles = {f"t{i}": {"has_flood": i % 3 == 0} for i in range(9)}
    assert stress_tile_ids(list(reversed(list(tiles))), tiles, 5) == ["t0", "t3", "t6", "t1", "t2"]


def _finished_run(run_dir, stress_selection):
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best.pt").write_text("weights")
    (run_dir / "metrics_val.json").write_text(json.dumps({"thresholds": {"building": 0.5}}))
    (run_dir / "metrics_test.json").write_text("{}")
    (run_dir / "timing.json").write_text("{}")
    (run_dir / "stress_test.json").write_text(json.dumps({"shift_px": {}, "selection": stress_selection}))


@pytest.fixture
def finished(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "manifest.json").write_text(json.dumps({"tiles": []}))
    (processed / "splits.json").write_text(json.dumps({"P1": {"train": [], "val": [], "test": []}}))
    calls = []
    monkeypatch.setattr(engine, "build_research_model", lambda name, **kw: torch.nn.Linear(1, 1))
    monkeypatch.setattr(engine, "_final_evaluation", lambda *a, **k: calls.append("eval") or "completed")
    monkeypatch.setattr(engine, "SN8CenterCropDataset", lambda *a, **k: pytest.fail("finished run was retrained"))
    cfg = {"id": "run", "model": "m", "protocol": "P1", "seed": 0}
    return cfg, processed, tmp_path / "run", calls


def test_finished_run_with_a_stale_stress_test_is_only_re_evaluated(finished):
    cfg, processed, run_dir, calls = finished
    _finished_run(run_dir, stress_selection=None)
    assert not evaluation_complete(run_dir)
    assert engine.train_run(cfg, processed, run_dir, device=torch.device("cpu")) == "completed"
    assert calls == ["eval"]
    assert not (run_dir / "environment.json").exists()  # the record of how it was trained is kept


def test_finished_run_with_a_current_evaluation_is_left_alone(finished):
    cfg, processed, run_dir, calls = finished
    _finished_run(run_dir, stress_selection=STRESS_SELECTION)
    assert engine.train_run(cfg, processed, run_dir, device=torch.device("cpu")) == "completed"
    assert calls == []
