"""
Experiment Queue.

Runs every pending experiment of the registry until done or out of session
time. Several workers (one per GPU) can share a results folder: runs are
claimed under a file lock, in this order: paused runs (continue what was
started), then pending runs by priority, then failed runs below the attempt
limit.

Failure handling, so one problem cannot burn through the whole queue:
- a run that failed (or paused) in this session is not picked up again until
  the next session;
- a run left 'running' by a worker that died (crash, kill, timed-out session)
  counts as a failed attempt and is resumed from its last checkpoint;
- a worker stops after MAX_CONSECUTIVE_FAILURES failures in a row
  (QueueAborted), which fails the session so the orchestrator notices.

Each run's status.json records state, attempts, timings and the last error;
results/PROGRESS.md summarizes all runs.
"""

import json
import logging
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import torch
from filelock import FileLock

from training.engine import train_run, write_json

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
MAX_CONSECUTIVE_FAILURES = 3
MIN_START_SECONDS = 1200  # do not start a run with less session time than this


class QueueAborted(RuntimeError):
    """Too many consecutive failures: likely an environment problem, not a run."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _alive(pid: Optional[int]) -> bool:
    if not pid or os.name == "nt":  # Windows (local tests only): os.kill(pid, 0) would send Ctrl+C
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def read_status(results_dir: Path, run_id: str) -> dict:
    path = results_dir / run_id / "status.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"id": run_id, "state": "pending", "attempts": 0}


def write_status(results_dir: Path, run_id: str, status: dict):
    (results_dir / run_id).mkdir(parents=True, exist_ok=True)
    write_json(results_dir / run_id / "status.json", {**status, "updated_at": _now()})


def claim_next(runs: list[dict], results_dir: Path, session: str, worker: str) -> Optional[dict]:
    """Claim the next run to work on (paused > pending by priority > retryable failures)."""
    with FileLock(str(results_dir / ".queue.lock")):
        candidates = []
        for cfg in runs:
            st = read_status(results_dir, cfg["id"])
            state = st["state"]
            same_session = st.get("session") == session
            if state == "running" and (not same_session or not _alive(st.get("pid"))):
                # Its worker died mid-run: count an attempt, resume from the last checkpoint
                st["attempts"] = st.get("attempts", 0) + 1
                st["error"] = f"worker died while running (session {st.get('session')})"
                state = st["state"] = "failed" if st["attempts"] >= MAX_ATTEMPTS else "paused"
                write_status(results_dir, cfg["id"], st)
                same_session = False
            if state == "paused" and not same_session:
                rank = 0
            elif state == "pending":
                rank = 1
            elif state == "failed" and st["attempts"] < MAX_ATTEMPTS and not same_session:
                rank = 2
            else:
                continue  # running elsewhere, finished, out of attempts, or failed/paused this session
            candidates.append((rank, cfg.get("priority", 99), cfg["id"], cfg, st))
        if not candidates:
            return None
        _, _, _, cfg, st = min(candidates, key=lambda c: c[:3])
        st.update({"state": "running", "session": session, "worker": worker, "pid": os.getpid(),
                   "started_at": st.get("started_at") or _now()})
        write_status(results_dir, cfg["id"], st)
        return cfg


def run_worker(
    runs: list[dict],
    processed_dir: Union[str, Path],
    results_dir: Union[str, Path],
    deadline: Optional[float] = None,
    session: Optional[str] = None,
    worker: str = "0",
    max_steps_per_epoch: Optional[int] = None,
    stop_after_epochs: Optional[int] = None,
    max_consecutive_failures: Optional[int] = MAX_CONSECUTIVE_FAILURES,
) -> dict[str, str]:
    """
    Work through the queue until no runs remain or session time runs out.

    Returns:
        Mapping run id → state reached in this call.

    Raises:
        QueueAborted: after max_consecutive_failures failures in a row.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    session = session or os.environ.get("QUEUE_SESSION") or uuid.uuid4().hex[:8]
    outcome = {}
    consecutive_failures = 0

    while True:
        if deadline is not None and deadline - time.time() < MIN_START_SECONDS:
            logger.info(f"[worker {worker}] less than {MIN_START_SECONDS}s left; stopping")
            break
        cfg = claim_next(runs, results_dir, session, worker)
        if cfg is None:
            break
        run_id = cfg["id"]
        logger.info(f"[worker {worker}] starting {run_id}")
        st = read_status(results_dir, run_id)
        t0 = time.time()
        try:
            state = train_run(
                cfg, processed_dir, results_dir / run_id, deadline=deadline,
                max_steps_per_epoch=max_steps_per_epoch, stop_after_epochs=stop_after_epochs,
            )
            st.update({"state": state, "error": None})
            if state == "completed":
                st["completed_at"] = _now()
            consecutive_failures = 0
        except Exception as e:
            st["attempts"] = st.get("attempts", 0) + 1
            st.update({"state": "failed", "error": f"{type(e).__name__}: {e}"})
            with open(results_dir / run_id / "error.log", "a", encoding="utf-8") as f:
                f.write(f"[{_now()}] attempt {st['attempts']}\n{traceback.format_exc()}\n")
            logger.error(f"[worker {worker}] {run_id} failed (attempt {st['attempts']}): {e}")
            consecutive_failures += 1
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass  # a broken CUDA context: the next claim fails too and the breaker stops the worker
        st["seconds_total"] = round(st.get("seconds_total", 0.0) + time.time() - t0, 1)
        write_status(results_dir, run_id, st)
        write_progress(runs, results_dir)
        outcome[run_id] = st["state"]
        if stop_after_epochs and st["state"] == "paused":
            break  # resume test: hand control back after the requested pause
        if max_consecutive_failures and consecutive_failures >= max_consecutive_failures:
            raise QueueAborted(f"worker {worker}: {consecutive_failures} runs failed in a row; last: {st['error']}")
    return outcome


def reset_failed(runs: list[dict], results_dir: Union[str, Path]) -> list[str]:
    """Give failed runs a fresh set of attempts (after the cause was fixed); returns their ids."""
    results_dir = Path(results_dir)
    reset = []
    for cfg in runs:
        st = read_status(results_dir, cfg["id"])
        if st["state"] == "failed":
            resumable = (results_dir / cfg["id"] / "checkpoints" / "last.pt").exists()
            st.update({"state": "paused" if resumable else "pending", "attempts": 0, "session": None})
            write_status(results_dir, cfg["id"], st)
            reset.append(cfg["id"])
    return reset


def write_progress(runs: list[dict], results_dir: Union[str, Path]):
    """Write results/PROGRESS.md (never raises: a progress page must not kill a worker)."""
    try:
        _write_progress(runs, Path(results_dir))
    except Exception as e:
        logger.warning(f"Could not write PROGRESS.md: {e}")


def _write_progress(runs: list[dict], results_dir: Path):
    """One row per run with state and headline metrics."""
    rows, done = [], 0
    for cfg in sorted(runs, key=lambda c: (c.get("priority", 99), c["id"])):
        st = read_status(results_dir, cfg["id"])
        done += st["state"] == "completed"
        headline = ""
        metrics_path = results_dir / cfg["id"] / "metrics_test.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text()).get("summary", {})
            headline = " · ".join(f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, (int, float)))
        rows.append(
            f"| {cfg['id']} | {cfg['model']} | {cfg['protocol']} | {cfg['seed']} | {st['state']} | "
            f"{st.get('attempts', 0)} | {headline} | {(st.get('error') or '')[:80]} |"
        )
    text = [
        "# Training progress",
        "",
        f"{done}/{len(runs)} runs completed · updated {_now()}",
        "",
        "| Run | Model | Protocol | Seed | State | Failed attempts | Test metrics | Last error |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    tmp = results_dir / f"PROGRESS.md.{os.getpid()}.tmp"  # per process: two workers write it
    tmp.write_text("\n".join(text), encoding="utf-8")
    os.replace(tmp, results_dir / "PROGRESS.md")
