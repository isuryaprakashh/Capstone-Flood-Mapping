#!/usr/bin/env python3
"""
Kaggle Training Orchestrator (runs on GitHub Actions every 30 minutes).

A small state machine persisted in orchestrator.json on the `results` branch:

    smoke ──► prepare ──► train ──(all runs finished)──► done
      │          │          │
      └──────────┴──────────┴──► stopped (needs attention; the workflow fails so GitHub emails)

Each tick: if a Kaggle session is running, do nothing. If it finished,
download its results (never checkpoints, which stay on Kaggle), commit them
to the `results` branch, check the outcome and advance. Then start the next
session if one is needed:

- smoke   (GPU): 12-tile dataset, every model trained briefly with a forced
          stop/resume, plus a GPU memory/speed probe. Training only starts
          if every smoke run completed and every model fits its batch size.
- prepare (CPU): full dataset preparation; its output is attached to every
          training session.
- train   (GPU T4 x2): the run queue for ~11.5 h; the next session attaches
          the previous one's output (results + checkpoints) and resumes.

The code commit is pinned when the smoke test is launched so every run uses
identical code; `--update-code` re-pins it to the current main branch.

Usage (GitHub Actions):
    python orchestration/kaggle_orchestrator.py tick
    python orchestration/kaggle_orchestrator.py tick --update-code
    python orchestration/kaggle_orchestrator.py resume --phase train
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_TEMPLATE = REPO_ROOT / "orchestration" / "kaggle_session.py"
JOB_PLACEHOLDER = "JOB = {}  # __JOB__ (replaced by the orchestrator)"

SLUGS = {"smoke": "sn8-smoke", "prepare": "sn8-prepare", "train": ("sn8-train-a", "sn8-train-b")}
SESSION_HOURS = {"smoke": 3.0, "prepare": 2.0, "train": 11.5}
KAGGLE_TIMEOUT_S = 12 * 3600 - 120
STUCK_AFTER_S = 13 * 3600
QUOTA_WAIT_S = 6 * 3600
MAX_PHASE_RETRIES = 2
MAX_RUN_ATTEMPTS = 3
TICK_ERRORS_BEFORE_ALERT = 3  # consecutive failing ticks (1.5 h) before one alert email
WEEKLY_GPU_HOURS = 30.0  # Kaggle's weekly GPU quota, reset Saturday 00:00 UTC
QUOTA_NEAR_H = 5.0  # an early failure with less than this left is treated as quota, not a bug
PIP_PACKAGES = ["rasterio>=1.3.8", "timm>=1.0.20", "filelock>=3.12", "pyyaml>=6.0"]
DOWNLOAD_PATTERN = r"^(?!processed/)(?!.*checkpoints/).*$"


def now() -> float:
    return time.time()


def stamp(t: float = None) -> str:
    return datetime.fromtimestamp(t or now(), timezone.utc).isoformat(timespec="seconds")


def quota_week_start(t: float) -> float:
    """Most recent Saturday 00:00 UTC, when Kaggle resets the weekly GPU quota."""
    day = datetime.fromtimestamp(t, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return (day - timedelta(days=(day.weekday() - 5) % 7)).timestamp()


# ─── Kaggle ─────────────────────────────────────────────────────────────────


def normalize_status(text: str) -> str:
    """Map `kaggle kernels status` output to queued/running/complete/error/cancelled/unknown."""
    m = re.search(r'has status "(?:KernelWorkerStatus\.)?(\w+)"', text)
    t = (m.group(1) if m else text).lower()
    for key, names in (
        ("cancelled", ("cancel",)),
        ("error", ("error", "fail")),
        ("complete", ("complete",)),
        ("running", ("running",)),
        ("queued", ("queued", "pending")),
    ):
        if any(n in t for n in names):
            return key
    return "unknown"


class KaggleCLI:
    """Thin wrapper around the official kaggle CLI (auth via KAGGLE_API_TOKEN)."""

    def __init__(self, username: str):
        self.username = username

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        cmd = ["kaggle", *args]
        logger.info("$ " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.stdout.strip():
            logger.info(result.stdout.strip()[-2000:])
        if check and result.returncode != 0:
            raise RuntimeError(f"kaggle {' '.join(args)} failed: {result.stderr.strip()[-2000:]}")
        return result

    def push(self, slug: str, script: str, gpu: bool, sources: list[str]):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "session.py").write_text(script)
            meta = {
                "id": f"{self.username}/{slug}",
                "title": slug,
                "code_file": "session.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": gpu,
                "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4" if gpu else "",
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": sources,
                "model_sources": [],
            }
            Path(tmp, "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
            args = ["kernels", "push", "-p", tmp, "--timeout", str(KAGGLE_TIMEOUT_S)]
            if gpu:
                args += ["--accelerator", "NvidiaTeslaT4"]
            self._run(*args)

    def quota(self) -> tuple[float, float] | None:
        """(GPU hours left this week, next reset time) from `kaggle quota`; None if unavailable."""
        result = self._run("quota", "--format", "json", check=False)
        try:
            gpu = next(r for r in json.loads(result.stdout) if r.get("resource") == "GPU")
            left = float(str(gpu["remaining"]).rstrip("h"))
            if float(str(gpu["total"]).rstrip("h")) <= 0:
                return None  # no allowance reported: don't block on it
        except Exception:
            return None
        try:
            reset = datetime.fromisoformat(gpu.get("refreshAt") or gpu.get("refresh_at")).timestamp()
        except Exception:
            reset = quota_week_start(now()) + 7 * 86400
        return left, reset

    def status(self, slug: str) -> str:
        result = self._run("kernels", "status", f"{self.username}/{slug}", check=False)
        return normalize_status(result.stdout + result.stderr)

    def download(self, slug: str, dest: Path, pattern: str = DOWNLOAD_PATTERN):
        dest.mkdir(parents=True, exist_ok=True)
        self._run("kernels", "output", f"{self.username}/{slug}", "-p", str(dest), "-o", "--file-pattern", pattern,
                  "--page-size", "200")


# ─── Results branch ─────────────────────────────────────────────────────────


class GitResultsStore:
    """
    The `results` branch checked out as a git worktree; commits use the repo
    owner's identity. Everything lives under experiments/results/ so the
    branch merges into main without touching other files.
    """

    def __init__(self, repo: Path = REPO_ROOT, branch: str = "results"):
        self.repo = repo
        self.branch = branch
        self.worktree = repo / ".results_worktree"
        self.root = self.worktree / "experiments" / "results"
        self.name = os.environ.get("RESULTS_GIT_NAME", "isuryaprakashh")
        self.email = os.environ.get("RESULTS_GIT_EMAIL", "isurya2005@gmail.com")

    def _git(self, *args: str, cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=cwd or self.repo, capture_output=True, text=True, check=check)

    def open(self):
        if self.worktree.exists():
            return
        exists = self._git("ls-remote", "--heads", "origin", self.branch).stdout.strip()
        if exists:
            self._git("fetch", "origin", f"{self.branch}:{self.branch}")
            self._git("worktree", "add", str(self.worktree), self.branch)
        else:
            self._git("worktree", "add", "--detach", str(self.worktree))
            self._git("checkout", "--orphan", self.branch, cwd=self.worktree)
            self._git("rm", "-rf", "--quiet", ".", cwd=self.worktree, check=False)
        self.root.mkdir(parents=True, exist_ok=True)
        readme = self.root / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Training results\n\nWritten automatically by the Kaggle orchestrator. "
                "See STATUS.md and PROGRESS.md.\n"
            )

    def commit(self, message: str):
        self._git("add", "-A", cwd=self.worktree)
        if not self._git("status", "--porcelain", cwd=self.worktree).stdout.strip():
            return
        self._git("-c", f"user.name={self.name}", "-c", f"user.email={self.email}",
                  "commit", "-q", "-m", message, cwd=self.worktree)
        self._git("push", "-q", "origin", f"HEAD:{self.branch}", cwd=self.worktree)


class DirResultsStore:
    """Plain directory store (tests and dry runs)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.commits = []

    def open(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(self, message: str):
        self.commits.append(message)


# ─── State machine ──────────────────────────────────────────────────────────


DEFAULT_STATE = {
    "phase": "smoke",
    "code_commit": None,
    "session": 0,
    "kernel": None,
    "kernel_mode": None,
    "pushed_at": None,
    "wait_until": None,
    "last_good_train": None,
    "phase_retries": 0,
    "stop_reason": None,
    "gpu_sessions": [],  # {"start", "hours"} per GPU session, for the weekly quota
    "tick_errors": 0,
    "events": [],
}


class Orchestrator:
    def __init__(self, kaggle, store, repo_url: str, current_commit: str, runs_file: Path = None):
        self.kaggle = kaggle
        self.store = store
        self.repo_url = repo_url
        self.current_commit = current_commit
        self.runs_file = runs_file or REPO_ROOT / "experiments" / "runs.yaml"

    # persistence
    @property
    def state_path(self) -> Path:
        return self.store.root / "orchestrator.json"

    def load(self) -> dict:
        if self.state_path.exists():
            return {**DEFAULT_STATE, **json.loads(self.state_path.read_text())}
        return json.loads(json.dumps(DEFAULT_STATE))

    def save(self, st: dict):
        st["events"] = st["events"][-200:]
        self.state_path.write_text(json.dumps(st, indent=2))
        self.write_status(st)

    def event(self, st: dict, message: str):
        logger.info(message)
        st["events"].append({"at": stamp(), "message": message})

    # main entry
    def tick(self, update_code: bool = False) -> dict:
        self.store.open()
        st = self.load()
        if update_code:
            st["code_commit"] = self.current_commit
            self.event(st, f"Code re-pinned to {self.current_commit[:7]}")

        if st["phase"] in ("done", "stopped"):
            logger.info(f"Phase {st['phase']}: nothing to do ({st.get('stop_reason') or 'finished'})")
            self.save(st)
            self.store.commit(f"orchestrator: {st['phase']}")
            return st

        if st["kernel"]:
            status = self.kaggle.status(st["kernel"])
            age = now() - (st["pushed_at"] or now())
            if status in ("queued", "running", "unknown") and age < STUCK_AFTER_S:
                logger.info(f"{st['kernel']} is {status} ({age / 3600:.1f} h)")
                self.save(st)
                return st
            self.finish(st, status if age < STUCK_AFTER_S else "error")
            self.save(st)  # a failed launch below must not redo this session's bookkeeping

        if st["phase"] in ("smoke", "prepare", "train") and not st["kernel"]:
            if st["wait_until"] and now() < st["wait_until"]:
                logger.info(f"Waiting for GPU quota until {stamp(st['wait_until'])}")
            elif st["phase"] != "prepare" and self.gpu_quota(st)[0] <= 0:
                # Kaggle refuses new GPU sessions once the quota is spent (a running one may finish)
                self.quota_wait(st, "Weekly GPU quota used up")
            else:
                st["wait_until"] = None
                self.launch(st)

        self.save(st)
        self.store.commit(f"orchestrator: {st['phase']} (session {st['session']})")
        return st

    def safe_tick(self, update_code: bool = False) -> tuple[dict, str | None]:
        """
        tick() that survives transient Kaggle/GitHub errors (the next tick
        retries). Returns (state, alert); the alert is set only on the tick
        that stops the pipeline or on the TICK_ERRORS_BEFORE_ALERT-th
        consecutive error, so a problem sends one email, not one per tick.
        """
        self.store.open()
        was_stopped = self.load()["phase"] == "stopped"
        try:
            st = self.tick(update_code)
        except Exception as e:
            st = self.load()  # last saved state
            alert = None
            if re.search(r"quota", str(e), re.I) and not st["kernel"]:
                self.quota_wait(st, "Kaggle refused the session (quota)")
            else:
                st["tick_errors"] += 1
                self.event(st, f"Tick error {st['tick_errors']}: {type(e).__name__}: {str(e)[-300:]}")
                if st["tick_errors"] == TICK_ERRORS_BEFORE_ALERT:
                    alert = f"{st['tick_errors']} consecutive orchestrator errors, latest: {e}"
            self.save(st)
            self.store.commit("orchestrator: tick error")
            return st, alert

        if st["tick_errors"]:
            st["tick_errors"] = 0
            self.event(st, "Recovered from tick errors")
            self.save(st)
            self.store.commit("orchestrator: recovered")
        if st["phase"] == "stopped" and not was_stopped:
            return st, f"Orchestrator stopped: {st['stop_reason']}"
        return st, None

    def resume(self, phase: str) -> dict:
        """Manually leave 'stopped' (after fixing the cause) and continue at `phase`."""
        self.store.open()
        st = self.load()
        st.update({"phase": phase, "stop_reason": None, "phase_retries": 0, "kernel": None, "wait_until": None})
        self.event(st, f"Manually resumed at phase '{phase}'")
        self.save(st)
        self.store.commit(f"orchestrator: resume {phase}")
        return st

    # session lifecycle
    def launch(self, st: dict):
        mode = st["phase"]
        if mode == "smoke" or not st["code_commit"]:
            st["code_commit"] = self.current_commit
        job = {
            "mode": mode,
            "repo": self.repo_url,
            "commit": st["code_commit"],
            "hours": SESSION_HOURS[mode],
            "queue": "kaggle",
            "pip": PIP_PACKAGES,
            "session": st["session"] + 1,
            "data_source": SLUGS["prepare"],
            "previous": st["last_good_train"] if mode == "train" else None,
        }
        sources = []
        if mode == "train":
            sources.append(f"{self.kaggle.username}/{SLUGS['prepare']}")
            a, b = SLUGS["train"]
            slug = b if st["last_good_train"] == a else a
            if st["last_good_train"]:
                sources.append(f"{self.kaggle.username}/{st['last_good_train']}")
        else:
            slug = SLUGS[mode]

        script = SESSION_TEMPLATE.read_text()
        if JOB_PLACEHOLDER not in script:
            raise RuntimeError("Session template is missing the JOB placeholder")
        script = script.replace(JOB_PLACEHOLDER, "JOB = " + repr(job))  # a Python literal (None, not null)
        self.kaggle.push(slug, script, gpu=mode != "prepare", sources=sources)
        st.update({"kernel": slug, "kernel_mode": mode, "pushed_at": now(), "session": st["session"] + 1})
        self.event(st, f"Started {mode} session {st['session']} on {slug} (code {st['code_commit'][:7]})")

    def finish(self, st: dict, status: str):
        slug, mode = st["kernel"], st["kernel_mode"]
        minutes = (now() - (st["pushed_at"] or now())) / 60
        self.event(st, f"{slug} ({mode}) ended: {status} after {minutes:.0f} min")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            try:
                self.kaggle.download(slug, tmp)
            except Exception as e:
                self.event(st, f"Could not download output of {slug}: {e}")
            log_text = self.collect(tmp, st["session"], mode)
            produced_results = (tmp / "results").exists()
            session_info = tmp / "session.json"
            run_minutes = json.loads(session_info.read_text())["minutes"] if session_info.exists() else minutes

        st["kernel"] = None
        if mode != "prepare":
            st["gpu_sessions"] = (st["gpu_sessions"] + [{"start": st["pushed_at"], "hours": run_minutes / 60}])[-50:]
        if mode == "smoke":
            ok, reason = self.smoke_passed()
            if ok:
                st.update({"phase": "prepare", "phase_retries": 0})
                self.event(st, "Smoke test passed; preparing the full dataset")
            else:
                self.retry_or_stop(st, f"Smoke test failed: {reason}")
        elif mode == "prepare":
            if status == "complete":
                st.update({"phase": "train", "phase_retries": 0})
                self.event(st, "Dataset prepared; training starts")
            else:
                self.retry_or_stop(st, f"Data preparation ended with status {status}")
        elif mode == "train":
            if produced_results:
                # Resume from the newest output with results, even if the session ended in an error
                st["last_good_train"] = slug
            quota_low = self.gpu_quota(st)[0] < QUOTA_NEAR_H
            if status != "complete" and minutes < 15 and (quota_low or re.search(r"quota|limit", log_text, re.I)):
                self.quota_wait(st, f"Session {slug} refused or cut short, GPU quota appears exhausted")
            elif status != "complete" and minutes < 15:
                self.retry_or_stop(st, f"Training session failed early with status {status}")
            if self.all_runs_finished():
                st["phase"] = "done"
                self.event(st, "All runs finished")

    def gpu_quota(self, st: dict) -> tuple[float, float]:
        """(GPU hours left this week, next reset): Kaggle's figures, else our own session tally."""
        reported = self.kaggle.quota() if hasattr(self.kaggle, "quota") else None
        if reported is not None:
            return reported
        start = quota_week_start(now())
        used = sum(s["hours"] for s in st["gpu_sessions"] if s["start"] and s["start"] >= start)
        return WEEKLY_GPU_HOURS - used, start + 7 * 86400

    def quota_wait(self, st: dict, reason: str):
        """Wait for the weekly reset when the quota is (nearly) spent, else retry in 6 h."""
        left, reset = self.gpu_quota(st)
        st["wait_until"] = reset + 600 if left < QUOTA_NEAR_H else now() + QUOTA_WAIT_S
        self.event(st, f"{reason} ({max(left, 0):.1f} GPU h left); waiting until {stamp(st['wait_until'])}")

    def retry_or_stop(self, st: dict, reason: str):
        st["phase_retries"] += 1
        if st["phase_retries"] > MAX_PHASE_RETRIES:
            st.update({"phase": "stopped", "stop_reason": reason})
            self.event(st, f"Stopped: {reason}")
        else:
            self.event(st, f"{reason} — retry {st['phase_retries']}/{MAX_PHASE_RETRIES}")

    def collect(self, downloaded: Path, session: int, mode: str) -> str:
        """Copy downloaded results into the results branch; return the session log text."""
        results = downloaded / "results"
        if mode == "smoke":
            # Judge each smoke session on its own output, never on an earlier attempt's files
            shutil.rmtree(self.store.root / "smoke", ignore_errors=True)
        if results.exists():
            shutil.copytree(results, self.store.root, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("checkpoints", "*.pt", ".queue.lock"))
            if mode == "train":
                # RESULTS.md / RESULTS.csv over the completed runs (no-op until one completes)
                subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "aggregate_results.py"),
                                "--results", str(self.store.root)], capture_output=True)
        sessions = self.store.root / "sessions"
        sessions.mkdir(exist_ok=True)
        log_text = ""
        for f in downloaded.glob("*.log"):
            log_text += f.read_text(errors="ignore")
            shutil.copy(f, sessions / f"session_{session:03d}_{mode}.log")
        if (downloaded / "session.json").exists():
            shutil.copy(downloaded / "session.json", sessions / f"session_{session:03d}_{mode}.json")
        return log_text

    # checks
    def smoke_passed(self) -> tuple[bool, str]:
        smoke = self.store.root / "smoke"
        statuses = {p.parent.name: json.loads(p.read_text()) for p in smoke.glob("*/status.json")}
        if not statuses:
            return False, "no smoke results"
        bad = [k for k, s in statuses.items() if s.get("state") != "completed"]
        if bad:
            return False, f"runs not completed: {bad}"
        probe_path = smoke / "gpu_probe.json"
        if not probe_path.exists():
            return False, "GPU probe missing"
        probe = json.loads(probe_path.read_text())["models"]
        too_big = [k for k, v in probe.items() if not v.get("fits_configured_batch")]
        if too_big:
            return False, f"models do not fit their batch size: {too_big} (see smoke/gpu_probe.json)"
        return True, ""

    def all_runs_finished(self) -> bool:
        sys.path.insert(0, str(REPO_ROOT))
        from training.config import load_runs

        runs = [r for r in load_runs(self.runs_file).values() if r.get("queue") == "kaggle"]
        for cfg in runs:
            path = self.store.root / cfg["id"] / "status.json"
            if not path.exists():
                return False
            s = json.loads(path.read_text())
            if s["state"] != "completed" and not (s["state"] == "failed" and s.get("attempts", 0) >= MAX_RUN_ATTEMPTS):
                return False
        return True

    def write_status(self, st: dict):
        lines = [
            "# Orchestrator status",
            "",
            f"- **Phase:** {st['phase']}" + (f" — {st['stop_reason']}" if st.get("stop_reason") else ""),
            f"- **Sessions started:** {st['session']}",
            f"- **Running:** {st['kernel'] or '—'}",
            f"- **Code commit:** {st['code_commit'] or '—'}",
            f"- **Waiting for quota until:** {stamp(st['wait_until']) if st['wait_until'] else '—'}",
            f"- **Updated:** {stamp()}",
            "",
            "Per-run progress: [PROGRESS.md](PROGRESS.md)",
            "",
            "## Recent events",
            "",
            *[f"- {e['at']} — {e['message']}" for e in reversed(st["events"][-30:])],
            "",
        ]
        (self.store.root / "STATUS.md").write_text("\n".join(lines))


# ─── GitHub helpers ─────────────────────────────────────────────────────────


def disable_workflow(workflow_file: str = "train.yml"):
    """Disable this workflow via the GitHub API once training is done."""
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not token or not repo:
        return
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow_file}/disable",
        method="PUT",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    try:
        urllib.request.urlopen(req, timeout=30)
        logger.info("Workflow disabled")
    except Exception as e:
        logger.warning(f"Could not disable workflow: {e}")


def main():
    parser = argparse.ArgumentParser(description="Kaggle training orchestrator")
    parser.add_argument("command", choices=["tick", "resume", "status"])
    parser.add_argument("--phase", choices=["smoke", "prepare", "train"], help="For 'resume'")
    parser.add_argument("--update-code", action="store_true", help="Re-pin the code commit to current main")
    args = parser.parse_args()

    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        sys.exit("Set the KAGGLE_USERNAME repository variable (Settings → Secrets and variables → Actions → Variables)")
    repo = os.environ.get("GITHUB_REPOSITORY", "isuryaprakashh/Capstone-Flood-Mapping")
    commit = os.environ.get("GITHUB_SHA") or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()

    orch = Orchestrator(KaggleCLI(username), GitResultsStore(), f"https://github.com/{repo}.git", commit)
    if args.command == "resume":
        if not args.phase:
            sys.exit("resume needs --phase")
        st = orch.resume(args.phase)
    elif args.command == "status":
        orch.store.open()
        st = orch.load()
        print(json.dumps({k: v for k, v in st.items() if k != "events"}, indent=2))
        return
    else:
        st, alert = orch.safe_tick(update_code=args.update_code)
        if alert:
            sys.exit(alert)  # failing run → one GitHub email

    if st["phase"] == "done":
        disable_workflow()


if __name__ == "__main__":
    main()
