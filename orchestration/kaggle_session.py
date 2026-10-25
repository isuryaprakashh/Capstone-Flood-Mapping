"""
Kaggle session entry point.

orchestration/kaggle_orchestrator.py pushes this script as a Kaggle kernel
after replacing the JOB placeholder below with the session's job. A session:

  1. checks the GPU and installs the few packages Kaggle's image lacks,
  2. clones this repository at the pinned commit,
  3. smoke / prepare: downloads SpaceNet 8 from its public AWS bucket and
     prepares it (a small subset for the smoke test),
     train: uses the prepared data attached from the prepare kernel,
  4. restores results and checkpoints from the previous session,
  5. runs the job (smoke test + GPU probe, or the training queue) until done
     or the session budget is spent,
  6. leaves results/ (and processed/ for prepare) in /kaggle/working, which
     Kaggle keeps as this session's output.
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

JOB = {}  # __JOB__ (replaced by the orchestrator)

START = time.time()
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp")
INPUT = Path("/kaggle/input")
S3 = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs"
AOIS = ("Germany_Training_Public", "Louisiana-East_Training_Public")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - START) / 60:.0f}m] {msg}", flush=True)


def sh(cmd: str, cwd: Path = None):
    log(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def find_input(relative: str, source: str) -> Path | None:
    """Locate a file/folder in the attached output of kernel `source` (/kaggle/input/.../<source>/...)."""
    if not INPUT.exists():
        return None
    matches = [p for depth in ("*", "*/*", "*/*/*") for p in sorted(INPUT.glob(f"{depth}/{relative}"))]
    matches = [p for p in matches if source in p.relative_to(INPUT).parts]
    return matches[0] if matches else None


def gpu_count() -> int:
    import torch

    n = torch.cuda.device_count()
    log(f"GPUs: {n} {[torch.cuda.get_device_name(i) for i in range(n)]}")
    return n


def download_dataset(raw: Path):
    """Download and extract the labelled AOIs (archives have no top-level folder)."""
    for aoi in AOIS:
        dest = raw / aoi
        if (dest / "PRE-event").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        archive = TMP / f"{aoi}.tar.gz"
        log(f"Downloading {aoi}")
        urllib.request.urlretrieve(f"{S3}/{aoi}.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(dest)
        archive.unlink()


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    mode = JOB["mode"]
    deadline = START + JOB["hours"] * 3600
    log(f"Job: {json.dumps(JOB)}")
    log(f"CPUs: {os.cpu_count()}; shared memory (DataLoader workers): {shutil.disk_usage('/dev/shm').total / 1e9:.1f} GB"
        if os.path.exists("/dev/shm") else f"CPUs: {os.cpu_count()}")

    if mode != "prepare" and gpu_count() == 0:
        sys.exit("No GPU attached: refusing to train on CPU")

    sh(f"{sys.executable} -m pip install -q " + " ".join(f'"{p}"' for p in JOB["pip"]))

    code = TMP / "code"
    if not code.exists():
        sh(f"git clone --quiet {JOB['repo']} {code}")
    sh(f"git checkout --quiet {JOB['commit']}", cwd=code)
    os.environ["CODE_GIT_COMMIT"] = JOB["commit"]
    py = sys.executable

    results = WORK / "results"
    if JOB.get("previous") and not results.exists():
        previous = find_input("results", JOB["previous"])
        if previous is None:
            sys.exit(f"Previous session output {JOB['previous']} is not attached")
        log(f"Restoring results and checkpoints from {previous}")
        shutil.copytree(previous, results, copy_function=shutil.copyfile)
        for path in [results, *results.rglob("*")]:  # inputs are read-only
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    results.mkdir(parents=True, exist_ok=True)

    if mode == "smoke":
        raw = TMP / "raw"
        download_dataset(raw)
        processed = TMP / "processed_smoke"
        sh(f"{py} scripts/prepare_data.py --data-root {raw} --out-dir {processed} --workers 4 --limit 12", cwd=code)
        sh(f"{py} scripts/run_queue.py --processed {processed} --results {results / 'smoke'} --smoke", cwd=code)
        sh(f"{py} scripts/gpu_probe.py --output {results / 'smoke' / 'gpu_probe.json'}", cwd=code)

    elif mode == "prepare":
        raw = TMP / "raw"
        download_dataset(raw)
        sh(f"{py} scripts/prepare_data.py --data-root {raw} --out-dir {WORK / 'processed'} --workers 4", cwd=code)

    elif mode == "train":
        manifest = find_input("processed/manifest.json", JOB["data_source"])
        if manifest is None:
            sys.exit("Prepared dataset not attached (expected <input>/processed/manifest.json)")
        sh(
            f"{py} scripts/run_queue.py --processed {manifest.parent} --results {results} "
            f"--queue {JOB['queue']} --gpus {max(1, gpu_count())} --deadline {deadline}"
            + (" --reset-failed" if JOB.get("reset_failed") else ""),
            cwd=code,
        )
    else:
        sys.exit(f"Unknown mode {mode}")

    (WORK / "session.json").write_text(json.dumps({
        **JOB, "started": START, "finished": time.time(), "minutes": round((time.time() - START) / 60, 1),
    }, indent=2))
    log("Session finished")


if __name__ == "__main__":
    main()
