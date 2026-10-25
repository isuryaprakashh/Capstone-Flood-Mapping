"""
Per-Process Memory Guard.

On Kaggle, exceeding the machine's RAM kills the whole session. The guard
watches this process's resident memory and, past a share of total RAM,
dumps every thread's stack (so the log shows what was allocating) and exits
with MEMORY_EXIT_CODE. The queue supervisor then restarts the worker and the
run counts a failed attempt, while the other GPU keeps training.

Linux only (reads /proc); a no-op elsewhere.
"""

import faulthandler
import logging
import os
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

MEMORY_EXIT_CODE = 4


def _rss_bytes() -> Optional[int]:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def start(fraction: float, interval_s: float = 2.0):
    """Exit this process if its resident memory exceeds `fraction` of total RAM."""
    if _rss_bytes() is None:
        return
    limit = fraction * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

    def watch():
        while True:
            rss = _rss_bytes() or 0
            if rss > limit:
                logger.error(f"Memory guard: {rss / 1e9:.1f} GB resident > {limit / 1e9:.1f} GB; exiting")
                faulthandler.dump_traceback(all_threads=True)
                sys.stderr.flush()
                os._exit(MEMORY_EXIT_CODE)
            time.sleep(interval_s)

    threading.Thread(target=watch, daemon=True, name="memguard").start()
    logger.info(f"Memory guard: this process stops at {limit / 1e9:.1f} GB")
