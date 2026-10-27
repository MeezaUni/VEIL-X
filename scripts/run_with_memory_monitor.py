"""Run one command and record its memory use (whole process tree) and system free memory.

Safety: if system available memory drops below ``--min-free-gb`` the monitored command's OWN
process tree is terminated (never any other process), and the exit code reports it. Results go
to ``--log`` as JSON (peak RSS of the tree, minimum available system memory, duration).

    .venv-gpu\\Scripts\\python.exe scripts\\run_with_memory_monitor.py --log m.json -- <cmd ...>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil


def tree_rss(proc: psutil.Process) -> int:
    """RSS of the process and its children; 0 if the tree has already exited (race-safe)."""
    try:
        procs = [proc, *proc.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    total = 0
    for p in procs:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            total += p.memory_info().rss
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    child = subprocess.Popen(cmd)  # command supplied by the operator
    proc = psutil.Process(child.pid)
    peak, min_avail, aborted = 0, float("inf"), False
    t0 = time.perf_counter()
    while child.poll() is None:
        peak = max(peak, tree_rss(proc))
        avail = psutil.virtual_memory().available
        min_avail = min(min_avail, avail)
        if avail < args.min_free_gb * 2**30:
            aborted = True
            victims = [proc]
            with contextlib.suppress(psutil.NoSuchProcess):
                victims = [*proc.children(recursive=True), proc]  # only the monitored tree
            for p in victims:
                with contextlib.suppress(psutil.NoSuchProcess):
                    p.terminate()
            break
        time.sleep(args.interval)
    rc = child.wait()
    rec = {
        "cmd": [Path(c).name if Path(c).is_absolute() else c for c in cmd],
        "exit_code": rc,
        "aborted_low_memory": aborted,
        "peak_tree_rss_gb": round(peak / 2**30, 2),
        "min_system_available_gb": round(min_avail / 2**30, 2),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    args.log.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec), flush=True)
    return 2 if aborted else rc


if __name__ == "__main__":
    sys.exit(main())
