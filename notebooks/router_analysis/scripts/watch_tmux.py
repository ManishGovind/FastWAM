"""Read-only watcher for the training in tmux session 1 (prints events to stdout)."""
import re
import subprocess
import sys
import time

TARGET = sys.argv[1] if len(sys.argv) > 1 else "1"
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # emit every Nth metric line after the first few
FIRST = int(sys.argv[3]) if len(sys.argv) > 3 else 5  # how many initial metric lines to emit
ERR = re.compile(r"Traceback|Error|error:|\bnan\b|NaN|\binf\b|out of memory|OOM|Killed|SIGTERM|exitcode|Exception")
NEW_REC = re.compile(r"^(\d\d/\d\d \[|\s+(INFO|WARNING|ERROR|DEBUG)\s+\||[A-Za-z(])")
seen, metric_n = set(), 0


def records(text):
    recs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if NEW_REC.match(line) or not recs:
            recs.append(line.rstrip())
        else:
            recs[-1] += " " + line.strip()
    return [re.sub(r"\s{2,}", " ", r) for r in recs]


def pane():
    return subprocess.run(["tmux", "capture-pane", "-p", "-J", "-t", TARGET, "-S", "-400"],
                          capture_output=True, text=True).stdout


first = True
while True:
    text = pane()
    recs = records(text)
    for r in recs:
        key = re.sub(r"^\d\d/\d\d \[\d\d:\d\d:\d\d\]", "", r)
        if key in seen:
            continue
        seen.add(key)
        if first:
            continue  # skip history on the first pass
        # Drop the known-benign video-decoder warnings that get glued onto records.
        r = re.sub(r"\S*video_utils\.py:\d+: UserWarning: torchcodec video decode failed.*?pyav\.( warnings\.warn\()?", "", r)
        r = re.sub(r"\S*_video_deprecation_warning\.py:\d+: UserWarning:.*?(warnings\.warn\(|$)", "", r).strip()
        if not r:
            continue
        if ERR.search(r) and "Registered" not in r:
            print("ERROR? " + r[:600], flush=True)
        elif "loss" in r and ("router" in r or "step" in r.lower()):
            metric_n += 1
            if metric_n <= FIRST or metric_n % EVERY == 0:
                print(f"METRIC#{metric_n} " + r[:700], flush=True)
        elif re.search(r"[Ss]av(ed|ing).*\.pt|checkpoint", r):
            print("CKPT " + r[:300], flush=True)
    first = False
    last = [l for l in text.splitlines() if l.strip()]
    if last and re.search(r"\$\s*$", last[-1]):
        print("ENDED: shell prompt is back in tmux " + TARGET + " | last lines: " + " || ".join(l.strip()[:200] for l in last[-4:-1]), flush=True)
        break
    time.sleep(30)
