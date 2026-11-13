"""Bounded process-tree and run-labelled Docker cleanup."""
import os
import re
import signal
import subprocess
import time


def alive(pid, created):
    import psutil
    try:
        process = psutil.Process(pid)
        return process.create_time() == created and process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def collect_tree(pid, tracked):
    import psutil
    try:
        parent = psutil.Process(pid)
        for process in [parent, *parent.children(recursive=True)]:
            try:
                tracked[process.pid] = process.create_time()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


def stop_tree(process, tracked, seconds):
    """TERM, then KILL remaining recorded descendants even if their parent exits."""
    import psutil
    enumeration_ok = True
    try:
        collect_tree(process.pid, tracked)
    except (OSError, psutil.AccessDenied):
        enumeration_ok = False
    deadline = time.monotonic() + seconds
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for pid, created in list(tracked.items()):
        if alive(pid, created):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    while time.monotonic() < deadline:
        process.poll()
        if not any(alive(pid, created) for pid, created in tracked.items()):
            break
        try:
            collect_tree(process.pid, tracked)
        except (OSError, psutil.AccessDenied):
            enumeration_ok = False
        time.sleep(min(.05, max(0, deadline - time.monotonic())))
    # ProcessGroup cleanup can spend 60 seconds waiting on a child; this outer
    # deadline also bounds that code and reaches children in new process sessions.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid, created in list(tracked.items()):
        if alive(pid, created):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and any(alive(pid, created) for pid, created in tracked.items()):
        time.sleep(.02)
    return enumeration_ok and not any(alive(pid, created) for pid, created in tracked.items())


def cleanup_containers(scope, *, runner=subprocess.run):
    if not re.fullmatch(r"[0-9a-f]{32}", scope):
        raise ValueError("Invalid private run scope")
    result = runner(["docker", "ps", "-aq", "--no-trunc", "--filter", "label=verifiers.run=" + scope],
                    capture_output=True, text=True, timeout=10, check=True)
    ids = result.stdout.split()
    if any(not re.fullmatch(r"[0-9a-f]{64}", container) for container in ids):
        raise RuntimeError("Unexpected container identity")
    if ids:
        runner(["docker", "rm", "-f", *ids], capture_output=True, text=True, timeout=10, check=True)
    remaining = runner(["docker", "ps", "-aq", "--no-trunc", "--filter", "label=verifiers.run=" + scope],
                       capture_output=True, text=True, timeout=10, check=True)
    if remaining.stdout.strip():
        raise RuntimeError("Run-labelled containers remain after cleanup")
    return len(ids)
