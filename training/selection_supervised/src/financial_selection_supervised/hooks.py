"""Scoped native interception: account first, then delegate without changing results."""
import asyncio
from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path
import hashlib
import sys
from uuid import uuid4

from .ledger import Ledger

CURRENT_ATTEMPT = ContextVar("selection_supervised_attempt", default=None)
BROKER_PID_ENV = "FINANCIAL_SELECTION_SUPERVISOR_BROKER_PID"
COMPONENT_SHA_ENV = "FINANCIAL_SELECTION_SUPERVISOR_COMPONENT_SHA"


def install_orchestrator(ledger):
    from prime_rl.orchestrator.envs import Env
    from prime_rl.env.client import EnvClient
    original_run, original_request = Env.run, EnvClient._request

    @wraps(original_run)
    async def run(self, *args, **kwargs):
        attempt = ledger.reserve()  # precedes native run/task construction/serialization
        token = CURRENT_ATTEMPT.set(attempt)
        try:
            result = await original_run(self, *args, **kwargs)
        except asyncio.CancelledError:
            ledger.finish(attempt, "cancelled")
            raise
        except BaseException:
            ledger.finish(attempt, "failed")
            raise
        else:
            ledger.finish(attempt, "returned")
            return result
        finally:
            CURRENT_ATTEMPT.reset(token)

    @wraps(original_request)
    async def request(self, request, response_type, timeout=None, on_delta=None, request_id=None):
        if request.method == "run":
            attempt = CURRENT_ATTEMPT.get()
            if attempt is None:
                raise RuntimeError("Native run escaped supervisor reservation")
            request_id = request_id or uuid4().hex
            ledger.bind_request(attempt, request_id)  # before first possible socket send
        return await original_request(self, request, response_type, timeout, on_delta, request_id)

    Env.run, EnvClient._request = run, request

    def restore():
        Env.run, EnvClient._request = original_run, original_request
    return restore


def supervised_worker_setup(*args):
    """Picklable callback composed with native logging, labels and PayloadHook setup."""
    from prime_rl.entrypoints.env_worker_pool import setup_worker
    import psutil
    # run_env_worker_pool also calls this callback in its broker. A broker boot is
    # never counted as worker readiness or allowed to consume the one worker boot.
    if os.getpid() != int(os.environ[BROKER_PID_ENV]):
        from .contract import verify_sources
        Ledger.from_env().worker_boot(os.getpid(), psutil.Process().create_time())
        verify_sources()
    # In the broker the native name is patched, so keep the original in a separate
    # module binding. Spawned workers import the unpatched native module.
    original = _original_worker_setup or setup_worker
    original(*args)


_original_worker_setup = None


def component_main(kind, path):
    from .contract import verify_sources
    verify_sources()
    expected = os.environ[COMPONENT_SHA_ENV]
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Native component configuration changed before startup")
    if kind == "orchestrator":
        install_orchestrator(Ledger.from_env())
        if os.environ.get("FINANCIAL_SELECTION_OPTIMIZER_PLAN_PATH"):
            from .gpu_observer import component_main as observed_component
            return observed_component("orchestrator", Path(path))
        from prime_rl.entrypoints.orchestrator import main
    elif kind == "pool":
        global _original_worker_setup
        import prime_rl.entrypoints.env_worker_pool as native
        os.environ[BROKER_PID_ENV] = str(os.getpid())
        _original_worker_setup = native.setup_worker
        native.setup_worker = supervised_worker_setup
        main = native.main
    else:
        raise ValueError("Unknown supervised component")
    sys.argv = ["orchestrator" if kind == "orchestrator" else "env-worker-pool", "@", str(path)]
    main()
