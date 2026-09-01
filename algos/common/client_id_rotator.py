"""IBKR API client-ID rotator — session-scoped, current-ID-aware, collision-proof.

WHY
---
Every IBKR API connection needs a UNIQUE clientId. IBKR's gateway rejects a
duplicate with error 326 ("Unable to connect as the client id is already in
use"). Historically this repo assigned clientIds statically and independently
across four subsystems (execution trading, algos data downloader, scripts,
portfolio_oversight), so they collided — e.g. `python -m
algos.common.update_market_data` hitting 326 because a region trading session
already held its base id.

IBKR has NO API to list connected clients, and the official docs state the
clientId "can be any integer" (the documented "32" is a *concurrent-connection*
cap, not a value range). So "current-id-aware" allocation is achieved by:

  1. RANDOM draw from a wide dedicated band (default 1000-9999), far from every
     legacy static id (regions 1-7, downloader 10, scripts 99, oversight 998),
     making accidental overlap rare.
  2. LOCAL REGISTRY reservation in a repo-root shared dir so concurrent local
     sessions across ALL subsystems coordinate. Reservation uses an atomic
     O_CREAT|O_EXCL file create (race-safe), keyed by (host, port) so different
     gateways don't share a pool. Stale reservations (owner PID dead) are
     reclaimed. Reservation files are PLAIN FILES removed with os.remove — we do
     NOT repeat the rmdir-on-non-empty-dir bug that broke the run_region.sh lock.
  3. LIVE PROBE (optional, default on): briefly connect with the candidate id;
     if the gateway returns 326 the id is in use by someone we can't see in the
     local registry (e.g. a manual TWS client or another machine) — mark it
     busy and pick another. If it connects, disconnect immediately and keep the
     reservation for the real session.

PUBLIC API
----------
    allocate_client_id(host, port, *, probe=True, label=None) -> int
    release_client_id(client_id, host, port) -> None
    session_client_id(host, port, ...)  # contextmanager, auto-releases

On exhaustion raises ClientIdAllocationError so callers can alert — never
silently proceed with a colliding id.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import random
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
# Wide band, well clear of all legacy static ids. clientId may be any integer
# per IBKR docs; the only real cap is 32 *concurrent* clients, which we stay
# far below. Random selection from a large band keeps collisions rare; the
# registry + probe guarantee uniqueness.
_BAND_MIN = int(os.environ.get("IBKR_CLIENTID_MIN", "1000"))
_BAND_MAX = int(os.environ.get("IBKR_CLIENTID_MAX", "9999"))

# Max distinct candidate ids to try before giving up.
_MAX_CANDIDATES = int(os.environ.get("IBKR_CLIENTID_MAX_CANDIDATES", "20"))

# Probe connect timeout (seconds) — how long to wait for connectAck/nextValidId
# or a 326 before deciding.
_PROBE_TIMEOUT = float(os.environ.get("IBKR_CLIENTID_PROBE_TIMEOUT", "3.0"))

# A reservation whose owner PID is dead is always reclaimable. We also treat a
# reservation older than this (even if we can't check the pid) as stale, so a
# crashed run on another host sharing the dir can't wedge the pool forever.
_RESERVATION_MAX_AGE_SECS = int(
    os.environ.get("IBKR_CLIENTID_RESERVATION_MAX_AGE", str(12 * 3600))
)

# In-process guard so threads in the same process don't race the same id before
# the on-disk reservation lands.
_local_lock = threading.Lock()
# Ids reserved by THIS process (for atexit cleanup): {(host,port,id): path}
_owned: dict[tuple[str, int, int], str] = {}


class ClientIdAllocationError(RuntimeError):
    """Raised when no free client id could be allocated."""


# ---------------------------------------------------------------------------
# Registry location
# ---------------------------------------------------------------------------
def _registry_dir() -> Path:
    """Repo-root shared registry dir (override via IBKR_CLIENTID_REGISTRY_DIR).

    Module lives at <repo>/algos/common/client_id_rotator.py, so parents[2] is
    the repo root.
    """
    override = os.environ.get("IBKR_CLIENTID_REGISTRY_DIR")
    if override:
        d = Path(override)
    else:
        d = Path(__file__).resolve().parents[2] / ".ibkr_client_ids"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reservation_path(host: str, port: int, client_id: int) -> Path:
    # host may contain dots; keep it filesystem-safe.
    safe_host = host.replace("/", "_").replace(":", "_")
    return _registry_dir() / f"{safe_host}_{port}_{client_id}.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another user — treat as alive (do not steal).
        return True
    except (OSError, TypeError):
        return False


def _reservation_is_stale(path: Path) -> bool:
    """True if a reservation file may be removed (dead owner or too old)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        # Unreadable/corrupt — consider stale so it can be cleared.
        return True
    pid = data.get("pid")
    ts = data.get("ts", 0)
    if isinstance(pid, int) and not _pid_alive(pid):
        return True
    if time.time() - ts > _RESERVATION_MAX_AGE_SECS:
        return True
    return False


def _sweep_stale(host: str, port: int) -> None:
    """Opportunistically remove stale reservations for this (host,port)."""
    safe_host = host.replace("/", "_").replace(":", "_")
    prefix = f"{safe_host}_{port}_"
    try:
        for p in _registry_dir().glob(f"{prefix}*.json"):
            if _reservation_is_stale(p):
                try:
                    os.remove(p)  # plain file — NOT rmdir
                    logger.debug("Swept stale client-id reservation %s", p.name)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.warning("Could not sweep reservation %s: %s", p.name, e)
    except OSError:
        pass


def _try_reserve(host: str, port: int, client_id: int, label: Optional[str]) -> bool:
    """Atomically reserve client_id. Returns True on success.

    Uses O_CREAT|O_EXCL so two concurrent processes can never both win the same
    id. If the file already exists but is stale, reclaim it and retry once.
    """
    path = _reservation_path(host, port, client_id)
    payload = json.dumps(
        {"pid": os.getpid(), "label": label or "", "ts": time.time(),
         "host": host, "port": port, "client_id": client_id}
    ).encode()
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        if _reservation_is_stale(path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                return False
            # Retry exactly once after reclaiming.
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                try:
                    os.write(fd, payload)
                finally:
                    os.close(fd)
                return True
            except FileExistsError:
                return False
        return False
    except OSError as e:
        logger.warning("Reservation create failed for id %s: %s", client_id, e)
        return False


def _unreserve(host: str, port: int, client_id: int) -> None:
    path = _reservation_path(host, port, client_id)
    try:
        # Only remove if it's ours (defensive against clobbering a live peer).
        try:
            data = json.loads(path.read_text())
            if data.get("pid") not in (os.getpid(), None):
                if _pid_alive(data.get("pid", -1)):
                    logger.debug(
                        "Not releasing id %s — owned by live PID %s",
                        client_id, data.get("pid"),
                    )
                    return
        except (OSError, ValueError):
            pass
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Could not release reservation for id %s: %s", client_id, e)


# ---------------------------------------------------------------------------
# Live probe
# ---------------------------------------------------------------------------
def _probe_free(host: str, port: int, client_id: int) -> Optional[bool]:
    """Connect briefly with client_id to see if it is free RIGHT NOW.

    Returns:
        True  -> connected (id free), already disconnected.
        False -> gateway reported 326 (id in use).
        None  -> inconclusive (gateway unreachable / timeout) — caller decides.
    """
    try:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper
    except ImportError:
        logger.debug("ibapi unavailable; skipping probe for id %s", client_id)
        return None

    class _Probe(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            self.ok = False
            self.in_use = False
            self.done = threading.Event()

        def error(self, reqId, *args):  # noqa: N802 — ibapi 10.x variadic
            # Extract errorCode across old/new signatures.
            code = None
            if len(args) >= 2 and isinstance(args[1], int):
                code = args[1]
            elif len(args) >= 1 and isinstance(args[0], int):
                code = args[0]
            if code == 326:
                self.in_use = True
                self.done.set()

        def connectAck(self):  # noqa: N802
            self.ok = True
            self.done.set()

        def nextValidId(self, orderId):  # noqa: N802
            self.ok = True
            self.done.set()

    app = _Probe()
    try:
        app.connect(host, port, client_id)
    except Exception as e:  # pragma: no cover - connect rarely raises sync
        logger.debug("Probe connect raised for id %s: %s", client_id, e)
        return None

    t = threading.Thread(target=app.run, daemon=True)
    t.start()
    app.done.wait(timeout=_PROBE_TIMEOUT)

    in_use = app.in_use
    connected = app.ok and not in_use
    try:
        app.disconnect()
    except Exception:
        pass
    time.sleep(0.3)  # let the gateway reap the socket before the real session

    if in_use:
        return False
    if connected:
        return True
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def allocate_client_id(
    host: str = "127.0.0.1",
    port: int = 4002,
    *,
    probe: bool = True,
    label: Optional[str] = None,
) -> int:
    """Reserve and return a unique IBKR client id for this session.

    Coordinates with other local sessions via an on-disk registry and (when
    ``probe`` is True) verifies against the live gateway so an id held by an
    invisible client (manual TWS / another host) is also avoided.

    The caller MUST release the id when done — either via release_client_id()
    or by using the session_client_id() context manager. A process-exit hook
    also releases anything still held by this process.

    Raises ClientIdAllocationError if no free id is found.
    """
    _sweep_stale(host, port)
    tried: set[int] = set()
    with _local_lock:
        for _ in range(_MAX_CANDIDATES):
            cid = random.randint(_BAND_MIN, _BAND_MAX)
            if cid in tried:
                continue
            tried.add(cid)

            if not _try_reserve(host, port, cid, label):
                continue  # reserved by someone else locally

            if probe:
                verdict = _probe_free(host, port, cid)
                if verdict is False:
                    # In use on the gateway by an invisible client — release
                    # our reservation and try a different id.
                    _unreserve(host, port, cid)
                    continue
                # verdict True or None (gateway unreachable): keep reservation.
                # If the gateway is down the caller will fail later anyway, but
                # the id is validly reserved for when it comes up.

            _owned[(host, port, cid)] = str(_reservation_path(host, port, cid))
            logger.info(
                "Allocated IBKR client id %s for %s:%s (label=%s, probe=%s)",
                cid, host, port, label or "-", probe,
            )
            return cid

    raise ClientIdAllocationError(
        f"Could not allocate a free IBKR client id for {host}:{port} after "
        f"{_MAX_CANDIDATES} candidates (band {_BAND_MIN}-{_BAND_MAX}). "
        f"Too many concurrent clients, or the gateway is rejecting all ids."
    )


def release_client_id(client_id: int, host: str = "127.0.0.1", port: int = 4002) -> None:
    """Release a previously allocated client id."""
    with _local_lock:
        _unreserve(host, port, client_id)
        _owned.pop((host, port, client_id), None)
    logger.info("Released IBKR client id %s for %s:%s", client_id, host, port)


@contextmanager
def session_client_id(
    host: str = "127.0.0.1",
    port: int = 4002,
    *,
    probe: bool = True,
    label: Optional[str] = None,
):
    """Context manager that allocates a client id and releases it on exit."""
    cid = allocate_client_id(host, port, probe=probe, label=label)
    try:
        yield cid
    finally:
        release_client_id(cid, host, port)


@atexit.register
def _release_all_on_exit() -> None:
    for (host, port, cid) in list(_owned.keys()):
        try:
            _unreserve(host, port, cid)
        except Exception:
            pass
    _owned.clear()
