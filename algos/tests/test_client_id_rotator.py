"""Unit tests for the IBKR client-id rotator (no live gateway needed).

Run:
    python -m pytest algos/tests/test_client_id_rotator.py -v
or directly:
    python algos/tests/test_client_id_rotator.py
"""

import os
import tempfile

# Force a throwaway registry dir BEFORE importing the module so the env var is
# read at import time.
_TMP = tempfile.mkdtemp(prefix="cidrot_")
os.environ["IBKR_CLIENTID_REGISTRY_DIR"] = _TMP

from algos.common import client_id_rotator as R  # noqa: E402


def _alloc(**kw):
    # probe=False so tests never need a live gateway.
    kw.setdefault("probe", False)
    return R.allocate_client_id(host="127.0.0.1", port=4002, **kw)


def test_allocate_is_in_band_and_reserved():
    cid = _alloc(label="t")
    assert R._BAND_MIN <= cid <= R._BAND_MAX
    assert R._reservation_path("127.0.0.1", 4002, cid).exists()
    R.release_client_id(cid, "127.0.0.1", 4002)
    assert not R._reservation_path("127.0.0.1", 4002, cid).exists()


def test_concurrent_allocations_are_distinct():
    ids = [_alloc(label=f"t{i}") for i in range(15)]
    assert len(set(ids)) == len(ids), f"duplicate ids allocated: {ids}"
    for cid in ids:
        R.release_client_id(cid, "127.0.0.1", 4002)


def test_release_then_realloc_pool_not_leaked():
    cid = _alloc()
    R.release_client_id(cid, "127.0.0.1", 4002)
    # Registry should be empty again (nothing leaked).
    leftovers = [p for p in os.listdir(_TMP) if p.endswith(".json")]
    assert leftovers == [], f"registry leaked reservations: {leftovers}"


def test_stale_dead_pid_reservation_is_reclaimable():
    # Manually plant a reservation owned by a guaranteed-dead PID.
    dead = 2_000_000_000
    while R._pid_alive(dead):
        dead += 1
    cid = 4242
    path = R._reservation_path("127.0.0.1", 4002, cid)
    import json
    path.write_text(json.dumps({"pid": dead, "ts": 0, "label": "leaked"}))
    assert path.exists()
    # A reserve attempt on the same id must reclaim the stale file and succeed.
    assert R._try_reserve("127.0.0.1", 4002, cid, "fresh") is True
    R.release_client_id(cid, "127.0.0.1", 4002)


def test_live_pid_reservation_blocks_reuse():
    # A reservation owned by THIS (alive) process must NOT be reclaimable.
    cid = 4243
    assert R._try_reserve("127.0.0.1", 4002, cid, "mine") is True
    # Second attempt for the same id (still alive) must fail.
    assert R._try_reserve("127.0.0.1", 4002, cid, "again") is False
    R.release_client_id(cid, "127.0.0.1", 4002)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} rotator unit tests passed.")
