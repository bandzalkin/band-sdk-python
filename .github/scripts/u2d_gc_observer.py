"""Diagnostic-only pytest plugin (fork CI only): passive cyclic-GC timing plus failure-only state.

Recorder: one ``gc.callbacks`` listener, appended once at configure time. Each
collection's start/stop pair (matched by generation and collecting thread) is
written by index into a fixed, preallocated ring of the most recent RING
collections, short ones included: sequence, generation, collecting thread, start
and end ``time.monotonic``, start and end ``time.thread_time`` of the collecting
thread, collected and uncollectable. A pending start opens only after all its
fields are set; any recorder error closes it, and a row's sequence is written
last, so only fully initialized matched intervals are published. Counters record
overwritten slots, unmatched starts/stops and recorder errors. The callback takes no locks and does no stack
inspection, serialization, file IO or container growth. Collector enablement,
thresholds and debug flags are never touched; nothing forces, disables, freezes
or retunes collection.

Failure-only capture: ``pytest.fail`` is replaced by an instance of a ``_Fail``
subclass with the same signature and the same class-level ``Exception``. It
forwards every call exactly once to the original, catches only the original's
failure class and bare-re-raises the same instance. Only when that failure comes
from the pinned helper line (``tests/runtime/conftest.py:48`` in
``wait_for_condition``) called from the pinned deferral target line
(``tests/runtime/test_deferral.py:145``) with ``recovery == "live"`` does it
first freeze a sequence/time cutoff and a ring copy, then copy scalar state: the
helper's timeout/deadline, the peer's marks, adapter and execution-context state
and every task's await chain. File identity is checkout-relative (relative to
pytest's rootpath), never a suffix match. Shutdown state is copied in
``pytest_runtest_makereport`` only for the exact checkout-relative target
(``tests/test_session_shutdown.py`` :: ``TestInstantShutdown`` ::
``test_stop_returns_quickly_when_idle``), gated before any freeze or traceback
walk, from the failed assertion's own traceback (ctx,
start, elapsed): after-stop state, not an in-flight snapshot. No frame object is
retained. Capture errors, including a BaseException such as KeyboardInterrupt
raised inside the diagnostics or an exception whose ``__str__`` raises, are
recorded by nonthrowing accounting and never replace the original failure.

Windows: initial-A nominal loop window [deadline - timeout, deadline]; the
interval (deadline, interception] is only an upper bound on when the failing
check ran and is reported separately. Shutdown window [start, start + elapsed],
excluding assertion formatting and reporting. Loop time is mapped to monotonic
time only when (a) report-time evidence recognizes the clock implementation
itself (stdlib ``BaseEventLoop.time`` or looptime 0.7 ``LoopTimeEventLoop.time``
delegating to it, no instance override, original ``time.monotonic``, looptime off,
one agreeing sample) and (b) the historical window is established separately:
trivially for a stdlib loop, or for a looptime loop when looptime's own per-item
decision says it was not enabled for the test and the test module does not
reference looptime. A sampled value alone never validates a clock; otherwise
attribution is Unknown.

Inference limits: brackets measure observed cyclic-collection intervals and
include other callbacks', finalizers' and descheduling cost; collecting-thread
CPU is not loop-thread CPU unless the threads match; a window with complete
coverage and no overlap excludes only an observed cyclic collection, not
refcount-triggered destructors, weakref callbacks or other stalls. Any unmatched
event, recorder error, pending collection or copy race makes the verdict Unknown
before overlap is interpreted; incomplete coverage prevents a negative
conclusion. Preallocation does not remove timestamp/callback/update overhead.

Reporting and teardown never raise: serialization, terminal and file failures are
recorded and mark the evidence invalid (``U2D_EVIDENCE_INVALID`` on any working
sink); teardown removes only the owned callback and the owned interception, by
identity, first, and never overwrites an intervening change to ``pytest.fail``,
the callback list or collector policy.
"""

from __future__ import annotations

import asyncio
import asyncio.base_events
import collections
import gc
import json
import os
import sys
import threading
import time
from array import array
from typing import NoReturn

import pytest
from _pytest import outcomes

HELPER_FILE = os.path.join("tests", "runtime", "conftest.py")
HELPER_FUNCTION = "wait_for_condition"
HELPER_FAIL_LINE = 48
DEFERRAL_FILE = os.path.join("tests", "runtime", "test_deferral.py")
DEFERRAL = "test_approval_beyond_first_page_releases_ordered_work"
DEFERRAL_CALL_LINE = 145
SHUTDOWN_FILE = os.path.join("tests", "test_session_shutdown.py")
SHUTDOWN = "test_stop_returns_quickly_when_idle"
RING = int(os.environ.get("U2D_GC_RING", "16384"))
CLOCK_TOLERANCE_S = 0.002

# Ring rows. Times: start wall, end wall, start thread CPU, end thread CPU.
# Meta: sequence, generation, thread ident, collected, uncollectable.
_TIMES, _META = 4, 5
_ring_t = array("d", bytes(8 * _TIMES * RING))
_ring_m = array("q", bytes(8 * _META * RING))
_pending_t = array("d", bytes(8 * 2))  # start wall, start thread CPU
_pending_m = array("q", bytes(8 * 3))  # open flag, generation, thread ident
_gen = array("d", bytes(8 * 3 * 3))  # per generation: count, total s, max s
# Counters: next sequence, overwritten slots, unmatched starts, unmatched stops, recorder errors.
SEQ, OVERWRITTEN, UNMATCHED_START, UNMATCHED_STOP, RECORDER_ERRORS = range(5)
_counters = array("q", bytes(8 * 5))

STATE: dict = {"installed": False}
CAPTURES: list[dict] = []
ERRORS: list[str] = []


def _gc_callback(phase, info) -> None:
    """Publishes a row only for a fully initialized start matched by its stop."""
    try:
        if phase == "start":
            if _pending_m[0]:
                _counters[UNMATCHED_START] += 1
            _pending_m[0] = 0
            generation = info["generation"]
            ident = threading.get_ident()
            start = time.monotonic()
            start_cpu = time.thread_time()
            _pending_m[1] = generation
            _pending_m[2] = ident
            _pending_t[0] = start
            _pending_t[1] = start_cpu
            _pending_m[0] = 1  # opened only after every start field is set
            return
        end = time.monotonic()
        end_cpu = time.thread_time()
        generation = info["generation"]
        collected = info["collected"]
        uncollectable = info["uncollectable"]
        if (
            not _pending_m[0]
            or _pending_m[1] != generation
            or _pending_m[2] != threading.get_ident()
        ):
            _pending_m[0] = 0
            _counters[UNMATCHED_STOP] += 1
            return
        _pending_m[0] = 0
        n = _counters[SEQ]
        t = _TIMES * (n % RING)
        m = _META * (n % RING)
        _ring_m[m] = -1  # the slot is invalid until its sequence is written last
        _ring_t[t] = _pending_t[0]
        _ring_t[t + 1] = end
        _ring_t[t + 2] = _pending_t[1]
        _ring_t[t + 3] = end_cpu
        _ring_m[m + 1] = generation
        _ring_m[m + 2] = _pending_m[2]
        _ring_m[m + 3] = collected
        _ring_m[m + 4] = uncollectable
        _ring_m[m] = n
        if n >= RING:
            _counters[OVERWRITTEN] += 1
        _counters[SEQ] = n + 1
        g = 3 * generation
        duration = end - _pending_t[0]
        _gen[g] += 1
        _gen[g + 1] += duration
        _gen[g + 2] = max(_gen[g + 2], duration)
    except Exception:  # noqa: BLE001 - a diagnostic callback must never disturb the collector
        _pending_m[0] = 0
        _counters[RECORDER_ERRORS] += 1


def _freeze() -> dict:
    """Cutoff before any expensive capture: sequence, time, counters and a ring copy."""
    STATE["freezes"] = STATE.get("freezes", 0) + 1
    cut_seq = _counters[SEQ]
    cut_wall = time.monotonic()
    ring_t, ring_m = _ring_t[:], _ring_m[:]
    return {
        "cut_seq": cut_seq,
        "cut_wall": cut_wall,
        "seq_after_copy": _counters[SEQ],
        "counters": list(_counters),
        "pending_open": bool(_pending_m[0]),
        "ring_t": ring_t,
        "ring_m": ring_m,
    }


def _rows(frozen: dict) -> list[dict]:
    cut = frozen["cut_seq"]
    rows = []
    for seq in range(max(0, cut - RING), cut):
        t, m = _TIMES * (seq % RING), _META * (seq % RING)
        if frozen["ring_m"][m] != seq:
            continue  # overwritten while copying; counted by the validation below
        rt, rm = frozen["ring_t"], frozen["ring_m"]
        rows.append(
            {
                "seq": seq,
                "generation": rm[m + 1],
                "thread": rm[m + 2],
                "start": rt[t],
                "end": rt[t + 1],
                "wall_s": rt[t + 1] - rt[t],
                "thread_cpu_s": rt[t + 3] - rt[t + 2],
                "collected": rm[m + 3],
                "uncollectable": rm[m + 4],
            }
        )
    return rows


def _overlaps(rows: list[dict], start: float, end: float, loop_thread: int) -> dict:
    hits = [r for r in rows if r["end"] >= start and r["start"] <= end]
    return {
        "window": [start, end],
        "count": len(hits),
        "clipped_overlap_s": sum(
            min(r["end"], end) - max(r["start"], start) for r in hits
        ),
        "wall_s": sum(r["wall_s"] for r in hits),
        "collecting_thread_cpu_s": sum(r["thread_cpu_s"] for r in hits),
        "on_loop_thread": sum(1 for r in hits if r["thread"] == loop_thread),
        "on_other_threads": sum(1 for r in hits if r["thread"] != loop_thread),
        "intervals": hits[-200:],
    }


def _verdict(
    frozen: dict,
    rows: list[dict],
    window_start: float | None,
    overlap: dict | None,
    clock_ok: bool,
) -> dict:
    counters = frozen["counters"]
    complete = frozen["cut_seq"] <= RING or (
        rows and rows[0]["start"] <= (window_start or float("-inf"))
    )
    expected_rows = min(frozen["cut_seq"], RING)
    clean = (
        counters[UNMATCHED_START] == 0
        and counters[UNMATCHED_STOP] == 0
        and counters[RECORDER_ERRORS] == 0
        and frozen["seq_after_copy"] == frozen["cut_seq"]
        and not frozen["pending_open"]
        and len(rows) == expected_rows
    )
    if not clock_ok or overlap is None:
        verdict = "Unknown: loop-to-monotonic mapping not validated"
    elif not clean:
        verdict = "Unknown: recorder or snapshot integrity not trusted"
    elif overlap["count"]:
        verdict = "observed cyclic-collection overlap"
    elif complete:
        verdict = (
            "no observed cyclic-collection overlap (excludes only cyclic collections)"
        )
    else:
        verdict = "Unknown: incomplete coverage"
    return {
        "verdict": verdict,
        "coverage_complete": bool(complete),
        "recorder_clean": clean,
        "counters_at_cutoff": dict(
            zip(
                (
                    "next_seq",
                    "overwritten",
                    "unmatched_start",
                    "unmatched_stop",
                    "recorder_errors",
                ),
                counters,
            )
        ),
        "cut_seq": frozen["cut_seq"],
        "cut_wall": frozen["cut_wall"],
        "pending_open_at_cutoff": frozen["pending_open"],
        "snapshot_rows_missing": expected_rows - len(rows),
        "oldest_retained_start": rows[0]["start"] if rows else None,
    }


def _after_capture(frozen: dict) -> dict:
    now = _counters[SEQ]
    return {
        "collections_after_cutoff": now - frozen["cut_seq"],
        "ring_slots_overwritten_after_cutoff": max(0, now - RING)
        - max(0, frozen["cut_seq"] - RING),
    }


_MONOTONIC = time.monotonic
_BASE_TIME = asyncio.base_events.BaseEventLoop.time


def _looptime_time():
    try:
        from looptime._internal.loops import LoopTimeEventLoop  # noqa: PLC0415 - optional, pinned 0.7
    except Exception:  # noqa: BLE001 - absent or moved: the implementation is not recognized
        return None, None
    return LoopTimeEventLoop, LoopTimeEventLoop.__dict__.get("time")


def _next_time_after(cls_type, owner) -> object:
    mro = cls_type.__mro__
    for cls in mro[mro.index(owner) + 1 :]:
        if "time" in cls.__dict__:
            return cls.__dict__["time"]
    return None


def _looptime_enabled_for(item) -> bool | None:
    """The looptime plugin's own per-test decision (markers and option); None if unknown."""
    if item is None:
        return None
    try:
        from looptime._internal.plugin import _get_options  # noqa: PLC0415 - pinned 0.7 decision function

        return _get_options(item) is not None
    except Exception:  # noqa: BLE001
        return None


def _clock(loop, item, module_globals: dict | None) -> dict:
    """Report-time evidence about the loop clock, kept apart from the historical-window inference.

    Report time: the loop's ``time`` must be a recognized implementation with no instance
    override (the stdlib ``BaseEventLoop.time`` or looptime 0.7's ``LoopTimeEventLoop.time``
    delegating to it), ``asyncio.base_events`` must still use the original ``time.monotonic``,
    looptime must be off now and one sample must agree with ``time.monotonic``.
    Historical window: established only if the clock cannot have been virtual during the test,
    i.e. a plain stdlib loop, or a looptime loop whose plugin decision for this test item is
    'not enabled' and whose test module does not reference looptime. Otherwise Unknown.
    """
    if loop is None:
        return {"validated": False, "reason": "no loop recovered"}
    looptime_cls, looptime_time = _looptime_time()
    impl = getattr(type(loop), "time", None)
    instance_override = "time" in getattr(loop, "__dict__", {})
    is_stdlib = impl is _BASE_TIME
    is_looptime = (
        looptime_time is not None
        and impl is looptime_time
        and _next_time_after(type(loop), looptime_cls) is _BASE_TIME
    )
    monotonic_original = (
        getattr(asyncio.base_events, "time", None) is time
        and time.monotonic is _MONOTONIC
    )
    looptime_on = getattr(loop, "looptime_on", None) if is_looptime else None
    before = time.monotonic()
    loop_now = loop.time()
    after = time.monotonic()
    offset = loop_now - (before + after) / 2
    report_ok = (
        (is_stdlib or is_looptime)
        and not instance_override
        and monotonic_original
        and not looptime_on
        and abs(offset) <= CLOCK_TOLERANCE_S
    )
    if is_stdlib:
        historical, basis = True, "stdlib BaseEventLoop.time has no virtual mode"
    elif is_looptime:
        enabled = _looptime_enabled_for(item)
        module_refs = module_globals is not None and any(
            "looptime" in getattr(value, "__module__", "")
            or getattr(value, "__name__", "") == "looptime"
            for value in module_globals.values()
        )
        historical = enabled is False and module_globals is not None and not module_refs
        basis = f"looptime plugin decision for the item: enabled={enabled}; test module references looptime: {module_refs if module_globals is not None else 'unknown'}"
    else:
        historical, basis = False, "clock implementation not recognized"
    return {
        "validated": bool(report_ok and historical),
        "report_time": {
            "recognized": bool(is_stdlib or is_looptime),
            "implementation": f"{getattr(impl, '__module__', None)}.{getattr(impl, '__qualname__', None)}",
            "instance_override": instance_override,
            "monotonic_original": monotonic_original,
            "looptime_on": looptime_on,
            "loop_minus_monotonic_s": offset,
            "ok": bool(report_ok),
        },
        "historical_window": {"established": bool(historical), "basis": basis},
        "loop_minus_monotonic_s": offset,
        "loop_type": f"{type(loop).__module__}.{type(loop).__qualname__}",
        "loop_closed": loop.is_closed(),
    }


def _await_chain(awaitable) -> list:
    chain, seen = [], set()
    while awaitable is not None and id(awaitable) not in seen and len(chain) < 64:
        seen.add(id(awaitable))
        frame = (
            getattr(awaitable, "cr_frame", None)
            or getattr(awaitable, "gi_frame", None)
            or getattr(awaitable, "ag_frame", None)
        )
        if frame is None:
            chain.append({"awaiting": type(awaitable).__qualname__})
            break
        chain.append([frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno])
        awaitable = (
            getattr(awaitable, "cr_await", None)
            or getattr(awaitable, "gi_yieldfrom", None)
            or getattr(awaitable, "ag_await", None)
        )
    return chain


def _task(task) -> dict:
    waiter = getattr(task, "_fut_waiter", None)
    return {
        "name": task.get_name(),
        "done": task.done(),
        "cancelled": task.cancelled() if task.done() else None,
        "cancelling": task.cancelling() if hasattr(task, "cancelling") else None,
        "chain": [] if task.done() else _await_chain(task.get_coro()),
        "fut_waiter": None if waiter is None else repr(waiter)[:300],
    }


def _value(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, asyncio.Task):
        return {"task": _task(value)}
    if isinstance(value, asyncio.Event):
        return {"event_set": value.is_set()}
    if isinstance(value, (list, tuple, set, frozenset, collections.deque)):
        items = list(value)
        if all(isinstance(i, (str, int, float, bool)) or i is None for i in items):
            return items[-50:]
        if all(
            isinstance(i, tuple)
            and all(isinstance(x, (str, int, float, bool)) or x is None for x in i)
            for i in items
        ):
            return [list(i) for i in items[-50:]]
        return {"type": type(value).__name__, "len": len(items)}
    if isinstance(value, dict):
        if any(isinstance(k, asyncio.Task) for k in value):
            return {"tasks": [_task(k) for k in value][:20]}
        return {"type": "dict", "len": len(value)}
    return {"type": type(value).__qualname__}


def _object_state(obj) -> dict:
    return {name: _value(value) for name, value in sorted(vars(obj).items())}


def _ctx_state(ctx) -> dict:
    state = _object_state(ctx)
    try:
        cycle = ctx._active_cycle_task
        state["_active_cycle_task"] = None if cycle is None else {"task": _task(cycle)}
    except Exception as error:  # noqa: BLE001 - diagnostic read only
        state["_active_cycle_task"] = f"unreadable: {type(error).__name__}"
    return state


def _relative(path) -> str | None:
    try:
        return os.path.relpath(os.path.realpath(os.fspath(path)), STATE["rootpath"])
    except Exception:  # noqa: BLE001 - unknown location is never a pinned target
        return None


def _is_pinned_initial_a(helper, target) -> bool:
    return (
        helper.f_code.co_name == HELPER_FUNCTION
        and helper.f_lineno == HELPER_FAIL_LINE
        and _relative(helper.f_code.co_filename) == HELPER_FILE
        and target is not None
        and target.f_code.co_name == DEFERRAL
        and target.f_lineno == DEFERRAL_CALL_LINE
        and _relative(target.f_code.co_filename) == DEFERRAL_FILE
        and target.f_locals.get("recovery") == "live"
    )


def _deferral_item():
    """The live deferral item, looked up only after a pinned failure (never on passing paths)."""
    session = STATE.get("session")
    for item in getattr(session, "items", ()) or ():
        callspec = getattr(item, "callspec", None)
        if (
            getattr(item, "originalname", None) == DEFERRAL
            and callspec is not None
            and callspec.params.get("recovery") == "live"
            and _relative(item.path) == DEFERRAL_FILE
        ):
            return item
    return None


def _is_pinned_shutdown(item) -> bool:
    cls = getattr(item, "cls", None)
    return (
        getattr(item, "name", None) == SHUTDOWN
        and cls is not None
        and cls.__name__ == "TestInstantShutdown"
        and _relative(item.path) == SHUTDOWN_FILE
    )


def _capture_initial_a(helper, target, failure: BaseException) -> None:
    loop = asyncio.get_running_loop()
    intercepted_loop = loop.time()
    frozen = _freeze()
    clock = _clock(loop, _deferral_item(), target.f_globals)
    helper_locals, test_locals = helper.f_locals, target.f_locals
    deadline, timeout = helper_locals.get("deadline"), helper_locals.get("timeout")
    record = {
        "kind": "initial-A helper failure, captured before test cleanup",
        "failure_id": id(failure),
        "failure_type": type(failure).__qualname__,
        "pinned": [
            HELPER_FILE,
            HELPER_FAIL_LINE,
            DEFERRAL_FILE,
            DEFERRAL_CALL_LINE,
            "live",
        ],
        "helper": {
            "timeout": timeout,
            "interval": helper_locals.get("interval"),
            "deadline_loop": deadline,
            "intercepted_loop": intercepted_loop,
        },
        "clock": clock,
        "peer": _object_state(test_locals["peer"]) if "peer" in test_locals else None,
        "adapter": _object_state(test_locals["adapter"])
        if "adapter" in test_locals
        else None,
    }
    rows = _rows(frozen)
    loop_thread = threading.get_ident()
    nominal = upper = None
    if clock["validated"] and deadline is not None and timeout is not None:
        shift = clock["loop_minus_monotonic_s"]
        nominal = _overlaps(
            rows, deadline - timeout - shift, deadline - shift, loop_thread
        )
        upper = _overlaps(rows, deadline - shift, intercepted_loop - shift, loop_thread)
    record["gc"] = {
        "nominal_window": nominal,
        "after_deadline_upper_bound_window": upper,
        "loop_thread": loop_thread,
        **_verdict(
            frozen,
            rows,
            None if nominal is None else nominal["window"][0],
            nominal,
            clock["validated"],
        ),
    }
    record["ctx"] = _ctx_state(test_locals["ctx"]) if "ctx" in test_locals else None
    record["tasks"] = [_task(t) for t in asyncio.all_tasks(loop)]
    record["gc"].update(_after_capture(frozen))
    CAPTURES.append(record)


class _ObservedFail(outcomes._Fail):
    """``pytest.fail`` that forwards once to the original and observes only the pinned failure."""

    def __call__(self, reason: str = "", pytrace: bool = True) -> NoReturn:
        __tracebackhide__ = True
        try:
            result = STATE["original_fail"](reason, pytrace)
        except STATE["failure_class"] as failure:
            helper = target = None
            try:
                helper = sys._getframe(1)
                target = helper.f_back
                if _is_pinned_initial_a(helper, target):
                    _capture_initial_a(helper, target, failure)
            except BaseException as error:  # noqa: BLE001 - even an interrupt inside diagnostics must not replace the original failure
                _note("initial-A capture", error)
            finally:
                del helper, target
            raise
        return result  # unreachable for pytest's own fail, which always raises


def _shutdown_capture(item, call) -> None:
    frozen = _freeze()
    test_locals = loop = None
    tb = call.excinfo.tb
    while tb is not None:
        frame = tb.tb_frame
        candidate = frame.f_locals.get("self")
        if loop is None and isinstance(candidate, asyncio.AbstractEventLoop):
            loop = candidate
        if (
            frame.f_code.co_name == SHUTDOWN
            and _relative(frame.f_code.co_filename) == SHUTDOWN_FILE
        ):
            test_locals = dict(frame.f_locals)
        tb = tb.tb_next
    del tb
    if test_locals is None or "ctx" not in test_locals:
        return
    start, elapsed = test_locals.get("start"), test_locals.get("elapsed")
    clock = _clock(loop, item, getattr(getattr(item, "module", None), "__dict__", None))
    rows = _rows(frozen)
    loop_thread = threading.get_ident()
    window = None
    if clock["validated"] and isinstance(start, float) and isinstance(elapsed, float):
        shift = clock["loop_minus_monotonic_s"]
        window = _overlaps(rows, start - shift, start + elapsed - shift, loop_thread)
    record = {
        "kind": "after-stop state from the failed assertion traceback (not an in-flight snapshot)",
        "node": item.nodeid,
        "error": f"{call.excinfo.typename}: {str(call.excinfo.value)[:300]}",
        "start_loop": start,
        "elapsed_loop_s": elapsed,
        "clock": clock,
        "gc": {
            "stop_window": window,
            "loop_thread": loop_thread,
            **_verdict(
                frozen,
                rows,
                None if window is None else window["window"][0],
                window,
                clock["validated"],
            ),
        },
        "ctx": _ctx_state(test_locals["ctx"]),
    }
    del test_locals
    record["gc"].update(_after_capture(frozen))
    CAPTURES.append(record)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    if call.when == "call" and call.excinfo is not None and _is_pinned_shutdown(item):
        try:
            _shutdown_capture(item, call)
        except BaseException as error:  # noqa: BLE001 - even an interrupt inside diagnostics must not break reporting
            _note("after-stop capture", error)
    yield


def pytest_configure(config) -> None:
    if STATE["installed"]:
        return
    original = pytest.fail
    observed = _ObservedFail()
    STATE.update(
        installed=True,
        original_fail=original,
        failure_class=original.Exception,
        observed_fail=observed,
        settings_at_install=(gc.isenabled(), gc.get_threshold(), gc.get_debug()),
        rootpath=os.path.realpath(os.fspath(config.rootpath)),
        freezes=0,
    )
    gc.callbacks.append(_gc_callback)
    pytest.fail = observed


def pytest_sessionstart(session) -> None:
    STATE["session"] = session


def _summary() -> dict:
    return {
        "counters": dict(
            zip(
                (
                    "next_seq",
                    "overwritten",
                    "unmatched_start",
                    "unmatched_stop",
                    "recorder_errors",
                ),
                _counters,
            )
        ),
        "ring": RING,
        "per_generation": [
            {
                "generation": g,
                "count": int(_gen[3 * g]),
                "total_s": _gen[3 * g + 1],
                "max_s": _gen[3 * g + 2],
            }
            for g in range(3)
        ],
        "settings_at_install": [
            STATE["settings_at_install"][0],
            list(STATE["settings_at_install"][1]),
            STATE["settings_at_install"][2],
        ],
        "settings_now": [gc.isenabled(), list(gc.get_threshold()), gc.get_debug()],
        "observer_errors": ERRORS,
        "freezes": STATE.get("freezes", 0),
        "captures": len(CAPTURES),
    }


def _note(prefix: str, error: BaseException) -> None:
    """Records a diagnostic failure; nothing here may raise, whatever the error does."""
    try:
        name = type(error).__name__
    except BaseException:  # noqa: BLE001
        name = "<unnamed>"
    try:
        text = f"{prefix}: {name}: {error}"
    except BaseException:  # noqa: BLE001 - an exception whose __str__ raises
        text = f"{prefix}: {name}: <unprintable>"
    try:
        ERRORS.append(text)
    except BaseException:  # noqa: BLE001
        STATE["unrecorded_errors"] = STATE.get("unrecorded_errors", 0) + 1


def _invalidate(reason: str) -> None:
    try:
        STATE.setdefault("evidence_invalid", []).append(reason)
    except BaseException as error:  # noqa: BLE001 - invalidation must not raise
        _note("evidence invalidation", error)


def _serialized_lines() -> list[str]:
    lines = []
    items = [("U2D_GC_SUMMARY", _summary)] + [
        ("U2D_FAILURE_CAPTURE", lambda c=c: c) for c in list(CAPTURES)
    ]
    for label, produce in items:
        try:
            lines.append(
                label + " " + json.dumps(produce(), sort_keys=True, default=repr)
            )
        except BaseException as error:  # noqa: BLE001 - reporting must not change the run
            _note(f"serialize {label}", error)
            _invalidate(f"serialize {label}")
            lines.append(label + ' {"serialization_failed": true}')
    return lines


def _write_file(lines: list[str], sink: str) -> bool:
    out = os.environ.get("U2D_GC_OUT")
    if not out:
        return True
    try:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return True
    except BaseException as error:  # noqa: BLE001 - reporting must not change the run
        _note(f"{sink} file output", error)
        _invalidate(f"{sink} file output")
        return False


def _write_terminal(terminalreporter, lines: list[str]) -> bool:
    try:
        terminalreporter.write_sep("=", "U2D GC OBSERVER")
        for line in lines:
            terminalreporter.write_line(line)
        return True
    except BaseException as error:  # noqa: BLE001 - reporting must not change the run
        _note("terminal output", error)
        _invalidate("terminal output")
        return False


def _invalid_marker() -> str:
    try:
        return "U2D_EVIDENCE_INVALID " + json.dumps(
            {
                "reasons": list(STATE.get("evidence_invalid", [])),
                "observer_errors": list(ERRORS),
            },
            default=repr,
        )
    except BaseException:  # noqa: BLE001
        return "U2D_EVIDENCE_INVALID {}"


def pytest_terminal_summary(terminalreporter) -> None:
    if not STATE.get("installed"):
        return
    lines = _serialized_lines()
    terminal_ok = _write_terminal(terminalreporter, lines)
    file_ok = _write_file(lines, "summary")
    if not (terminal_ok and file_ok) or STATE.get("evidence_invalid"):
        marker = [_invalid_marker()]
        if terminal_ok:
            _write_terminal_line(terminalreporter, marker[0])
        if file_ok:
            _write_file(marker, "invalid-marker")


def _write_terminal_line(terminalreporter, line: str) -> None:
    try:
        terminalreporter.write_line(line)
    except BaseException as error:  # noqa: BLE001
        _note("terminal output", error)


def pytest_unconfigure(config) -> None:
    if not STATE.get("installed"):
        return
    removed = owned_fail = False
    try:
        for index, callback in enumerate(gc.callbacks):
            if callback is _gc_callback:
                del gc.callbacks[index]
                removed = True
                break
    except BaseException as error:  # noqa: BLE001 - cleanup must not raise
        _note("listener removal", error)
    try:
        owned_fail = pytest.fail is STATE["observed_fail"]
        if owned_fail:
            pytest.fail = STATE["original_fail"]
    except BaseException as error:  # noqa: BLE001 - cleanup must not raise
        _note("interception removal", error)
    STATE["installed"] = False
    try:
        report = {
            "listener_removed": removed,
            "interception_restored": owned_fail,
            "intervening_pytest_fail_left_in_place": not owned_fail,
            "collector_settings_changed_since_install": (
                gc.isenabled(),
                gc.get_threshold(),
                gc.get_debug(),
            )
            != STATE["settings_at_install"],
            "evidence_invalid": list(STATE.get("evidence_invalid", [])),
        }
        line = "U2D_GC_REMOVAL " + json.dumps(report, sort_keys=True, default=repr)
    except BaseException as error:  # noqa: BLE001
        _note("removal report", error)
        line = "U2D_GC_REMOVAL {}"
    if not _write_file([line], "removal"):
        try:
            sys.__stderr__.write(_invalid_marker() + "\n")
        except BaseException as error:  # noqa: BLE001 - last-resort sink must not raise
            _note("stderr marker", error)
