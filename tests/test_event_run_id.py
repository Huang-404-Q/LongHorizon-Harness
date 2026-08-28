"""Event ids must carry the run identity the dashboard/API validates against.

The dashboard derives a run's id from the run directory (the reserved id
under ``--runs-root``) or from ``log_dir.parent.name`` for a pinned
``--log-dir``; the worker previously stamped ids from the event log's
grandparent path component instead. Every layout except the default
``<runs-root>/<run-id>/lh_harness`` shape diverged from the reader, and the
event normalizer rejected every record ("event_id does not belong to the
requested run") — an empty feed for a healthy run.
"""

from __future__ import annotations

import json
from pathlib import Path

from lh_harness.manager import _append_event
from lh_harness.types import HarnessConfig


def _read_event_ids(events_path: Path) -> list[str]:
    return [
        json.loads(line)["event_id"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_configured_run_id_overrides_the_path_derived_stamp(tmp_path: Path) -> None:
    log_dir = tmp_path / "custom" / "deep" / "myrun"
    events_path = log_dir / "role_orchestration" / "events.jsonl"

    config = HarnessConfig(log_dir=str(log_dir), run_id="20260828T000000Z_abcd1234")
    _append_event(events_path, "role_harness_start", {}, run_event_id=getattr(config, "run_id", None))

    assert _read_event_ids(events_path) == ["20260828T000000Z_abcd1234:000001"]


def test_without_a_configured_id_the_stamp_falls_back_to_the_log_parent(tmp_path: Path) -> None:
    log_dir = tmp_path / "runs" / "run-2" / "lh_harness"
    events_path = log_dir / "role_orchestration" / "events.jsonl"

    _append_event(events_path, "role_harness_start", {})

    assert _read_event_ids(events_path) == ["run-2:000001"]


def test_the_reader_accepts_the_configured_stamp_and_rejects_foreign_ones(tmp_path: Path) -> None:
    from lh_harness.webapi.events import EventTailer

    log_dir = tmp_path / "custom" / "deep" / "myrun"
    events_path = log_dir / "role_orchestration" / "events.jsonl"
    run_id = "20260828T000000Z_abcd1234"

    config = HarnessConfig(log_dir=str(log_dir), run_id=run_id)
    _append_event(events_path, "role_harness_start", {}, run_event_id=getattr(config, "run_id", None))

    tailer = EventTailer(events_path, run_id=run_id)
    assert len(tailer.read()) == 1, "the pinned --log-dir feed must not be empty"

    foreign = EventTailer(events_path, run_id="some-other-run")
    assert len(foreign.read()) == 0, "the cross-run guard must keep rejecting foreign ids"
