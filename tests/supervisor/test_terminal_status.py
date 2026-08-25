from __future__ import annotations

import json
from pathlib import Path

from lh_harness.supervisor.service import RunSupervisor, _terminal_status_for_exit


def _report(status: str, *, completion_satisfied: bool = False) -> dict:
    return {
        "schema_version": 2,
        "status": status,
        "completion_satisfied": completion_satisfied,
        "rounds": [],
    }


# --- pure mapping -----------------------------------------------------------


def test_nonzero_exit_preserves_deliberate_blocked_outcome() -> None:
    lifecycle, report_status = _terminal_status_for_exit(
        report=_report("blocked"), returncode=1
    )
    assert (lifecycle, report_status) == ("blocked", "blocked")


def test_nonzero_exit_preserves_deliberate_incomplete_outcome() -> None:
    lifecycle, report_status = _terminal_status_for_exit(
        report=_report("incomplete"), returncode=1
    )
    assert (lifecycle, report_status) == ("incomplete", "incomplete")


def test_nonzero_exit_with_completed_claim_still_fails_on_missing_authority() -> None:
    lifecycle, report_status = _terminal_status_for_exit(
        report=_report("completed"), returncode=1
    )
    assert (lifecycle, report_status) == ("failed", "completed")


def test_nonzero_exit_with_authorised_completion_still_fails() -> None:
    # A process that died abnormally is a crash even when it left an
    # authorised report behind; the authority check applies to both exit
    # paths so a late/hand-written report cannot relabel a dead worker.
    lifecycle, report_status = _terminal_status_for_exit(
        report=_report("completed", completion_satisfied=True), returncode=1
    )
    assert (lifecycle, report_status) == ("failed", "completed")


def test_nonzero_exit_with_crash_report_stays_a_failure() -> None:
    # Crash reports carry ``failed``; an unsolicited death with a
    # ``cancelled`` report and no operator action is a failure too (the
    # operator-stop paths set ``requested_action`` and cancel earlier).
    assert _terminal_status_for_exit(report=_report("failed"), returncode=137) == (
        "failed",
        "failed",
    )
    assert _terminal_status_for_exit(report=_report("cancelled"), returncode=137) == (
        "failed",
        "cancelled",
    )


def test_nonzero_exit_without_a_report_is_a_failure() -> None:
    lifecycle, report_status = _terminal_status_for_exit(report={}, returncode=1)
    assert (lifecycle, report_status) == ("failed", "")


def test_nonzero_exit_with_nonterminal_report_is_a_failure() -> None:
    lifecycle, _ = _terminal_status_for_exit(report=_report("running"), returncode=1)
    assert lifecycle == "failed"


def test_requested_stop_still_cancels_even_with_incomplete_report() -> None:
    lifecycle, report_status = _terminal_status_for_exit(
        report=_report("incomplete"), returncode=1, requested_action="stop"
    )
    assert (lifecycle, report_status) == ("cancelled", "incomplete")


# --- through the real supervisor --------------------------------------------


class _ExitedProcess:
    pid = 4242

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode

    def poll(self):
        return self.returncode


def _create_and_exit(monkeypatch, tmp_path: Path, returncode: int, report: dict | None):
    monkeypatch.setattr(
        "lh_harness.supervisor.service.subprocess.Popen",
        lambda *args, **kwargs: _ExitedProcess(returncode),
    )
    monkeypatch.setattr("lh_harness.supervisor.service.os.killpg", lambda *args, **kwargs: None)
    supervisor = RunSupervisor(tmp_path / "runs", workspace_root=tmp_path / "workspace")
    created = supervisor.create_run(task="reach a terminal audit outcome")
    run_id = created["id"]
    if report is not None:
        logs = tmp_path / "runs" / run_id / "lh_harness"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "report.json").write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8"
        )
    return supervisor, run_id, tmp_path / "runs" / run_id


def test_supervisor_reports_incomplete_not_failed_for_nonzero_exit(monkeypatch, tmp_path: Path) -> None:
    supervisor, run_id, run_dir = _create_and_exit(
        monkeypatch,
        tmp_path,
        returncode=1,
        report=_report("incomplete", completion_satisfied=False),
    )

    status = supervisor.status(run_id)

    # The CLI exits non-zero for every non-completed run; the durable report
    # is the authority, so the lifecycle must not be relabelled ``failed``.
    assert status["status"] == "incomplete"
    assert status["report_status"] == "incomplete"
    assert status["alive"] is False
    assert (run_dir / "lh_harness" / "crash_report.json").exists() is False


def test_supervisor_still_reports_failed_when_no_report_survives(monkeypatch, tmp_path: Path) -> None:
    supervisor, run_id, run_dir = _create_and_exit(
        monkeypatch, tmp_path, returncode=1, report=None
    )

    status = supervisor.status(run_id)

    assert status["status"] == "failed"
    assert status["alive"] is False
    assert (run_dir / "lh_harness" / "crash_report.json").exists() is True