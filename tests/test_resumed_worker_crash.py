"""A crashed resume must be reported as the crash, not the previous success.

A supervised ``resume --continue`` reopens the same run directory.  The report
of the generation that finished it (``report.json``) is the only source the
worker and the supervisor fall back to when the new generation dies, so a
stale ``complete`` report lets a crashed resume leave the run reporting its
predecessor's success with zero crash evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from lh_harness.cli import _write_bootstrap_failure
from lh_harness.environment.local import LocalEnvironment
from lh_harness.manager import run
from lh_harness.types import EpisodeResult, HarnessConfig, ManagedRound

# The previous generation ended with a durable success.
_STALE_SUCCESS_REPORT = {
    "schema_version": 2,
    "status": "complete",
    "task": "finish the refactor",
    "completion_satisfied": True,
    "rounds_run": 3,
    "max_rounds": 3,
}


def _clean_audit(summary: str) -> str:
    return (
        "Status: complete\n"
        "Integrity: clean\n"
        "Contract audit: aligned\n\n"
        f"Summary:\n{summary}"
    )


def _seed_completed_run(tmp_path: Path) -> Path:
    """Leave the durable artifacts of a completed generation behind."""
    role = tmp_path / "logs" / "role_orchestration"
    role.mkdir(parents=True, exist_ok=True)
    ledger = ManagedRound(
        round_index=1,
        next_step="cli",
        plan_text="plan 1",
        executor_output="output 1",
        auditor_report=_clean_audit("round 1 verified"),
        harness_feedback="",
        task_state="state 1",
        task_contract="contract 1",
        related_report_refs=[],
        manager_status={},
        executor_status={},
        auditor_status={},
    )
    (role / "rounds.jsonl").write_text(
        json.dumps(asdict(ledger), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = dict(_STALE_SUCCESS_REPORT)
    (tmp_path / "logs" / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (role / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return tmp_path / "logs"


# The previous generation's remote copies are irrelevant to the local
# authority; the resumed run must not spend test time on them.
async def no_remote_write(*_args: Any, **_kwargs: Any) -> None:
    return None


def _resumed_run(tmp_path: Path, *, monkeypatch: pytest.MonkeyPatch):
    """Run the management loop as a resumed worker whose first agent call dies."""

    class CrashingAgent:
        async def run_episode(self, _prompt, _env, _budget, live_trajectory_path=None):
            raise RuntimeError("provider exploded during resume")

    monkeypatch.setattr("lh_harness.manager._write_remote_text", no_remote_write)
    monkeypatch.setattr("lh_harness.manager._write_remote_round_text", no_remote_write)
    return run(
        task="finish the refactor",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=HarnessConfig(
            max_total_episodes=1,
            workspace_path=str(tmp_path / "workspace"),
            harness_dir=str(tmp_path / "harness"),
            log_dir=str(tmp_path / "logs"),
        ),
        agent=CrashingAgent(),
        resume=True,
    )


@pytest.mark.asyncio
async def test_a_crashed_resume_does_not_report_the_previous_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_dir = _seed_completed_run(tmp_path)

    report = await _resumed_run(tmp_path, monkeypatch=monkeypatch)

    assert report["status"] == "failed"
    assert report["abort_reason"] == "worker_exception"
    assert report["completion_satisfied"] is False
    assert "provider exploded during resume" in str(report["failure_reason"])

    # The CLI exits from this report: it must be the crash, not the old success.
    persisted = json.loads((log_dir / "report.json").read_text(encoding="utf-8"))
    assert persisted == report
    assert persisted["status"] == "failed"
    assert (log_dir / "role_orchestration" / "report.json").read_text(encoding="utf-8")
    role_report = json.loads(
        (log_dir / "role_orchestration" / "report.json").read_text(encoding="utf-8")
    )
    assert role_report["status"] == "failed"


def test_a_crashed_bootstrap_after_a_resume_replaces_the_stale_report(
    tmp_path: Path,
) -> None:
    # Agent construction can die before the management loop starts; the CLI's
    # bootstrap failure writer is the next writer of the same report.
    log_dir = _seed_completed_run(tmp_path)

    _write_bootstrap_failure(
        str(log_dir),
        "finish the refactor",
        RuntimeError("adapter setup failed"),
        max_rounds=1,
        resume=True,
    )

    persisted = json.loads((log_dir / "report.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["abort_reason"] == "worker_bootstrap_failure"
    assert persisted["completion_satisfied"] is False
    assert persisted["task"] == "finish the refactor"


@pytest.mark.asyncio
async def test_a_crash_after_the_report_still_preserves_a_valid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The original guard: a failure during the post-report remote sync must not
    # erase a valid local report.  The resumed run above is different only
    # because it is a *new generation* reopening the run.
    _seed_completed_run(tmp_path)

    class CompletedAgent:
        async def run_episode(self, _prompt, _env, _budget, live_trajectory_path=None):
            return EpisodeResult(
                status="done",
                actions_log="Next: done\n\nCurrent Task State:\nall finished",
            )

    async def crashing_remote_write(env, path: str, text: str) -> None:
        raise RuntimeError("remote storage fell over after the local report")

    monkeypatch.setattr("lh_harness.manager._write_remote_text", crashing_remote_write)
    monkeypatch.setattr("lh_harness.manager._write_remote_round_text", crashing_remote_write)

    report = await run(
        task="finish the refactor",
        env=LocalEnvironment(str(tmp_path / "tmp")),
        config=HarnessConfig(
            max_total_episodes=1,
            workspace_path=str(tmp_path / "workspace"),
            harness_dir=str(tmp_path / "harness"),
            log_dir=str(tmp_path / "logs"),
        ),
        agent=CompletedAgent(),
        resume=False,
    )

    # No resume: the report on disk belongs to this generation, so the crash
    # path must leave it untouched.
    assert report["status"] == "complete"
    persisted = json.loads((tmp_path / "logs" / "report.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "complete"
    assert persisted["task"] == "finish the refactor"