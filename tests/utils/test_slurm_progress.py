"""Tests for the progress-threshold notifier (Server Alarm.md §6.1)."""

from __future__ import annotations

import pytest

from mt.utils.slurm_progress import (
    ProgressNotifier,
    is_lead_worker,
    validate_recipient,
)

class RecordingSender:
    """Stand-in mailer that records calls instead of sending."""

    def __init__(self, succeed=True):
        self.succeed = succeed
        self.calls = []

    def __call__(self, recipient, subject, body):
        self.calls.append((recipient, subject, body))
        return self.succeed

def make(total=100, sender=None, **kwargs):
    return ProgressNotifier(total, "centaur-evaluation", "a.b@uni.lu",
                            sender=sender or RecordingSender(), **kwargs)

def thresholds_sent(sender):
    """Which threshold each message was triggered by, read from the body."""

    return [int(line.split()[3].rstrip("%"))
            for _, _, body in sender.calls
            for line in body.splitlines() if line.startswith("Triggered")]

def test_only_the_four_thresholds_notify():
    sender = RecordingSender()
    notifier = make(sender=sender)
    for current in [24, 25, 49, 50, 69, 70, 89, 90, 100]:
        notifier.update(current)
    assert thresholds_sent(sender) == [25, 50, 70, 90]

def test_repeated_progress_does_not_resend():
    sender = RecordingSender()
    notifier = make(sender=sender)
    for _ in range(5):
        notifier.update(25)
    assert thresholds_sent(sender) == [25]

def test_jump_sends_only_the_newest_threshold():
    sender = RecordingSender()
    notifier = make(sender=sender)
    notifier.update(20)
    notifier.update(80)
    assert thresholds_sent(sender) == [70]
    # the skipped ones stay skipped rather than arriving late
    notifier.update(85)
    assert thresholds_sent(sender) == [70]

def test_completion_sends_nothing():
    sender = RecordingSender()
    notifier = make(sender=sender)
    notifier.update(100)
    assert sender.calls == []

def test_a_new_notifier_starts_from_a_clean_slate():
    first = RecordingSender()
    make(sender=first).update(50)
    second = RecordingSender()
    make(sender=second).update(50)
    assert thresholds_sent(first) == [50]
    assert thresholds_sent(second) == [50]

@pytest.mark.parametrize("current", [-1, 101])
def test_out_of_range_progress_is_rejected(current):
    with pytest.raises(ValueError, match="outside"):
        make().update(current)

def test_backwards_progress_is_rejected():
    notifier = make()
    notifier.update(60)
    with pytest.raises(ValueError, match="backwards"):
        notifier.update(30)

@pytest.mark.parametrize("total", [0, -5])
def test_non_positive_total_is_rejected(total):
    with pytest.raises(ValueError, match="positive"):
        make(total=total)

def test_send_failure_does_not_raise_and_retries_next_update(capsys):
    failing = RecordingSender(succeed=False)
    notifier = make(sender=failing)
    assert notifier.update(30) is None
    assert len(failing.calls) == 1
    # still unsettled, so the same threshold is attempted again
    assert notifier.update(31) is None
    assert thresholds_sent(failing) == [25, 25]

def test_sender_exception_is_contained(capsys):
    def explode(recipient, subject, body):
        raise OSError("mailer vanished")

    notifier = make(sender=explode)
    assert notifier.update(30) is None
    assert "failed" in capsys.readouterr().out

def test_body_leaks_no_paths_hosts_or_argv(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "5604923")
    sender = RecordingSender()
    notifier = make(total=200, sender=sender)
    notifier.update(50)
    _, subject, body = sender.calls[0]
    assert "centaur-evaluation" in body
    assert "5604923" in body
    assert "50 of 200" in body
    assert "25.0%" in body
    for leak in ["/", "iris-", "python", "--model", ".csv"]:
        assert leak not in body, f"{leak!r} leaked into {body!r}"
    # the subject must state real progress, never the threshold that tripped
    assert subject == "[mt] centaur-evaluation: 50/200 (25%)"

@pytest.mark.parametrize("recipient", [
    "-nfoo@example.com",
    "--debug@example.com",
    "a.b@uni.lu\nBcc: other@example.com",
    "not-an-email",
    "",
])
def test_hostile_recipients_are_rejected(recipient):
    with pytest.raises(ValueError):
        validate_recipient(recipient)

def test_lead_worker_covers_both_topologies(monkeypatch):
    # model-sharded: one process, no shard variable at all
    monkeypatch.delenv("MT_SHARD", raising=False)
    assert is_lead_worker()
    # data-parallel: only shard 0 notifies, so no duplicate mail
    monkeypatch.setenv("MT_SHARD", "0/4")
    assert is_lead_worker()
    for shard in ["1/4", "2/4", "3/4"]:
        monkeypatch.setenv("MT_SHARD", shard)
        assert not is_lead_worker()
