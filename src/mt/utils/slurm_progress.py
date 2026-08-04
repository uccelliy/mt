"""Progress-threshold email notifications for long Slurm analyses."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import time

THRESHOLDS = (25, 50, 70, 90)
SEND_TIMEOUT_SECONDS = 20
RECIPIENT_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

def is_lead_worker(shard=None):
    """True for the one process that should send notifications."""

    # Data-parallel launchers pass --shard k/n and pin one GPU per process;
    # a model-sharded launcher runs a single process with no shard at all.
    # Letting only shard 0 notify covers both topologies without either
    # launcher having to know which one it is.
    if shard is None:
        shard = os.environ.get("MT_SHARD", "")
    if not shard:
        return True
    return str(shard).split("/")[0].strip() == "0"

def validate_recipient(recipient):
    """Reject anything that could be read as an option or a header."""

    if not isinstance(recipient, str) or not recipient:
        raise ValueError("recipient must be a non-empty string")
    # a leading dash would be parsed as an option by mailx and mail, which
    # take the recipient as a positional argument
    if recipient.startswith("-"):
        raise ValueError(f"recipient must not start with '-': {recipient!r}")
    if not RECIPIENT_PATTERN.match(recipient):
        raise ValueError(f"not a valid email address: {recipient!r}")
    return recipient

def find_mailer():
    """Return (path, style) for the first available local mail command."""

    sendmail = shutil.which("sendmail")
    for candidate in [sendmail, "/usr/sbin/sendmail", "/usr/lib/sendmail"]:
        if candidate and Path(candidate).exists():
            return candidate, "sendmail"
    for name in ["mailx", "mail"]:
        path = shutil.which(name)
        if path:
            return path, "mailx"
    return None, None

def send_mail(recipient, subject, body, *, timeout=SEND_TIMEOUT_SECONDS):
    """Hand one message to the local mail system; True if it accepted it."""

    mailer, style = find_mailer()
    if mailer is None:
        return False
    if style == "sendmail":
        argv = [mailer, "-t", "-oi"]
        payload = (f"To: {recipient}\n"
                   f"Subject: {subject}\n"
                   f"MIME-Version: 1.0\n"
                   f"Content-Type: text/plain; charset=UTF-8\n"
                   f"\n{body}\n")
    else:
        argv = [mailer, "-s", subject, recipient]
        payload = f"{body}\n"
    # a list argv with no shell keeps the subject and body out of any shell
    completed = subprocess.run(argv, input=payload, text=True,
                               timeout=timeout, capture_output=True,
                               check=False)
    return completed.returncode == 0

class ProgressNotifier:
    """Email once per crossed progress threshold, at most once each."""

    def __init__(self, total, label, recipient, *, thresholds=THRESHOLDS,
                 sender=send_mail, clock=time.monotonic):
        if not isinstance(total, int) or total <= 0:
            raise ValueError(f"total must be a positive int, got {total!r}")
        self.total = total
        self.label = label
        self.recipient = validate_recipient(recipient)
        self.thresholds = tuple(sorted(thresholds))
        self.sender = sender
        self.clock = clock
        self.started = clock()
        self.current = 0
        # thresholds already handled, whether sent or deliberately skipped
        self.settled = set()

    def update(self, current):
        """Record progress and send at most one email."""

        if not isinstance(current, int):
            raise ValueError(f"current must be an int, got {current!r}")
        if not 0 <= current <= self.total:
            raise ValueError(f"current {current} outside [0, {self.total}]")
        if current < self.current:
            raise ValueError(f"progress went backwards: {self.current} -> "
                             f"{current}")
        self.current = current

        percent = 100.0 * current / self.total
        reached = [t for t in self.thresholds
                   if t <= percent and t not in self.settled]
        if not reached:
            return None
        # everything below the newest crossed threshold is stale: a jump from
        # 20% to 80% should not deliver 25% and 50% after the fact
        newest = reached[-1]
        self.settled.update(reached[:-1])
        # the run is over; Slurm's END/FAIL mail is the terminal signal
        if current == self.total:
            self.settled.add(newest)
            return None
        if self._send(newest, percent):
            self.settled.add(newest)
            return newest
        # leave `newest` unsettled so the next update() retries it
        return None

    def _send(self, threshold, percent):
        # The subject reports where the run actually is, not which threshold
        # tripped: crossing 25% is an internal trigger, and showing "at 25%"
        # on a message sent at 40% reads as a contradiction.
        subject = (f"[mt] {self.label}: {self.current}/{self.total} "
                   f"({percent:.0f}%)")
        body = self._body(threshold, percent)
        try:
            return bool(self.sender(self.recipient, subject, body))
        except Exception as error:
            # a notification must never take the analysis down with it
            print(f"progress email for {threshold}% failed: {error}",
                  flush=True)
            return False

    def _body(self, threshold, percent):
        """Build the message; deliberately excludes paths, hosts and argv."""

        elapsed = int(self.clock() - self.started)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        job = os.environ.get("SLURM_JOB_ID", "not running under Slurm")
        return (f"Analysis: {self.label}\n"
                f"Slurm Job ID: {job}\n"
                f"Triggered by the {threshold}% threshold\n"
                f"Progress: {self.current} of {self.total} "
                f"({percent:.1f}%)\n"
                f"Elapsed in this process: "
                f"{hours:02d}:{minutes:02d}:{seconds:02d}\n")
