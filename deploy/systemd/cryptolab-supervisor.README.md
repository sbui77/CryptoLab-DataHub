# Hardening 4.4 — persistent autonomous supervisor

The supervisor (`.claude/bin/supervisor`) is bounded on purpose: one
invocation runs at most `--max-cycles` cycles and then stops, leaving
everything it needs to continue in durable runtime state. That makes
it correct but not persistent — something has to invoke it again.

These units are that something. They are `--user` units, not system
units, and they are the only files here that concern the control
plane; `cryptolab-derivatives-rest.*` and `cryptolab-liquidation.*`
are unrelated system units for data collection.

| File | Role |
| --- | --- |
| `cryptolab-supervisor.timer` | the schedule; the unit an operator enables |
| `cryptolab-supervisor.service` | one bounded batch of supervisor cycles |
| `cryptolab-supervisor-guard` | read-only `ExecCondition=`; decides whether a cycle may start |

## Why `--user` and not a system service

The supervisor spawns Claude worker sessions using a developer's own
credentials and drives that developer's worktree. It is that person's
process. A system service would run it as a service account that owns
neither.

## Why oneshot plus a timer, and not `Restart=always`

A long-lived supervised process fights the supervisor's design. The
supervisor is deliberately bounded and deliberately resumable, and its
non-zero exits are mostly *decisions*, not crashes:

| Exit | Meaning |
| --- | --- |
| 0 | the batch completed, or the task reached `READY_FOR_HUMAN_REVIEW` |
| 10 | `WAITING_FOR_HUMAN` — a Human Gate is open |
| 11 | blocked on permission |
| 12 | blocked technically |
| 13 | another orchestrator holds the worktree lock |
| 14 | `EXTERNAL_REVIEW` could not be performed |

`Restart=always` would re-enter immediately on exit 10 — pinning the
orchestrator against the very boundary that exists to hold it. So the
service never restarts, and the timer decides when the next batch
begins.

The unit declares `SuccessExitStatus=10 13`, because neither is a
fault in the run: 10 is the run stopping correctly at a human decision
boundary, and 13 is a worktree already busy. 11, 12 and 14 remain
failures and will show up as a failed unit, which is what they are.

## Why a separate guard program

A timer fires on a schedule and knows nothing about the lifecycle. On
its own it would re-activate the supervisor every interval while a
Human Gate sits open, or after the task reached
`READY_FOR_HUMAN_REVIEW` — burning a cycle artifact and a journal
entry each time for a condition only a person can clear.

`cryptolab-supervisor-guard` reads `.claude/runtime/status.json` and
exits 0 to proceed or 1 to skip. As an `ExecCondition=`, exit 1 makes
systemd skip the unit and record it as *successful*, so a blocked task
is quiet rather than noisy.

It proceeds only when the runtime state is `RUNNING`, the schema is
`3.1`, and no test requirement is recorded `FAIL`. Everything else —
including a missing, unreadable, or unrecognised status file — is a
skip. It fails closed in every direction.

Two properties are deliberate:

- **The guard never invokes the supervisor.** systemd composes
  `ExecCondition=` and `ExecStart=`, and systemd is configured by an
  operator. So the guard adds no new path to `.claude/bin/supervisor`
  and needs no permission boundary of its own — running it grants
  exactly what `statusctl show` already grants.
- **The guard is not the safety boundary.** The supervisor re-derives
  the same decision under the controller lock immediately before it
  acts, and that check is authoritative. The guard runs unlocked and a
  moment earlier, so at worst a race lets a cycle start that the
  supervisor then stops by itself.

## The reviewer credential

`EXTERNAL_REVIEW` needs `OPENAI_API_KEY` present when the service
starts. The unit references it and never stores it:

```ini
EnvironmentFile=-%h/.config/cryptolab/supervisor.env
```

That file lives outside the repository, outside every cycle artifact,
and outside runtime state. **Nothing in this repository creates, reads
or populates it.** The operator creates it, with permissions that
match what it holds:

```sh
install -d -m 700 ~/.config/cryptolab
install -m 600 /dev/null ~/.config/cryptolab/supervisor.env
# then edit it to contain, at minimum:
#   OPENAI_API_KEY=...
# optionally:
#   OPENAI_REVIEW_MODEL=...
#   OPENAI_BASE_URL=...
#   OPENAI_REVIEW_TIMEOUT=...
```

The leading `-` makes the file optional, so an absent credential does
not stop the unit from starting. It does not become an approval
either: the supervisor stops at `EXTERNAL_REVIEW` with exit 14 and
reports `FAILED`, exactly as the contract requires. A missing reviewer
is a fault in the run, never a pass.

## Installation is an operator action

Installing and enabling these units writes outside the worktree, into
`~/.config/systemd/user`, and changes what the machine does on its
own. That is an operating-system change under CLAUDE.md §11, so it is
**not** performed by the implementation agent and is not part of the
delivered change. A human runs it:

```sh
install -d -m 755 ~/.config/systemd/user

install -m 644 \
  deploy/systemd/cryptolab-supervisor.service \
  deploy/systemd/cryptolab-supervisor.timer \
  ~/.config/systemd/user/

systemctl --user daemon-reload

# Dry-run the decision first; this changes nothing.
deploy/systemd/cryptolab-supervisor-guard ; echo "exit=$?"

systemctl --user enable --now cryptolab-supervisor.timer
```

Before enabling the timer, check the paths. The units assume the
worktree is at `%h/projects/CryptoLab-DataHub-claude`; if it is not,
change `WorkingDirectory`, `ExecCondition` and `ExecStart` together so
they all name the same worktree.

To let the timer keep running when the operator is not logged in:

```sh
sudo loginctl enable-linger "$USER"
```

That is a separate operating-system change with its own consequences —
an unattended agent that keeps working after you log out — and it is
deliberately not bundled into the step above.

## Operating it

```sh
systemctl --user list-timers cryptolab-supervisor.timer
systemctl --user status cryptolab-supervisor.service
journalctl --user -u cryptolab-supervisor.service -n 200

# Run one batch now, outside the schedule.
systemctl --user start cryptolab-supervisor.service

# Stop the schedule. Does not interrupt a batch already running.
systemctl --user disable --now cryptolab-supervisor.timer
```

Guard decisions appear in the journal as one line per tick:

```text
SUPERVISOR_GUARD skip: runtime state is 'BLOCKED_HUMAN_DECISION'; \
    Human Gate HG-20260812-009 awaits a human decision
```

The supervisor's own outcome appears on its `SUPERVISOR_RESULT` line.

## Tuning

- `--max-cycles` in `ExecStart` bounds one batch. Smaller batches
  return to the guard more often; larger ones spend less time on
  start-up.
- `OnUnitInactiveSec` is measured from when the service last
  *stopped*, so it is a real gap between batches regardless of how
  long a batch takes. `OnUnitActiveSec` would not be.
- `TimeoutStartSec` bounds a wedged run. Being killed there is
  recoverable: `statusctl` transitions are atomic and lock-serialized,
  and the controller lock is an OS file lock the kernel releases when
  the process dies.
