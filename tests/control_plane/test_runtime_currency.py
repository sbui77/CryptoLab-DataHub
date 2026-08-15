"""
Hardening 4.3 — the installed control plane is the one that runs.

Every other control-plane test resolves the version under test: the
staged payload while a delivery is pending, the installed file once it
has landed. That is right for a test of behaviour and blind to one
thing — whether the file a live run actually executes is the file that
was tested.

C1-FIX-C is what that blindness costs. The reviewer's response schema
was corrected, the suite went green at 786 passed, and the live call
still returned:

    Invalid schema for response_format 'external_review_verdict':
    findings.items.required is missing 'location'.

Nothing was wrong with the correction. `.claude/bin/supervisor` loads
`.claude/bin/openai_reviewer` by location, that file had not been
re-installed, and no test in the suite looked at it.

So these tests deliberately ignore the staging area and exercise the
installed artifacts:

    RC-01 the installed reviewer posts a schema the strict Structured
          Outputs subset accepts, with `location` required;
    RC-02 the installed control plane matches the delivery manifest,
          so a corrected program cannot sit in staging while a stale
          one serves live traffic;
    RC-03 the supervisor resolves the reviewer from its own directory,
          which is why RC-01 is asked of the installed file;
    RC-04 the installed settings deny every OPERATOR-RUN orchestrator
          to a Claude session, which is what section 23A asserts and
          what a live session is actually bound by.

A failure here is a deployment fact, not a code defect: the staged
payload is correct and has not been installed. The fix is to run the
approved installer, never to relax the assertion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from conftest import (
    PROJECT_ROOT,
    STAGING_DIR,
    installed_hash,
    staged_manifest,
)


REVIEWER_DESTINATION = ".claude/bin/openai_reviewer"

SUPERVISOR_DESTINATION = ".claude/bin/supervisor"


def strict_schema_errors(
    schema,
    path: tuple = (),
) -> list[str]:
    """
    Every strict-subset violation, in the provider's own wording.

    Duplicated from the reviewer's own tests on purpose: this module
    must be able to judge a schema produced by an installed program
    that may predate the checker entirely.
    """

    errors: list[str] = []

    if isinstance(schema, list):
        for index, entry in enumerate(schema):
            errors.extend(
                strict_schema_errors(
                    entry,
                    path + (index,),
                )
            )

        return errors

    if not isinstance(schema, dict):
        return errors

    properties = schema.get("properties")

    if isinstance(properties, dict):
        required = schema.get("required")

        if not isinstance(required, list):
            errors.append(
                f"in {path}, 'required' is missing."
            )

        else:
            missing = [
                key
                for key in properties
                if key not in required
            ]

            if missing:
                errors.append(
                    f"in {path}, 'required' must include every key "
                    "in properties. Missing "
                    + ", ".join(
                        repr(key)
                        for key in missing
                    )
                    + "."
                )

        if schema.get("additionalProperties") is not False:
            errors.append(
                f"in {path}, 'additionalProperties' must be false."
            )

    for key, value in schema.items():
        if key in {
            "enum",
            "const",
            "description",
            "required",
            "type",
        }:
            continue

        errors.extend(
            strict_schema_errors(
                value,
                path + (key,),
            )
        )

    return errors


def delivery_hint(
    destination: str,
) -> str:
    """
    Say what a failure means and what clears it.

    A deployment assertion that only says "False is not True" invites
    the reader to edit the test.
    """

    installed = installed_hash(destination)

    staged = None

    for entry in staged_manifest().get("files") or []:
        if entry.get("destination") == destination:
            staged = entry.get("sha256")

    return (
        f"\n  destination : {destination}"
        f"\n  installed   : {installed}"
        f"\n  manifest    : {staged}"
        "\n  The corrected program is staged and not installed. "
        "A human runs var/hardening-4.3/install to deliver it; "
        "nothing here may write a protected file."
    )


# ================================================================
# RC-01 — THE INSTALLED REVIEWER'S REQUEST BODY
# ================================================================


@pytest.fixture
def installed_reviewer(
    load_installed_module,
):
    return load_installed_module(
        REVIEWER_DESTINATION,
        "installed_openai_reviewer",
    )


def test_the_installed_reviewer_requires_location_in_findings_items(
    installed_reviewer,
):
    """
    RC-01: the exact live failure, asked of the exact live file.

    Built through the installed program's own request builder, so this
    is the body a real EXTERNAL_REVIEW would post.
    """

    config = installed_reviewer.Config(
        api_key="sk-test-not-a-real-key",
        model="gpt-5",
        base_url="https://api.example.invalid/v1",
        timeout=1.0,
    )

    _url, _headers, body = installed_reviewer.build_request(
        {
            "task": "currency-check",
        },
        config,
    )

    posted = json.loads(
        json.dumps(
            body,
            sort_keys=True,
        )
    )

    schema = posted["text"]["format"]["schema"]

    items = schema["properties"]["findings"]["items"]

    assert "location" in items["required"], (
        "The installed reviewer posts findings.items without "
        "'location' in required, which the API rejects."
        + delivery_hint(REVIEWER_DESTINATION)
    )

    assert strict_schema_errors(schema) == [], (
        "The installed reviewer posts a schema the strict "
        "Structured Outputs subset refuses."
        + delivery_hint(REVIEWER_DESTINATION)
    )

    # Strict structured output is the posture, and a stale artifact
    # must not be read as licence to loosen it.
    assert posted["text"]["format"]["strict"] is True
    assert posted["text"]["format"]["type"] == "json_schema"


# ================================================================
# RC-02 — INSTALLED MATCHES THE DELIVERY MANIFEST
# ================================================================


def test_every_installed_control_plane_file_matches_the_manifest():
    """
    RC-02: no corrected program may sit in staging unnoticed.

    The manifest is the description of what the control plane should
    be. While an entry's installed bytes differ from it, the live
    system is running something other than the version under test —
    which is the whole of C1-FIX-C, stated as a check that runs every
    time the suite does.
    """

    manifest = staged_manifest()

    if not manifest:
        pytest.skip(
            "No staged delivery manifest; nothing declares what the "
            "installed control plane should be."
        )

    stale = []

    for entry in manifest.get("files") or []:
        destination = str(
            entry.get("destination") or ""
        )

        expected = str(
            entry.get("sha256") or ""
        )

        source = STAGING_DIR / str(
            entry.get("source") or ""
        )

        if not destination or not source.is_file():
            continue

        # The manifest must itself be current, or the comparison
        # below would compare an installed file against a hash of
        # something no longer staged.
        found = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()

        assert found == expected, (
            f"The staged payload for {destination} does not match "
            "the manifest hash; regenerate the manifest."
        )

        if installed_hash(destination) != expected:
            stale.append(destination)

    assert not stale, (
        "The installed control plane is stale relative to the staged "
        "delivery: "
        + ", ".join(stale)
        + "".join(
            delivery_hint(destination)
            for destination in stale
        )
    )


# ================================================================
# RC-03 — WHY THE INSTALLED FILE IS THE ONE THAT MATTERS
# ================================================================


def test_the_supervisor_loads_the_reviewer_from_its_own_directory(
    load_installed_module,
):
    """
    RC-03: the runtime path, pinned.

    `load_sibling` resolves by location, so the reviewer a live run
    uses is `.claude/bin/openai_reviewer` and nothing else — no import
    path, no staging copy, no environment override. That is the reason
    RC-01 asks the installed file rather than the source.
    """

    supervisor = load_installed_module(
        SUPERVISOR_DESTINATION,
        "installed_supervisor",
    )

    resolved = Path(
        supervisor.reviewer.__file__
    ).resolve()

    assert resolved == (
        PROJECT_ROOT / REVIEWER_DESTINATION
    ).resolve()

    assert Path(
        supervisor.autopilot.__file__
    ).resolve() == (
        PROJECT_ROOT / ".claude" / "bin" / "autopilot"
    ).resolve()


# ================================================================
# RC-04 — THE CONTRACT'S DENY CLAIM, ASKED OF THE DENY LIST
# ================================================================


ORCHESTRATOR_PROGRAMS = (
    "autopilot",
    "supervisor",
    "openai_reviewer",
)

SETTINGS_DESTINATION = ".claude/settings.json"


def test_the_installed_settings_deny_every_orchestrator():
    """
    RC-04: section 23A's claim, checked against the file that enforces
    it.

    23A names three OPERATOR-RUN programs and states that
    `.claude/settings.json` denies them to a Claude session. Two of
    them were denied. `autopilot` was not, and it is the one that
    drives statusctl transitions, opens gates and records test results
    — so the program a worker could most usefully start was the one
    nothing stopped it from starting.

    The prose was the only place the rule existed, which is why this
    asks the deny list instead. The installed file is the one asked,
    for RC-02's reason: a session is bound by the settings on disk, not
    by the staged copy that will replace them.
    """

    settings = json.loads(
        (
            PROJECT_ROOT / SETTINGS_DESTINATION
        ).read_text()
    )

    deny = settings["permissions"]["deny"]

    missing = [
        program
        for program in ORCHESTRATOR_PROGRAMS
        if not {
            f"Bash(./.claude/bin/{program})",
            f"Bash(./.claude/bin/{program} *)",
        }
        <= set(deny)
    ]

    assert not missing, (
        "CLAUDE.md 23A says .claude/settings.json denies every "
        "OPERATOR-RUN orchestrator to a Claude session. It does not "
        "deny: "
        + ", ".join(missing)
        + ". Both the bare and the argument form are required; a "
        "session that may pass arguments may start the program."
        + delivery_hint(SETTINGS_DESTINATION)
    )
