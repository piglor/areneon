"""The owner-session guard and the private output root (Python twins of the TypeScript ones)."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import termios
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from areneon_ml.private.owner_session import (
    OWNER_SESSION_REFUSALS,
    OwnerSession,
    OwnerSessionError,
    agent_refusal_code,
    assert_guarded_child,
    assert_owner_session,
    open_controlling_terminal,
)

REPO = Path(__file__).resolve().parents[3]
SHARED = json.loads((REPO / "fixtures/private-guard/refusals.json").read_text(encoding="utf-8"))
PROBE = """
import json, sys
from areneon_ml.private.owner_session import (
    OwnerSessionError, assert_guarded_child, assert_owner_session)
mode, out = sys.argv[1:3]
try:
    (assert_guarded_child if mode == "child" else assert_owner_session)()
    result = {"ok": True}
except OwnerSessionError as error:
    result = {"ok": False, "code": error.code, "message": str(error)}
open(out, "w").write(json.dumps(result))
"""


def session(
    env: Mapping[str, str] | None = None,
    flags: tuple[bool, bool, bool] = (True, True, True),
    opened: list[int] | None = None,
    fail_open: bool = False,
) -> OwnerSession:
    def open_terminal() -> None:
        if fail_open:
            raise OwnerSessionError("no_controlling_terminal")
        if opened is not None:
            opened.append(1)

    return OwnerSession(env=env or {}, is_terminal=lambda: flags, open_terminal=open_terminal)


def code_of(run: Callable[[], object]) -> str:
    try:
        run()
    except OwnerSessionError as error:
        caught = error
    else:
        return "ok"
    assert str(caught) == OWNER_SESSION_REFUSALS[caught.code]
    return caught.code


def test_words_equal_the_shared_fixture() -> None:
    assert SHARED["owner_session"] == OWNER_SESSION_REFUSALS


def test_passes_a_terminal_session_and_opens_the_terminal_once() -> None:
    opened: list[int] = []
    assert code_of(lambda: assert_owner_session(session(opened=opened))) == "ok"
    assert opened == [1]


@pytest.mark.parametrize("name", ["CLAUDECODE", "AI_AGENT"])
def test_refuses_an_agent_before_anything_else(name: str) -> None:
    opened: list[int] = []
    guard = session({name: "1"}, flags=(False, False, False), opened=opened)
    assert code_of(lambda: assert_owner_session(guard)) == "agent_session"
    assert opened == []
    assert agent_refusal_code({"CLAUDECODE": ""}) is None


@pytest.mark.parametrize("flags", [(False, True, True), (True, False, True), (True, True, False)])
def test_refuses_each_piped_stdio(flags: tuple[bool, bool, bool]) -> None:
    opened: list[int] = []
    assert code_of(lambda: assert_owner_session(session(flags=flags, opened=opened))) == (
        "not_a_terminal"
    )
    assert opened == []


def test_refuses_without_a_controlling_terminal() -> None:
    assert code_of(lambda: assert_owner_session(session(fail_open=True))) == (
        "no_controlling_terminal"
    )


def test_guarded_child_checks_the_environment_only() -> None:
    assert code_of(lambda: assert_guarded_child({})) == "ok"
    assert code_of(lambda: assert_guarded_child({"CLAUDECODE": "1"})) == "agent_session"
    assert code_of(lambda: assert_guarded_child({"AI_AGENT": "1"})) == "agent_session"


def test_the_defaults_read_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDECODE", "1")
    assert code_of(assert_guarded_child) == "agent_session"
    assert code_of(assert_owner_session) == "agent_session"
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.delenv("AI_AGENT", raising=False)
    assert code_of(assert_guarded_child) == "ok"
    # Under pytest the stdio are captured, so the default session refuses.
    assert code_of(assert_owner_session) in {"not_a_terminal", "no_controlling_terminal"}


def test_open_controlling_terminal_opens_or_says_the_fixed_words() -> None:
    try:
        open_controlling_terminal()
    except OwnerSessionError as error:
        caught = error
    else:
        return
    assert caught.code == "no_controlling_terminal"


def _clean_env(extra: Mapping[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "AI_AGENT")}
    env.update(extra)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    return env


def _run_probe(
    tmp_path: Path, mode: str, mask: str, ctty: bool, extra: Mapping[str, str] | None = None
) -> dict[str, object]:
    master, slave = os.openpty()
    out = tmp_path / f"out-{mode}-{mask}-{ctty}-{len(list(tmp_path.iterdir()))}.json"
    std = [slave if flag == "t" else subprocess.PIPE for flag in mask]

    def set_ctty() -> None:
        if ctty:
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", PROBE, mode, str(out)],
            stdin=std[0],
            stdout=std[1],
            stderr=std[2],
            env=_clean_env(extra or {}),
            start_new_session=True,
            preexec_fn=set_ctty,  # noqa: PLW1509 (no threads in the child)
        )
        proc.communicate(timeout=60)
    finally:
        os.close(master)
        os.close(slave)
    result: dict[str, object] = json.loads(out.read_text(encoding="utf-8"))
    return result


def test_real_child_passes_with_a_controlling_terminal(tmp_path: Path) -> None:
    assert _run_probe(tmp_path, "top", "ttt", True) == {"ok": True}


@pytest.mark.parametrize("name", ["CLAUDECODE", "AI_AGENT"])
def test_real_child_refuses_an_agent_even_on_a_terminal(tmp_path: Path, name: str) -> None:
    result = _run_probe(tmp_path, "top", "ttt", True, {name: "1"})
    assert result["code"] == "agent_session"


@pytest.mark.parametrize("mask", ["ptt", "tpt", "ttp", "ppp"])
def test_real_child_refuses_a_pipe(tmp_path: Path, mask: str) -> None:
    assert _run_probe(tmp_path, "top", mask, False)["code"] == "not_a_terminal"


def test_real_child_refuses_without_a_controlling_terminal(tmp_path: Path) -> None:
    assert _run_probe(tmp_path, "top", "ttt", False)["code"] == "no_controlling_terminal"


def test_real_guarded_child_with_pipes(tmp_path: Path) -> None:
    assert _run_probe(tmp_path, "child", "ppp", False) == {"ok": True}
    assert _run_probe(tmp_path, "child", "ppp", False, {"CLAUDECODE": "1"})["code"] == (
        "agent_session"
    )


def test_open_controlling_terminal_closes_what_it_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = os.open
    seen: list[int] = []

    def fake_open(path: str, flags: int) -> int:
        assert path == "/dev/tty"
        seen.append(real_open(os.devnull, os.O_WRONLY))
        return seen[-1]

    monkeypatch.setattr(os, "open", fake_open)
    open_controlling_terminal()
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(seen[0])


def test_a_missing_or_closed_stdio_stream_is_not_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Closed:
        def isatty(self) -> bool:
            raise ValueError("I/O operation on closed file")

    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.delenv("AI_AGENT", raising=False)
    monkeypatch.setattr(sys, "stdin", None)
    assert code_of(assert_owner_session) == "not_a_terminal"
    monkeypatch.setattr(sys, "stdin", Closed())
    assert code_of(assert_owner_session) == "not_a_terminal"
