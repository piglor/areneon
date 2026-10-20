"""The repository's gitleaks configuration (ADR-014 §10).

No key-shaped string is committed: each test builds one in memory. The rule's
regex is unit-tested here; the gitleaks binary, when it is on PATH (or named
by ``GITLEAKS``), must report a generated identity. The CI ``gitleaks`` job
runs the same check with the pinned binary.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / ".gitleaks.toml"
BECH32 = "QPZRY9X8GF2TVDW0S3JN54KHCE6MUA7L"
PREFIX = "AGE-SECRET-KEY-PQ-" + "1"
GITLEAKS = os.environ.get("GITLEAKS") or shutil.which("gitleaks")


def identity(length: int = 58) -> str:
    """A fresh key-shaped string; it is never written inside the repository."""
    return PREFIX + "".join(secrets.choice(BECH32) for _ in range(length))


def rule() -> dict[str, object]:
    rules = tomllib.loads(CONFIG.read_text(encoding="utf-8"))["rules"]
    assert [r["id"] for r in rules] == ["age-secret-key-pq"]
    return dict(rules[0])


def test_the_config_extends_the_default_rules_and_adds_the_pq_rule() -> None:
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["extend"] == {"useDefault": True}
    assert rule()["keywords"] == ["age-secret-key-pq-1"]


def test_the_regex_matches_an_identity_and_nothing_near_it() -> None:
    regex = re.compile(str(rule()["regex"]))
    key = identity()
    assert regex.fullmatch(key)
    assert regex.search(f"secret = '{key}'\n")
    # Negative controls: too short, a character outside the alphabet, the X25519
    # form (the default rule's), and the wrong case.
    assert not regex.search(identity(57))
    assert not regex.search(key[:-1] + "B")
    assert not regex.search("AGE-SECRET-KEY-1" + key[len(PREFIX) :])
    assert not regex.search(key.lower())
    # Every keyword is in the lower-cased text a prefilter would see.
    assert str(rule()["keywords"][0]) in key.lower()  # type: ignore[index]


@pytest.mark.skipif(GITLEAKS is None, reason="gitleaks is not installed")
class TestWithTheBinary:
    def scan(
        self, tmp_path: Path, config: Path | None, text: str, name: str = "notes.txt"
    ) -> list[dict[str, str]]:
        source = tmp_path / "scan"
        (source / name).parent.mkdir(parents=True)
        (source / name).write_text(text, encoding="utf-8")
        report = tmp_path / "report.json"
        command = [str(GITLEAKS), "dir", str(source), "--no-banner", "--report-format", "json"]
        command += ["--report-path", str(report)]
        if config is not None:
            command += ["--config", str(config)]
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        assert done.returncode in (0, 1), done.stderr
        findings: list[dict[str, str]] = json.loads(report.read_text(encoding="utf-8"))
        assert (done.returncode == 1) == bool(findings)
        return findings

    def test_a_generated_identity_is_reported(self, tmp_path: Path) -> None:
        findings = self.scan(tmp_path, CONFIG, f"identity: {identity()}\n")
        assert [f["RuleID"] for f in findings] == ["age-secret-key-pq"]

    def test_the_default_rules_alone_miss_it(self, tmp_path: Path) -> None:
        # Mutation control: without this repository's rule the identity passes (M7).
        plain = tmp_path / "default.toml"
        plain.write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
        assert self.scan(tmp_path, plain, f"identity: {identity()}\n") == []

    def test_the_default_rules_still_apply(self, tmp_path: Path) -> None:
        x25519 = "AGE-SECRET-KEY-" + "1" + "".join(secrets.choice(BECH32) for _ in range(58))
        findings = self.scan(tmp_path, CONFIG, f"identity: {x25519}\n")
        assert [f["RuleID"] for f in findings] == ["age-secret-key"]

    def test_text_near_the_pattern_is_not_reported(self, tmp_path: Path) -> None:
        assert self.scan(tmp_path, CONFIG, f"identity: {identity(57)}\n") == []

    def test_a_generic_key_line_elsewhere_is_still_reported(self, tmp_path: Path) -> None:
        text = f'{{"generation_key": "{secrets.token_hex(32)}"}}\n'
        findings = self.scan(tmp_path, CONFIG, text, "notes/keys.json")
        assert [f["RuleID"] for f in findings] == ["generic-api-key"]
