"""Pools (`cswap pool`, `cswap run --pool`) and setup-token usage probing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth
from claude_swap.exceptions import AccountNotFoundError
from claude_swap.models import Platform
from claude_swap.oauth import probe_setup_token_usage  # real one, pre-autouse-stub
from claude_swap.session import SessionManager
from claude_swap.switcher import ClaudeAccountSwitcher

# Shape captured live from Claude Code 2.1.296 (`claude -p --output-format stream-json`).
RATE_LIMIT_EVENT = {
    "type": "rate_limit_event",
    "rate_limit_info": {
        "status": "allowed", "resetsAt": 1792015800, "rateLimitType": "five_hour",
        "unifiedWindows": {
            "five_hour": {"utilization": 0.67, "resetsAt": 1792015800},
            "seven_day": {"utilization": 0.1, "resetsAt": 1792458000},
        },
    },
}
SETUP_TOKEN_CREDS = json.dumps(
    {"claudeAiOauth": {"accessToken": "sk-ant-oat01-x", "scopes": ["user:inference"]}}
)


class TestRateLimitParsing:
    def test_maps_fractions_and_epochs_to_usage_api_shape(self):
        out = "\n".join(["not json", json.dumps({"type": "assistant"}), json.dumps(RATE_LIMIT_EVENT)])
        data, status = oauth.parse_rate_limit_events(out)
        assert status is None
        assert data["five_hour"]["utilization"] == 67.0
        assert data["seven_day"]["utilization"] == 10.0
        assert data["five_hour"]["resets_at"].startswith("2026-10-14T")
        assert oauth.account_headroom(oauth.build_usage_result(data)) == 33.0

    def test_revoked_token_reports_retry_status(self):
        out = json.dumps({"type": "system", "subtype": "api_retry", "error_status": 401})
        assert oauth.parse_rate_limit_events(out) == (None, 401)


class TestProbe:
    def _run(self, stdout: str):
        done = subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")
        with patch("claude_swap.oauth.shutil.which", return_value="/bin/claude"), \
             patch("claude_swap.oauth.subprocess.run", return_value=done) as run:
            return probe_setup_token_usage("sk-ant-oat01-x"), run

    def test_success_and_isolated_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/somewhere")
        outcome, run = self._run(json.dumps(RATE_LIMIT_EVENT))
        assert outcome.error is None and outcome.usage["five_hour"]["pct"] == 67.0
        env = run.call_args.kwargs["env"]
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-x"
        assert "ANTHROPIC_BASE_URL" not in env
        assert env["CLAUDE_CONFIG_DIR"] != "/somewhere"

    def test_no_event_is_an_error(self):
        outcome, _ = self._run("")
        assert outcome.usage is None and outcome.error == "no-rate-limit-event"

    def test_setup_tokens_skip_the_usage_endpoint(self, monkeypatch):
        monkeypatch.setattr(
            oauth, "probe_setup_token_usage",
            lambda t: oauth.UsageOutcome({"five_hour": {"pct": 1.0}}),
        )
        with patch.object(oauth, "request_usage_data") as endpoint:
            got = oauth.try_fetch_usage_for_account("1", "e", SETUP_TOKEN_CREDS, False)
        endpoint.assert_not_called()
        assert got.usage == {"five_hour": {"pct": 1.0}}

    def test_full_logins_are_not_setup_tokens(self):
        assert not oauth.is_setup_token({"accessToken": "a", "refreshToken": "r",
                                          "scopes": ["user:inference"]})


class TestPools:
    def _switcher(self, temp_home: Path) -> ClaudeAccountSwitcher:
        s = ClaudeAccountSwitcher()
        s.platform = Platform.LINUX
        s._setup_directories()
        s._init_sequence_file()
        for num in (1, 2, 3):
            email = f"u{num}@example.com"
            s._write_account_credentials(str(num), email, SETUP_TOKEN_CREDS)
            s._write_account_config(str(num), email, json.dumps(
                {"oauthAccount": {"emailAddress": email}}))
            data = s._get_sequence_data()
            data["accounts"][str(num)] = {"email": email, "organizationUuid": ""}
            data["sequence"].append(num)
            s._write_json(s.sequence_file, data)
        return s

    def test_membership_add_remove_list(self, temp_home: Path):
        s = self._switcher(temp_home)
        s.set_pool_membership("CVP", ["3", "1"], member=True)
        s.set_pool_membership("other", ["1"], member=True)
        assert s.pools() == {"cvp": ["1", "3"], "other": ["1"]}
        s.set_pool_membership("cvp", ["1"], member=False)
        assert s.pools() == {"cvp": ["3"], "other": ["1"]}

    def test_pick_prefers_headroom_then_known_then_order(self, temp_home: Path):
        s = self._switcher(temp_home)
        s.set_pool_membership("cvp", ["1", "2", "3"], member=True)
        u = lambda pct: {"five_hour": {"pct": pct}, "seven_day": {"pct": 0.0}}
        with patch.object(s, "_usage_by_account", return_value={"1": u(90), "2": None, "3": u(20)}):
            assert s.pick_pool_account("cvp") == "3"
        with patch.object(s, "_usage_by_account", return_value={"1": u(90), "2": None, "3": None}):
            assert s.pick_pool_account("cvp") == "1"  # unknown ranks below measured
        s.set_account_disabled("1", True)  # reserved for pools, not skipped
        with patch.object(s, "_usage_by_account", return_value={"1": u(10), "3": u(20)}):
            assert s.pick_pool_account("cvp") == "1"
        with pytest.raises(AccountNotFoundError):
            s.pick_pool_account("nope")

    def test_run_pool_execs_setup_token_via_env(self, temp_home: Path, monkeypatch):
        s = self._switcher(temp_home)
        s.set_pool_membership("cvp", ["2"], member=True)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "leak")
        m = SessionManager(s)
        with patch("claude_swap.session.shutil.which", return_value="/bin/claude"), \
             patch.object(s, "_usage_by_account", return_value={}), \
             patch.object(m, "_exec", side_effect=SystemExit) as exec_, \
             pytest.raises(SystemExit):
            m.run_pool("cvp", ["--resume"])
        _bin, args = exec_.call_args.args
        env = exec_.call_args.kwargs["env"]
        assert args == ["--resume"]
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-x"
        assert "ANTHROPIC_API_KEY" not in env and "CLAUDE_CONFIG_DIR" not in env
