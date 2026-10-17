"""Pools (`cswap pool`, `cswap run --pool`, `cswap auto --pool`, env-defined
pools, the `claude` shim) and setup-token usage probing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth, pools
from claude_swap.exceptions import AccountNotFoundError, ValidationError
from claude_swap.models import Platform
# The real ones, imported before conftest's autouse stubs replace them.
from claude_swap.oauth import probe_setup_token_usage, setup_token_has_profile_scope
from claude_swap.session import SessionManager
from claude_swap.settings import AutoSwitchSettings
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

    def test_profile_scope_is_asked_once_and_routes_to_the_endpoint(self, tmp_path, monkeypatch):
        import io
        import urllib.error

        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        refused = urllib.error.HTTPError("u", 403, "scope", {}, io.BytesIO(b""))
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=refused) as call:
            assert setup_token_has_profile_scope("inference-only") is False
            assert setup_token_has_profile_scope("inference-only") is False
        assert call.call_count == 1
        with patch("claude_swap.oauth.urllib.request.urlopen", side_effect=OSError("down")) as call:
            assert setup_token_has_profile_scope("t2") is False
            assert setup_token_has_profile_scope("t2") is False
        assert call.call_count == 2  # undecided: not cached
        with patch("claude_swap.oauth.urllib.request.urlopen") as call:
            assert setup_token_has_profile_scope("t2") is True

        monkeypatch.setattr(oauth, "setup_token_has_profile_scope", lambda t: True)
        monkeypatch.setattr(oauth, "probe_setup_token_usage", lambda t: pytest.fail("probed"))
        with patch.object(oauth, "request_usage_data", return_value={
            "five_hour": {"utilization": 12.0}, "seven_day": {"utilization": 3.0},
        }):
            got = oauth.try_fetch_usage_for_account("1", "e", SETUP_TOKEN_CREDS, False)
        assert got.usage["five_hour"]["pct"] == 12.0

    def test_full_logins_are_not_setup_tokens(self):
        assert not oauth.is_setup_token({"accessToken": "a", "refreshToken": "r",
                                          "scopes": ["user:inference"]})


@pytest.fixture(autouse=True)
def isolated_pool_env(monkeypatch):
    """No CLAUDE_POOL* from the machine running the tests; and run_pool's
    in-process scrub of CLAUDE_POOL_TOKENS_* is undone after each test."""
    with patch.dict(os.environ):
        for key in [k for k in os.environ if k.startswith("CLAUDE_POOL")]:
            del os.environ[key]
        yield


def _creds(token: str) -> str:
    return json.dumps({"claudeAiOauth": {"accessToken": token, "scopes": ["user:inference"]}})


def _u(pct: float) -> dict:
    return {"five_hour": {"pct": pct}, "seven_day": {"pct": 0.0}}


class TestPools:
    def _switcher(self, temp_home: Path) -> ClaudeAccountSwitcher:
        s = ClaudeAccountSwitcher()
        s.platform = Platform.LINUX
        s._setup_directories()
        s._init_sequence_file()
        for num in (1, 2, 3):
            email = f"u{num}@example.com"
            s._write_account_credentials(str(num), email, _creds(f"sk-ant-oat01-{num}"))
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
        with patch.object(s, "pool_usage", return_value={"1": _u(90), "2": None, "3": _u(20)}):
            assert s.pick_pool_account("cvp") == "3"
        with patch.object(s, "pool_usage", return_value={"1": _u(90), "2": None, "3": None}):
            assert s.pick_pool_account("cvp") == "1"  # unknown ranks below measured
        s.set_account_disabled("1", True)  # reserved for pools, not skipped
        with patch.object(s, "pool_usage", return_value={"1": _u(10), "3": _u(20)}):
            assert s.pick_pool_account("cvp") == "1"
        with pytest.raises(AccountNotFoundError):
            s.pick_pool_account("nope")

    def test_pool_usage_only_fetches_members(self, temp_home: Path):
        s = self._switcher(temp_home)
        with patch.object(s, "_collect_usage_entries", return_value={}) as collect:
            s.pool_usage(["2", "3"])
        assert [i[0] for i in collect.call_args.args[0]] == [2, 3]

    def _launch(self, s, manager, args, usage):
        with patch("claude_swap.session.shutil.which", return_value="/bin/claude"), \
             patch.object(s, "pool_usage", return_value=usage), \
             patch.object(pools, "spawn_watcher") as spawn, \
             patch.object(manager, "_exec", side_effect=SystemExit) as exec_, \
             pytest.raises(SystemExit):
            manager.run_pool("cvp", args)
        return exec_.call_args, spawn

    def test_run_pool_launches_in_the_pool_profile(self, temp_home: Path, monkeypatch):
        s = self._switcher(temp_home)
        s.set_pool_membership("cvp", ["2", "3"], member=True)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "leak")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "amp-token")
        monkeypatch.setenv("CLAUDE_POOL_TOKENS_OTHER", "x:sk-ant-oat01-secret")
        m = SessionManager(s)
        call, spawn = self._launch(s, m, ["--resume"], {"2": _u(50), "3": _u(5)})
        (_bin, args), env = call.args, call.kwargs["env"]
        profile = pools.profile_dir(s.backup_dir, "cvp")
        assert args == ["--resume"]
        assert env["CLAUDE_CONFIG_DIR"] == str(profile)
        assert pools.profile_access_token(profile) == "sk-ant-oat01-3"
        assert not {"ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_POOL_TOKENS_OTHER"} & set(env)
        assert spawn.call_count == 1 and "CLAUDE_POOL_TOKENS_OTHER" not in spawn.call_args.args[2]

        # Relaunch: 3 now trails 2 by less than the hysteresis, so the pool
        # stays put (live sessions share the profile); by more, it moves.
        call, _ = self._launch(s, m, [], {"2": _u(30), "3": _u(35)})
        assert pools.profile_access_token(profile) == "sk-ant-oat01-3"
        call, _ = self._launch(s, m, [], {"2": _u(30), "3": _u(45)})
        assert pools.profile_access_token(profile) == "sk-ant-oat01-2"

        # Health checks: no ranking, no watcher.
        with patch.object(s, "pool_headroom", side_effect=AssertionError):
            _, spawn = self._launch(s, m, ["--version"], {})
        spawn.assert_not_called()

    def test_implicit_pool_without_members_runs_plain_claude(self, temp_home: Path, monkeypatch):
        s = self._switcher(temp_home)
        monkeypatch.delenv("CLAUDE_POOL", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "amp-token")
        m = SessionManager(s)
        with patch("claude_swap.session.shutil.which", return_value="/bin/claude"), \
             patch.object(m, "_exec", side_effect=SystemExit) as exec_, \
             pytest.raises(SystemExit):
            m.run_pool(None, ["-p", "hi"])
        assert exec_.call_args.kwargs["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "amp-token"
        monkeypatch.setenv("CLAUDE_POOL", "default")  # named: must exist
        with patch("claude_swap.session.shutil.which", return_value="/bin/claude"), \
             pytest.raises(AccountNotFoundError):
            m.run_pool(None, [])


class TestSwitchTarget:
    T, H = 90.0, 10.0

    def target(self, current, headroom, at_launch=False):
        return pools.switch_target(current, headroom, self.T, self.H, at_launch=at_launch)

    def test_watcher_waits_for_the_threshold(self):
        assert self.target("1", {"1": 15.0, "2": 90.0}) is None  # 85% used
        assert self.target("1", {"1": 10.0, "2": 90.0}) == "2"  # 90% used

    def test_launch_moves_on_hysteresis_alone(self):
        assert self.target("1", {"1": 70.0, "2": 79.0}, at_launch=True) is None
        assert self.target("1", {"1": 70.0, "2": 80.0}, at_launch=True) == "2"

    def test_needs_a_clear_gain_unless_exhausted(self):
        assert self.target("1", {"1": 5.0, "2": 14.0}) is None
        assert self.target("1", {"1": 0.0, "2": 3.0}) == "2"
        assert self.target("1", {"1": 0.0, "2": 0.0}) is None

    def test_unknowns(self):
        assert self.target("1", {"1": None, "2": 90.0}) is None
        assert self.target("1", {"1": 5.0, "2": None}) is None
        assert self.target(None, {"1": None, "2": 40.0, "3": 60.0}) == "3"

    def test_sentinel_usage_counts_as_exhausted(self):
        assert pools.headroom_by_member({"1": "token expired", "2": None}, ()) == {"1": 0.0, "2": None}


class TestWatcher:
    def test_switches_on_a_fresh_measurement_only(self, temp_home: Path):
        s = TestPools()._switcher(temp_home)
        s.set_pool_membership("cvp", ["1", "2"], member=True)
        profile = pools.profile_dir(s.backup_dir, "cvp")
        pools.write_profile_credentials(profile, _creds("sk-ant-oat01-1"))
        events = []
        w = pools.PoolWatcher(s, "cvp", AutoSwitchSettings(), events.append)
        # The re-measured target turned out no better: stay.
        with patch.object(s, "pool_usage", side_effect=[{"1": _u(95), "2": _u(20)}, {"2": _u(92)}]):
            assert w.tick() == 2
        assert pools.profile_access_token(profile) == "sk-ant-oat01-1"
        with patch.object(s, "pool_usage", side_effect=[{"1": _u(95), "2": _u(20)}, {"2": _u(25)}]):
            assert w.tick() == 0
        assert pools.profile_access_token(profile) == "sk-ant-oat01-2"
        assert events[-1]["event"] == "switch" and events[-1]["to"] == "2"


class TestEnvPools:
    def test_parse(self):
        text = " a:sk-ant-oat01-A,\nsk-ant-oat01-B  B.x:sk-ant-oat01-C "
        assert pools.parse_token_list(text) == [
            ("a", "sk-ant-oat01-A"), (None, "sk-ant-oat01-B"), ("b.x", "sk-ant-oat01-C")
        ]
        with pytest.raises(ValidationError):
            pools.parse_token_list("a:")
        assert pools.tokens_env_var("my-pool") == "CLAUDE_POOL_TOKENS_MY_POOL"

    def _token_of(self, s, email):
        num, _, _ = s.resolve_account(email)
        return json.loads(s.read_account_credentials(num, email))["claudeAiOauth"]["accessToken"]

    def test_sync_registers_updates_and_prunes_only_its_own(self, temp_home: Path):
        s = TestPools()._switcher(temp_home)
        s.set_pool_membership("cyber", ["1"], member=True)  # added by hand
        pools.sync_env_pool(s, "cyber", pools.parse_token_list("samir:T1 T2"))
        unnamed = pools.env_account_email("cyber", None, "T2")
        assert s.pools()["cyber"] == ["1", "4", "5"]
        assert s.resolve_account("samir")[1] == "samir@cyber.token.local"
        assert self._token_of(s, unnamed) == "T2"

        pools.sync_env_pool(s, "cyber", pools.parse_token_list("samir:T1b"))
        assert s.pools()["cyber"] == ["1", "4"]  # T2's account removed, 1 kept
        assert self._token_of(s, "samir@cyber.token.local") == "T1b"
        with pytest.raises(AccountNotFoundError):
            s.resolve_account(unnamed)


class TestShim:
    def test_drops_itself_from_path(self, tmp_path: Path, monkeypatch):
        from claude_swap import cli

        real, shim = tmp_path / "real", tmp_path / "shim"
        real.mkdir(); shim.mkdir()
        (real / "claude").write_text("")
        target = tmp_path / "cswap"
        target.write_text("")
        (shim / "claude").symlink_to(target)
        monkeypatch.setenv("PATH", f"{shim}:{real}")
        monkeypatch.setattr(sys, "argv", [str(shim / "claude"), "--resume"])
        with patch.object(cli, "_run_command") as run:
            cli.main()
        run.assert_called_once_with(["--pool", "--", "--resume"])
        assert os.environ["PATH"] == str(real)
