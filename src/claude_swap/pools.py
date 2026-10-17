"""Pool launches: env-defined pools, the pool profile, and its watcher.

``cswap run --pool NAME`` runs a setup-token member in the pool's own config
dir, ``<backup_root>/pools/<name>/``, whose ``.credentials.json`` holds the
member the pool is on right now. A running Claude Code re-reads that file
before its next request, so ``cswap auto --pool NAME`` (which every launch
spawns for itself) moves live sessions to another member by rewriting it —
without touching the default login, which ``cswap auto`` keeps rotating on
its own. Setup tokens never rotate, so any number of concurrent launches can
share the profile. Settings and history are linked in from ``~/.claude``
like a session profile's.

A pool can also be defined by the environment: ``CLAUDE_POOL_TOKENS_<NAME>``
lists setup tokens (``[name:]token``, separated by spaces, commas or
newlines), and every launch makes the store's env-registered members of that
pool match it. That is how a sandbox whose secrets arrive as environment
variables (an Amp orb) gets its accounts with no setup step. ``CLAUDE_POOL``
names the pool when ``--pool`` does not.

This module must not import ``switcher`` or ``session`` (both import us).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from claude_swap import oauth
from claude_swap.exceptions import ValidationError
from claude_swap.locking import FileLock
from claude_swap.models import normalize_alias

if TYPE_CHECKING:
    from claude_swap.settings import AutoSwitchSettings
    from claude_swap.switcher import ClaudeAccountSwitcher

POOL_ENV = "CLAUDE_POOL"
TOKENS_ENV_PREFIX = "CLAUDE_POOL_TOKENS_"
DEFAULT_POOL = "default"

# Wrapper health checks (`--version`, `auth status`) on a short timeout: no
# usage-ranked pick, no watcher.
QUICK_ARGS = ("--version", "-v", "auth")

# Account records registered from CLAUDE_POOL_TOKENS_<NAME> carry this key
# (the pool's name), so a later list that drops them removes exactly those.
ENV_POOL_KEY = "envPool"

_LOG_MAX_BYTES = 1 << 20


def pools_dir(backup_dir: Path) -> Path:
    return backup_dir / "pools"


def profile_dir(backup_dir: Path, pool: str) -> Path:
    return pools_dir(backup_dir) / pool


def log_path(backup_dir: Path, pool: str) -> Path:
    return pools_dir(backup_dir) / f"{pool}.log"


def normalize_pool(name: str) -> str:
    try:
        return normalize_alias(name)
    except ValueError as e:
        raise ValidationError(str(e).replace("alias", "pool name")) from e


def tokens_env_var(pool: str) -> str:
    """``CLAUDE_POOL_TOKENS_<NAME>``: upper-cased, non-alphanumerics as ``_``."""
    return TOKENS_ENV_PREFIX + re.sub(r"[^A-Za-z0-9]", "_", pool).upper()


def without_pool_tokens(env: dict[str, str]) -> dict[str, str]:
    """Every pool's token list stays out of claude (and its tools' env)."""
    return {k: v for k, v in env.items() if not k.startswith(TOKENS_ENV_PREFIX)}


def parse_token_list(text: str) -> list[tuple[str | None, str]]:
    """``[name:]token`` entries separated by whitespace or commas.

    Setup tokens contain no colon, so ``name:token`` splits on the first one.
    Names become aliases, so they follow alias rules.
    """
    entries: list[tuple[str | None, str]] = []
    for raw in re.split(r"[\s,]+", text.strip()):
        if not raw:
            continue
        name, sep, token = raw.partition(":")
        if not sep:
            name, token = "", raw
        if not token:
            raise ValidationError(f"empty token after '{name}:'")
        if name:
            try:
                name = normalize_alias(name)
            except ValueError as e:
                raise ValidationError(f"token name: {e}") from e
        entries.append((name or None, token))
    return entries


def env_account_email(pool: str, name: str | None, token: str) -> str:
    """Stable label for an env-registered token: ``name@pool.token.local``,
    or a token digest when unnamed (so reordering the list moves nothing)."""
    local = name or "token-" + hashlib.sha256(token.encode()).hexdigest()[:8]
    domain = re.sub(r"[^a-z0-9.-]", "-", pool)
    return f"{local}@{domain}.token.local"


def sync_env_pool(
    switcher: ClaudeAccountSwitcher, pool: str, entries: list[tuple[str | None, str]]
) -> None:
    """Make the store's env-registered members of ``pool`` exactly ``entries``.

    New or changed tokens are (re-)registered, members the list dropped are
    removed with their accounts, and names become aliases when no other
    account holds them. Accounts added by hand are never touched. Cheap when
    nothing changed: one credentials read per entry.
    """
    wanted = {env_account_email(pool, name, token): (name, token) for name, token in entries}
    pools_dir(switcher.backup_dir).mkdir(parents=True, exist_ok=True)
    with FileLock(pools_dir(switcher.backup_dir) / ".sync.lock", timeout=60):
        records = (switcher._get_sequence_data() or {}).get("accounts", {})
        by_email = {
            r.get("email"): num for num, r in records.items() if not r.get("organizationUuid")
        }
        for email, (_name, token) in wanted.items():
            num = by_email.get(email)
            if num is not None and _stored_access_token(switcher, num, email) == token:
                continue
            switcher.add_account_from_token(token, email=email, assume_yes=True)

        for num, record in list(records.items()):
            if record.get(ENV_POOL_KEY) == pool and record.get("email") not in wanted:
                switcher.remove_account(num, assume_yes=True)

        data = switcher._get_sequence_data() or {}
        changed = False
        aliases: list[tuple[str, str]] = []
        for num, record in data.get("accounts", {}).items():
            email = record.get("email")
            if email not in wanted or record.get("organizationUuid"):
                continue
            pools = record.get("pools", [])
            if record.get(ENV_POOL_KEY) != pool or pool not in pools:
                record[ENV_POOL_KEY] = pool
                record["pools"] = [*[p for p in pools if p != pool], pool]
                changed = True
            name = wanted[email][0]
            if name and record.get("alias") != name and not switcher._find_account_by_alias(name):
                aliases.append((num, name))
        if changed:
            switcher._write_json(switcher.sequence_file, data)
        for num, name in aliases:
            switcher.set_alias(num, name)


def _stored_access_token(switcher: ClaudeAccountSwitcher, num: str, email: str) -> str | None:
    creds = switcher.read_account_credentials(num, email)
    data = oauth.extract_oauth_data(creds) if creds else None
    return data.get("accessToken") if data else None


# -- the pool profile ------------------------------------------------------


def profile_access_token(profile: Path) -> str | None:
    try:
        creds = (profile / ".credentials.json").read_text(encoding="utf-8")
    except OSError:
        return None
    data = oauth.extract_oauth_data(creds)
    return data.get("accessToken") if data else None


def write_profile_credentials(profile: Path, credentials: str) -> None:
    """Point the pool profile at a member: one atomic replace, which a
    running Claude Code picks up before its next request."""
    from claude_swap.session import delete_macos_keychain_entry

    profile.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        os.chmod(profile, 0o700)
    # Claude reads its hashed keychain entry before the plaintext file.
    delete_macos_keychain_entry(profile)
    path = profile / ".credentials.json"
    tmp = path.with_name(f".credentials.json.{os.getpid()}.tmp")
    tmp.write_text(credentials, encoding="utf-8")
    if sys.platform != "win32":
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def seed_profile_config(profile: Path, theme: str | None) -> None:
    """Skip onboarding (claude shows it while ``theme`` or
    ``hasCompletedOnboarding`` is missing); keep everything else claude wrote."""
    path = profile / ".claude.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    if data.get("hasCompletedOnboarding") and data.get("theme"):
        return
    data["hasCompletedOnboarding"] = True
    data.setdefault("theme", theme or "dark")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if sys.platform != "win32":
        os.chmod(path, 0o600)


def current_member(
    switcher: ClaudeAccountSwitcher, profile: Path, members: list[str]
) -> str | None:
    """The member whose token the profile holds, if any."""
    token = profile_access_token(profile)
    if not token:
        return None
    for num in members:
        _, email, _ = switcher.resolve_account(num)
        if _stored_access_token(switcher, num, email) == token:
            return num
    return None


# -- choosing a member -----------------------------------------------------


def best_member(headroom: dict[str, float | None]) -> str | None:
    """Most headroom; measured beats unknown; ties keep the given order."""
    if not headroom:
        return None
    return max(headroom, key=lambda n: (headroom[n] is not None, headroom[n] or 0.0))


def switch_target(
    current: str | None,
    headroom: dict[str, float | None],
    threshold: float,
    hysteresis: float,
    *,
    at_launch: bool,
) -> str | None:
    """The member to move to, or None to stay on ``current``.

    Without a current member, the best one. An exhausted current member
    moves to any member with headroom left. Otherwise a move needs a target
    with at least ``hysteresis`` points more headroom, and — except at
    launch, where the pick is free — the current member at ``threshold``.
    An unmeasured current member stays: no move is provably better.
    """
    if current not in headroom:
        return best_member(headroom)
    others = {n: h for n, h in headroom.items() if n != current}
    target = best_member(others)
    cur, best = headroom[current], others.get(target) if target else None
    if target is None or cur is None or best is None:
        return None
    if cur <= 0:
        return target if best > 0 else None
    if not at_launch and 100.0 - cur < threshold:
        return None
    return target if best - cur >= hysteresis else None


def headroom_by_member(
    usage: dict[str, dict | str | None], models: tuple[str, ...]
) -> dict[str, float | None]:
    """Headroom per member; a sentinel (dead token, no credentials) is 0."""
    return {
        num: (
            oauth.account_headroom(value, models)
            if isinstance(value, dict)
            else (0.0 if isinstance(value, str) else None)
        )
        for num, value in usage.items()
    }


# -- the watcher (`cswap auto --pool`) --------------------------------------


def spawn_watcher(pool: str, pid: int, env: dict[str, str], backup_dir: Path) -> None:
    """Start ``cswap auto --pool`` for a launch, logging to ``pools/<name>.log``.

    It outlives this process's exec into claude and exits when ``pid`` (the
    claude about to be exec'd, or this process on Windows) does. Its stdio is
    detached so it never holds a wrapper's pipes open.
    """
    log = log_path(backup_dir, pool)
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        if log.stat().st_size > _LOG_MAX_BYTES:
            log.unlink()
    except OSError:
        pass
    with open(log, "a", encoding="utf-8") as out:
        subprocess.Popen(
            [sys.executable, "-m", "claude_swap", "auto", "--pool", pool,
             "--pid", str(pid), "--json"],
            stdin=subprocess.DEVNULL, stdout=out, stderr=out, env=env,
            close_fds=True, cwd=str(backup_dir),
        )


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # type: ignore[attr-defined]
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))  # type: ignore[attr-defined]
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PoolWatcher:
    """Moves a pool profile to another member before the current one runs out.

    One tick: collect the setup-token members' usage (store-backed, so every
    surface shares one cadence and backoff), decide with
    :func:`switch_target`, re-measure the target before committing, then
    rewrite the profile's credentials. Concurrent launches each run a
    watcher; a lock lets one tick at a time and the rest stand by.
    """

    def __init__(
        self,
        switcher: ClaudeAccountSwitcher,
        pool: str,
        settings: AutoSwitchSettings,
        emit: Callable[[dict], None],
        *,
        dry_run: bool = False,
    ):
        from claude_swap.settings import parse_model_names

        self.switcher = switcher
        self.pool = pool
        self.settings = settings
        self.models = parse_model_names(settings.model)
        self.emit = emit
        self.dry_run = dry_run
        self.profile = profile_dir(switcher.backup_dir, pool)
        self._lock = FileLock(pools_dir(switcher.backup_dir) / f".{pool}.watch.lock")
        self._holding = False

    def tick(self) -> int:
        """0 switched, 1 error, 2 no action (``cswap auto --once`` codes)."""
        members = self.switcher.pool_setup_token_members(self.pool)
        if not members:
            self.emit({"event": "error", "message": f"pool '{self.pool}' has no setup-token members"})
            return 1
        current = current_member(self.switcher, self.profile, members)
        usage = self.switcher.pool_usage(members, fetch=set(members), scheduled=True)
        headroom = headroom_by_member(usage, self.models)
        self.emit({"event": "poll", "current": current, "headroom": headroom})
        args = (self.settings.threshold, self.settings.hysteresis_pct)
        target = switch_target(current, headroom, *args, at_launch=False)
        if target is None:
            return 2
        # Commit only on a fresh measurement of the target.
        usage.update(self.switcher.pool_usage([target], fetch={target}))
        headroom = headroom_by_member(usage, self.models)
        if switch_target(current, headroom, *args, at_launch=False) != target:
            return 2
        if not self.dry_run:
            _, email, _ = self.switcher.resolve_account(target)
            write_profile_credentials(
                self.profile, self.switcher.read_account_credentials(target, email)
            )
        self.emit({
            "event": "switch", "from": current, "to": target,
            "headroom": {current: headroom.get(current), target: headroom[target]},
            "dryRun": self.dry_run,
        })
        return 0

    def run(self, pid: int | None, once: bool = False) -> int:
        """Tick every ``interval_seconds`` until ``pid`` exits (forever without
        one). Only the lock holder ticks; the others take over when it exits."""
        if once:
            return self.tick()
        next_tick = 0.0
        while pid is None or pid_alive(pid):
            if not self._holding:
                self._holding = self._lock.acquire(timeout=0)
            now = time.monotonic()
            if self._holding and now >= next_tick:
                try:
                    self.tick()
                except Exception as e:  # noqa: BLE001 - keep watching; the next tick may succeed
                    self.emit({"event": "error", "message": repr(e)})
                next_tick = now + self.settings.interval_seconds
            time.sleep(5)
        if self._holding:
            self._lock.release()
        return 0
