"""Claude Code CLI bridge: run the user's own `claude` headlessly via `claude -p`.

Where codex writes its answer to an `-o` file, the Claude Code CLI is well-behaved
on stdout: `--output-format json` prints a single JSON object with the answer in
`.result`, the session id in `.session_id`, and metadata (`.subtype`, `.is_error`,
`.total_cost_usd`, `.usage`, `.num_turns`, `.duration_ms`). So unlike agy (which
never writes stdout) and unlike codex (which needs an `-o` file), we read the
answer straight from stdout JSON. No transcript scraping on the happy path.

HARNESS ROUTING. `model` picks which executable actually runs:

- None, or an Anthropic first-party name (fable/opus/sonnet/haiku or `claude-*`),
  uses the plain `claude` binary against this Anthropic account.
- Anything else is a claude-os harness model: we exec the `claude-os` wrapper
  (the Rust harness in ~/dev/personal-configs/claude-os) with
  `CLAUDE_OS_MODEL=<id>` in the env. claude-os resolves the id to a record
  (endpoint base URL + token env var + tier overrides, or an Ollama cloud model)
  and `exec`s `claude` with the right env, so DeepSeek/Kimi/Z.AI/Ollama runs land
  on THAT provider's quota, not Anthropic's. The bridge passes its flags through
  untouched — the harness only changes the env, never the argv.

  CAVEAT: claude-os prints a "Launching → ..." banner to STDOUT before exec'ing
  claude. Every stdout parse here therefore locates the first `{` instead of
  json.loads()-ing the whole buffer. `CLAUDE_OS_DRYRUN=1` is never used for an
  answer — it is a diagnostics primitive (prints the resolved command, masked).

CONTINUE / RESUME. claude scopes sessions to the working directory and its git
worktrees, so the subprocess cwd is ALWAYS the workspace. A fresh run returns the
new session id in stdout JSON; we pin it per-workspace (in-memory, guarded by a
lock). claude_continue resumes the pinned id with `--resume <sid>`, or falls back
to `--continue` (claude's native "most recent conversation in this cwd") when the
pin is gone — no on-disk scanning needed, unlike codex's rollout files.

PERMISSIONS. Default sandbox is "default" = explicit `--permission-mode auto`
(automode): the inner claude auto-classifies requests and approves on its own
judgment, matching the user's normal Claude Code behavior. Explicit choices force
a mode: "read-only" -> dontAsk + a read-only allowlist, "workspace-write" ->
acceptEdits + scoped read-only git Bash (deliberately NO unrestricted Bash), and
"danger-full-access" -> --dangerously-skip-permissions. None of these is an OS
sandbox — they are tool-level permission boundaries.

RECURSION GUARD. Every invocation also passes `--strict-mcp-config --mcp-config
'{"mcpServers":{}}'` so the inner claude loads NO MCP servers — otherwise it
would read the user's ~/.claude.json and pull in agent-intern itself, letting a
delegated
sub-agent spawn nested `claude -p` sessions (worst under a swarm). Set
CLAUDE_BRIDGE_INHERIT_MCP=1 to disable the guard (the inner session then keeps
its normal MCP config) — a deliberate security trade-off, not a free feature:
that env var is inherited by every subprocess the server spawns, so only set it
when you explicitly want a delegated sub-agent to have MCP access. Note the
guard only covers MCP servers: hooks, plugins, skills, and CLAUDE.md still load.

SECURITY. claude runs as an autonomous agent. Automode and the write sandboxes
let it modify files under its own judgment; only sandbox="read-only" and the
dontAsk allowlist are a real read-only posture, and even that is tool-level, not
an OS boundary. Only run with trusted prompts on trusted content.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional

# The Claude Code executable. Defaults to "claude" (resolved via PATH); set
# CLAUDE_BIN to an explicit path when it isn't on the server's PATH. Mirrors
# CODEX_BIN / AGY_BIN. Read once at import; the launching process's env wins.
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")

# The claude-os harness wrapper (see module docstring). Only used for harness
# model ids. Set CLAUDE_OS_BIN to an explicit path if it isn't on PATH.
CLAUDE_OS_BIN = os.environ.get("CLAUDE_OS_BIN", "claude-os")

# claude's state home. Honored via CLAUDE_CONFIG_DIR (default ~/.claude). The
# projects dir holds per-cwd transcripts used only by read_history.
CLAUDE_CONFIG_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude"))
PROJECTS_DIR = CLAUDE_CONFIG_DIR / "projects"
CREDENTIALS_FILE = CLAUDE_CONFIG_DIR / ".credentials.json"

# claude -p --permission-mode accepts exactly these (plus aliases). "default" is
# the claude backend's own choice: explicit automode. The other three are the
# same names every other bridge uses, but here they force a permission MODE, not
# an OS sandbox — honest, because claude has no filesystem-level sandbox.
SANDBOX_MODES = ("default", "read-only", "workspace-write", "danger-full-access")
DEFAULT_SANDBOX = "default"

# Anthropic first-party model aliases. Any of these (or a name starting with
# "claude-") goes to the plain `claude` binary; everything else is a claude-os
# harness id. This keeps Anthropic model names from ever being routed to the
# harness (which would treat them as Ollama ids and fail loudly).
ANTHROPIC_FIRST_PARTY = frozenset({"fable", "opus", "sonnet", "haiku"})

# Shorthand -> real claude-os model id. The zsh wrappers claude-ds /
# claude-ds-flash / claude-k3 are CLAUDE_OS_MODEL pins; accept both the full
# wrapper names and short aliases so a coordinator can write model="ds-flash".
HARNESS_MODEL_ALIASES = {
    "claude-ds": "deepseek-v4-pro",
    "claude-ds-flash": "deepseek-v4-flash",
    "claude-k3": "kimi-k3",
    "ds": "deepseek-v4-pro",
    "ds-flash": "deepseek-v4-flash",
    "k3": "kimi-k3",
}

# workspace -> session id, captured from stdout JSON after each fresh ask so
# claude_continue resumes the exact session rooted at that workspace. Guarded by
# a lock because MCP tools may run on different threads. Lives only for the
# process; claude's own `--continue` is the restart-proof fallback.
_PINNED: dict[str, str] = {}
_PIN_LOCK = threading.Lock()

# Sentinel distinguishing "continue with no pin -> --continue" from a real
# session id (so a session id is never mistaken for the fallback).
_CONTINUE = object()

# claude kills background bash tasks ~5s after the final result; the stdout pipe
# can stay open that long behind a lingering child, so after process exit we give
# the reader thread this long to drain the remaining result events.
_STREAM_DRAIN_S = 6.0


def _spawn_kwargs() -> dict:
    """Detach the child: no console window on Windows, a new session elsewhere.

    POSIX uses start_new_session so the child becomes a session leader; that is
    what lets _kill_tree kill the whole process group on timeout (claude spawns
    Node/Bash children that would otherwise survive a plain kill()).
    """
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {"start_new_session": True}


def _env_truthy(name: str) -> bool:
    """True when an env var is set to a truthy value (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def normalize_workspace(ws: Optional[str]) -> str:
    """Absolute path for `ws`, or the server's cwd when omitted."""
    return os.path.abspath(ws) if ws else os.getcwd()


def validate_sandbox(mode: str) -> str:
    """Return `mode` if valid, else raise ValueError listing the allowed values."""
    if mode not in SANDBOX_MODES:
        raise ValueError(f"invalid sandbox {mode!r}; expected one of: {', '.join(SANDBOX_MODES)}")
    return mode


def validate_model(model: Optional[str]) -> Optional[str]:
    """Return `model` unchanged if it resolves to a launcher, else raise ValueError.

    Fail-fast mirror of cursor_bridge.validate_model: catches typos on harness
    ids at normalize time (swarm) instead of letting the run fail deep inside a
    subprocess call.
    """
    if model is None:
        return None
    _resolve_launcher(str(model).strip())  # raises ValueError for unknown ids
    return model


# ------------------------------------------------------------------ launcher routing
def _claude_os_config_dir() -> Path:
    return Path(os.environ.get("CLAUDE_OS_CONFIG_DIR") or (Path.home() / ".config" / "claude-os"))


def _models_file() -> Path:
    return _claude_os_config_dir() / "models.txt"


def _last_used_model() -> Optional[str]:
    """The model id claude-os last launched (last-model), or None."""
    try:
        v = (_claude_os_config_dir() / "last-model").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return v or None


def _load_harness_models() -> dict[str, dict]:
    """Parse ~/.config/claude-os/models.txt into {id: {kind, desc, token_env}}.

    Returns {} when the file is absent — claude-os then falls back to its
    compiled-in defaults (which live in Rust source and aren't readable here), so
    we can't validate ids in that case and pass them through permissively.
    """
    out: dict[str, dict] = {}
    try:
        text = _models_file().read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        f = t.split("|")
        kind = f[0].strip()
        if kind not in ("ollama", "ollama-1m", "endpoint") or len(f) < 2:
            continue
        rec: dict = {"kind": kind, "desc": f[2].strip() if len(f) > 2 else ""}
        if kind == "endpoint" and len(f) > 4:
            rec["token_env"] = f[4].strip()
        out[f[1].strip()] = rec
    return out


def _resolve_launcher(model: Optional[str]) -> tuple[str, dict[str, str], Optional[str]]:
    """(bin, extra_env, --model arg) for a model choice, or raise ValueError.

    None / Anthropic first-party -> plain `claude`, no env, pass `--model`.
    Anything else -> the claude-os harness with CLAUDE_OS_MODEL set, no `--model`
    (the harness's ANTHROPIC_MODEL override, including any `[1m]` suffix, must
    win — passing `--model` would shadow it and lose the 1M window).
    """
    if model is None:
        return (CLAUDE_BIN, {}, None)
    m = str(model).strip()

    # Harness aliases and the bare harness name are checked BEFORE the claude-*
    # first-party rule: "claude-ds" / "claude-ds-flash" / "claude-k3" and the
    # bare "claude-os" all start with "claude-" but are harness names, not
    # Anthropic models — routing them to plain claude was a real bug (they'd run
    # `claude --model claude-ds` against Anthropic instead of the harness).
    if m in HARNESS_MODEL_ALIASES:
        resolved = HARNESS_MODEL_ALIASES[m]
    elif m == "claude-os":
        # The bare harness name is a picker, not a model; headless can't run the
        # picker, so fall back to whatever claude-os launched last.
        last = _last_used_model()
        if not last:
            raise ValueError(
                "model 'claude-os' has no saved last-model "
                "(~/.config/claude-os/last-model); pass a specific harness model id "
                "(e.g. 'ds-flash', 'k3', or an id from the models.txt list)"
            )
        resolved = last
    elif m in ANTHROPIC_FIRST_PARTY or m.startswith("claude-"):
        return (CLAUDE_BIN, {}, m)
    else:
        resolved = m

    models = _load_harness_models()
    if models and resolved not in models:
        available = ", ".join(sorted(models)) or "(none)"
        raise ValueError(f"unknown claude-os model {resolved!r}; available: {available}")
    return (CLAUDE_OS_BIN, {"CLAUDE_OS_MODEL": resolved}, None)


def launcher_env(model: Optional[str]) -> dict[str, str]:
    """Extra env vars (CLAUDE_OS_MODEL) the subprocess must inherit."""
    return _resolve_launcher(model)[1]


# ------------------------------------------------------------------ permission flags
def _permission_flags(sandbox: str) -> list[str]:
    """argv mapping one sandbox value to claude permission flags.

    "default" explicitly passes `--permission-mode auto` (automode) rather than
    relying on the user's settings — deterministic. Read-only and workspace-write
    force a mode with an explicit --allowedTools allowlist; note there is
    deliberately NO unrestricted Bash anywhere: acceptEdits covers Write/Edit and
    common filesystem commands, and the only Bash approved is a scoped read-only
    git set (a raw "Bash" allow would defeat the workspace boundary).
    """
    if sandbox == "default":
        return ["--permission-mode", "auto"]
    if sandbox == "read-only":
        return [
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            "Read",
            "Glob",
            "Grep",
            "Ls",
            "WebSearch",
            "WebFetch",
        ]
    if sandbox == "workspace-write":
        return [
            "--permission-mode",
            "acceptEdits",
            "--allowedTools",
            "Bash(git diff *)",
            "Bash(git status *)",
            "Bash(git log *)",
        ]
    if sandbox == "danger-full-access":
        return ["--dangerously-skip-permissions"]
    raise ValueError(f"invalid sandbox {sandbox!r}; expected one of: {', '.join(SANDBOX_MODES)}")


def _mcp_guard_flags() -> list[str]:
    """Recursion-guard argv: inner claude loads NO MCP servers (see module docstring).

    --strict-mcp-config restricts to exactly what --mcp-config names; an empty
    `{"mcpServers":{}}` names nothing, so no MCP servers load — the inner session
    can't reach agent-intern and spawn nested claude -p sessions. The bare "{}"
    form is REJECTED by claude's config validation ("mcpServers: expected record,
    received undefined") and aborts the run at startup — verified live, so it must
    never be used.
    """
    if _env_truthy("CLAUDE_BRIDGE_INHERIT_MCP"):
        return []
    return ["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']


def _kill_tree(proc) -> None:
    """Kill `proc` and its whole process group/session (claude's Node/Bash children)."""
    if proc is None or proc.pid is None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
        )
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


# ------------------------------------------------------------------ session pinning
def get_pinned(workspace: str) -> Optional[str]:
    """The session id pinned to `workspace` this run, or None."""
    with _PIN_LOCK:
        return _PINNED.get(workspace)


def _pin(workspace: str, session_id: str) -> None:
    with _PIN_LOCK:
        _PINNED[workspace] = session_id


def _resolve_resume_session(workspace: str, continue_conv: bool):
    """None (fresh run), a pinned session id, or the _CONTINUE sentinel.

    Unlike codex (which raises when no prior session exists), claude has a native
    `--continue` that picks the most recent conversation in the cwd — so with no
    pin we return the sentinel instead of raising.
    """
    if not continue_conv:
        return None
    return get_pinned(workspace) or _CONTINUE


# ------------------------------------------------------------------ conversation history
def _encode_cwd(workspace: str) -> str:
    """Claude's project-dir encoding: '-' then each non-alphanumeric run becomes '-'.

    The regex already turns a leading '/' into '-', so strip any leading '-' the
    separator pass produced before prepending the canonical leading '-' (otherwise
    /home/... would encode to '--home-...'). Lossy for paths that mix dots/dashes
    (a/b-c and a/b.c collide), which is why read_history prefers scanning by
    session id when a pin exists.
    """
    return "-" + re.sub(r"[^0-9A-Za-z]+", "-", os.path.abspath(workspace)).lstrip("-")


def _parse_transcript(path: Path) -> list[dict]:
    """Prior turns from one claude transcript JSONL: [{role, content}, ...].

    Tolerant of the v2.1.x schema: `user` records whose message.content is a
    string and origin.kind == "human" are real prompts (tool results and
    environment_context injections are user-role records WITHOUT that origin, so
    they're skipped); `assistant` records contribute their text blocks. Returns
    [] on any read/parse error.
    """
    turns: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        rtype = rec.get("type")
        if rtype == "user":
            msg = rec.get("message") or {}
            content = msg.get("content")
            origin = rec.get("origin") or {}
            if isinstance(content, str) and origin.get("kind") == "human":
                txt = content.strip()
                if txt:
                    turns.append({"role": "user", "content": txt})
        elif rtype == "assistant":
            content = (rec.get("message") or {}).get("content")
            if isinstance(content, list):
                txt = "".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ).strip()
                if txt:
                    turns.append({"role": "assistant", "content": txt})
    return turns


def _find_transcript(workspace: str, sid: Optional[str]) -> Optional[Path]:
    """The transcript file backing the workspace's session, or None.

    Prefers scanning for the exact <sid>.jsonl under the projects dir (the cwd
    encoding is lossy; don't trust the dir name). With no sid, falls back to the
    newest transcript in the encoded-cwd dir — the conversation `--continue`
    would pick.
    """
    if not PROJECTS_DIR.exists():
        return None
    if sid:
        for p in PROJECTS_DIR.rglob("*.jsonl"):
            if p.is_file() and p.stem == sid:
                return p
        return None
    d = PROJECTS_DIR / _encode_cwd(workspace)
    if not d.is_dir():
        return None
    files = [p for p in d.glob("*.jsonl") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)


def read_history(workspace: str, continue_conv: bool) -> list[dict]:
    """Prior turns of the claude session rooted at `workspace`: [{role, content}, …].

    Oldest first, for the watch view's conversation history. Resolves the session
    the way a resume would (in-memory pin, then --continue's choice), then parses
    its transcript. Returns [] for a fresh ask, an unresolved session, or any read
    error.
    """
    if not continue_conv:
        return []
    sid = get_pinned(workspace)
    path = _find_transcript(workspace, sid)
    if path is None:
        return []
    return _parse_transcript(path)


# ------------------------------------------------------------------ output parsing
def _parse_json_result(stdout: str) -> dict:
    """Parse claude's `--output-format json` result from stdout.

    Tolerates the claude-os "Launching → ..." banner that precedes the JSON on
    the same fd (the harness prints it before exec'ing claude), by parsing from
    the first '{'. Raises RuntimeError with the raw tail when nothing parses.
    """
    text = (stdout or "").strip()
    if not text:
        raise RuntimeError("claude produced empty stdout")
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"claude stdout was not JSON (no '{{' found): {text[-300:]}")
    try:
        parsed = json.loads(text[start:])
    except ValueError as e:
        raise RuntimeError(
            f"claude stdout did not parse as JSON: {e}\nstdout tail: {text[-300:]}"
        ) from None
    if not isinstance(parsed, dict):
        raise RuntimeError(f"claude stdout JSON was not an object: {text[-300:]}")
    return parsed


def _result_is_error(parsed: dict) -> bool:
    """True when the result object signals an in-run failure.

    claude exits 0 for in-run failures (auth, quota, refusal), so returncode alone
    is not enough: check is_error and any non-success subtype.
    """
    if parsed.get("is_error"):
        return True
    subtype = parsed.get("subtype")
    return subtype not in (None, "success")


# ------------------------------------------------------------------ running claude
def build_args(
    prompt: str,
    workspace: str,
    sandbox: str,
    model: Optional[str],
    resume_session,
    json_stream: bool = False,
    effort: Optional[str] = None,
) -> list[str]:
    """argv for a headless `claude -p` run (fresh, --resume, or --continue).

    `workspace` is not in the argv — it is the subprocess cwd, which is exactly
    what scopes claude's sessions. The prompt is the final positional arg.
    `json_stream` switches --output-format to stream-json for watch mode. For
    harness models, argv[0] is `claude-os` and the CLAUDE_OS_MODEL env is returned
    separately via launcher_env().

    ORDER QUIRK: _mcp_guard_flags() is placed BEFORE _permission_flags(), not
    after. This is load-bearing, not stylistic. claude CLI v2.1.221 has an
    arg-parser bug: when `--mcp-config`'s JSON value is the token immediately
    before the trailing positional prompt, with no other recognized flag between
    them, claude mis-resolves the mcp-config value and tries to treat the prompt
    text as (part of) a file path, failing with "Invalid MCP configuration: MCP
    config file not found: <cwd>/<prompt text>". Verified at the shell:
        claude -p --output-format json --permission-mode auto --strict-mcp-config \
            --mcp-config '{"mcpServers":{}}' 'ping'   # fails
        claude -p --output-format json --strict-mcp-config --mcp-config '...' \
            --permission-mode auto 'ping'              # succeeds
    This bites for real whenever nothing else sits between the mcp guard and the
    prompt: harness models never pass --model (see _resolve_launcher's
    docstring), and a run with no --effort/--resume has nothing else to fill the
    gap either. _permission_flags() always returns at least one flag token for
    every sandbox mode, so placing it right after _mcp_guard_flags() guarantees a
    recognized flag always separates the mcp-config value from the prompt. Do
    not reorder this back or special-case it away with a dummy trailing flag —
    the reorder is the structurally-correct fix; see test_claude.py's
    test_build_args_mcp_config_value_never_immediately_precedes_prompt.
    """
    bin, _env, model_arg = _resolve_launcher(model)
    args = [bin, "-p"]
    if json_stream:
        args += ["--output-format", "stream-json", "--verbose", "--include-partial-messages"]
    else:
        args += ["--output-format", "json"]
    args += _mcp_guard_flags()
    args += _permission_flags(sandbox)
    if model_arg:
        args += ["--model", model_arg]
    if effort:
        args += ["--effort", effort]
    if resume_session is _CONTINUE:
        args.append("--continue")
    elif resume_session:
        args += ["--resume", resume_session]
    args.append(prompt)
    return args


def run_claude(
    prompt: str,
    workspace: str,
    sandbox: str = DEFAULT_SANDBOX,
    model: Optional[str] = None,
    continue_conv: bool = False,
    timeout_s: int = 180,
    effort: Optional[str] = None,
    pin: bool = True,
) -> str:
    """Run `claude -p --output-format json` (fresh or resume) and return the answer.

    Reads the answer from stdout JSON. On a fresh run, captures the new session
    id and pins it to `workspace` so a later claude_continue can resume the exact
    session — pass `pin=False` for parallel swarm workers (one-shot, no pin).

    Signature is positional-friendly so it can be handed to server.py's
    _run_with_progress(run_fn, args, ...) unchanged.
    """
    validate_sandbox(sandbox)
    # The subprocess cwd must exist; create it for write-capable sandboxes but
    # never for read-only (a read-only run must not mutate the filesystem, and a
    # missing workspace is a legitimate failure for it).
    if sandbox != "read-only":
        os.makedirs(workspace, exist_ok=True)
    env = {**os.environ, **launcher_env(model)}
    resume_session = _resolve_resume_session(workspace, continue_conv)
    args = build_args(prompt, workspace, sandbox, model, resume_session, effort=effort)

    # Popen + communicate instead of subprocess.run so a timeout leaves the child
    # alive long enough for _kill_tree to take the whole group down (run() kills
    # only the direct child and would orphan the Node/Bash descendants).
    proc = subprocess.Popen(
        args,
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        **_spawn_kwargs(),
    )
    try:
        out, err = proc.communicate(timeout=timeout_s + 30)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        raise RuntimeError(
            f"claude timed out after {timeout_s + 30}s\nstderr: {str(err)[-1000:]}"
        ) from None
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude exited {proc.returncode}\n"
            f"stderr: {(err or '')[-1000:]}\n"
            f"stdout: {(out or '')[-500:]}"
        )
    parsed = _parse_json_result(out)
    if _result_is_error(parsed):
        raise RuntimeError((parsed.get("result") or "claude reported an in-run error").strip())
    answer = (parsed.get("result") or "").strip()
    if not answer:
        raise RuntimeError(f"claude produced an empty result. stderr: {(err or '')[-300:]}")
    if not continue_conv and pin:
        sid = parsed.get("session_id")
        if isinstance(sid, str) and sid:
            _pin(workspace, sid)
    return answer


def run_claude_streaming(
    prompt: str,
    workspace: str,
    sandbox: str = DEFAULT_SANDBOX,
    model: Optional[str] = None,
    continue_conv: bool = False,
    timeout_s: int = 180,
    effort: Optional[str] = None,
    on_event=None,
    pin: bool = True,
) -> str:
    """Run `claude -p --output-format stream-json` and stream events live.

    Like run_claude, but feeds each parsed NDJSON event to `on_event(event_dict)`
    as it arrives (this is how watch mode renders steps live), and reads the
    answer from the final `result` event's `.result` (with the accumulated
    text_delta as a fallback). Completion is driven by PROCESS EXIT with a
    deadline — not stdout EOF — because claude can leave a child holding the
    stdout pipe open for the ~5s background-task grace after the result.
    """
    validate_sandbox(sandbox)
    # See run_claude: never create the workspace for a read-only run.
    if sandbox != "read-only":
        os.makedirs(workspace, exist_ok=True)
    env = {**os.environ, **launcher_env(model)}
    resume_session = _resolve_resume_session(workspace, continue_conv)
    args = build_args(
        prompt, workspace, sandbox, model, resume_session, json_stream=True, effort=effort
    )

    state: dict = {"answer": None, "error": None, "session_id": None, "delta": []}
    proc = None
    try:
        proc = subprocess.Popen(
            args,
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            **_spawn_kwargs(),
        )
        err_chunks: list[str] = []

        def _handle(line: str) -> None:
            line = line.strip()
            if not line:
                return
            try:
                ev = json.loads(line)
            except ValueError:
                return
            etype = ev.get("type")
            if etype == "result":
                if ev.get("is_error"):
                    state["error"] = (ev.get("result") or "claude reported an error").strip()
                else:
                    ans = (ev.get("result") or "").strip()
                    if ans:
                        state["answer"] = ans
                sid = ev.get("session_id")
                if isinstance(sid, str) and sid:
                    state["session_id"] = sid
            elif etype == "stream_event":
                delta = ((ev.get("event") or {}).get("delta") or {}).get("text")
                if isinstance(delta, str):
                    state["delta"].append(delta)
            if on_event is not None:
                try:
                    on_event(ev)
                except Exception:  # noqa: BLE001 — a viewer hiccup must not kill the run
                    pass

        def _pump_stdout() -> None:
            try:
                for line in proc.stdout:
                    _handle(line)
            except (ValueError, OSError):
                pass  # pipe closed (e.g. on kill)

        def _pump_stderr() -> None:
            try:
                for line in proc.stderr:
                    err_chunks.append(line)
            except (ValueError, OSError):
                pass

        ot = threading.Thread(target=_pump_stdout, daemon=True)
        et = threading.Thread(target=_pump_stderr, daemon=True)
        ot.start()
        et.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout_s + 30)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            proc.wait()
        # The final `result` event was written BEFORE process exit, so the reader
        # thread usually already has it. Only wait out the pipe-closing lag behind
        # a lingering child when we're still missing a terminal event.
        if state["answer"] is None and state["error"] is None:
            ot.join(timeout=_STREAM_DRAIN_S)
        else:
            ot.join(timeout=1)
        et.join(timeout=1)

        stderr = "".join(err_chunks)
        if timed_out:
            raise RuntimeError(f"claude timed out after {timeout_s + 30}s (watched)")
        if proc.returncode not in (0, None):
            raise RuntimeError(f"claude exited {proc.returncode}\nstderr: {(stderr or '')[-1000:]}")
        if state["error"]:
            raise RuntimeError(state["error"])
        answer = state["answer"] or "".join(state["delta"]).strip()
        if not answer:
            raise RuntimeError(f"claude produced no final message. stderr: {(stderr or '')[-300:]}")
        if not continue_conv and pin and state["session_id"]:
            _pin(workspace, state["session_id"])
        return answer
    finally:
        if proc is not None and proc.poll() is None:
            # safety net: an exception path must never leave claude's tree running
            _kill_tree(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


# ------------------------------------------------------------------ diagnostics
def claude_version() -> Optional[str]:
    """`claude --version` text (first line), or None if claude can't be run."""
    try:
        proc = subprocess.run(
            [CLAUDE_BIN, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            **_spawn_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = ((proc.stdout or "") + (proc.stderr or "")).strip()
    return text.splitlines()[0] if text else None


def claude_auth_status() -> tuple[bool, str]:
    """(logged_in, detail). Spends no quota — `claude auth status` only reads creds."""
    try:
        proc = subprocess.run(
            [CLAUDE_BIN, "auth", "status", "--json"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            **_spawn_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        return (False, f"could not run `claude auth status`: {e}")
    if proc.returncode != 0:
        detail = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return (False, detail.splitlines()[0] if detail else "not logged in")
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return (True, (proc.stdout or "logged in").strip().splitlines()[0])
    logged = bool(data.get("loggedIn"))
    method = data.get("authMethod") or "?"
    provider = data.get("apiProvider") or "?"
    return (logged, f"loggedIn={logged} method={method} provider={provider}")


def _claude_os_bin_path() -> Optional[str]:
    """Absolute path to the claude-os binary, or None if not installed."""
    if os.path.sep in CLAUDE_OS_BIN:
        return CLAUDE_OS_BIN if os.path.isfile(CLAUDE_OS_BIN) else None
    return shutil.which(CLAUDE_OS_BIN)


def claude_os_status() -> list[tuple[str, bool, str]]:
    """Setup diagnostics for the claude-os harness. Spends no quota."""
    rows: list[tuple[str, bool, str]] = []
    if _claude_os_bin_path() is None:
        rows.append(
            (
                "claude-os harness",
                False,
                f"not found on PATH (set CLAUDE_OS_BIN; tried {CLAUDE_OS_BIN!r})",
            )
        )
        return rows

    models = _load_harness_models()
    if models:
        detail = f"{len(models)} model(s) loaded from {_models_file()}"
    else:
        detail = "installed (using claude-os compiled-in model list)"
    rows.append(("claude-os harness", True, detail))

    last = _last_used_model()
    rows.append(("claude-os last model", bool(last), last or "none"))

    missing = sorted(
        {
            r.get("token_env")
            for r in models.values()
            if r.get("token_env") and not os.environ.get(r["token_env"])
        }
    )
    if missing:
        rows.append(("claude-os tokens", False, "unset: " + ", ".join(missing)))
    else:
        rows.append(("claude-os tokens", True, "all endpoint token env vars set"))
    return rows


def status_rows() -> list[tuple[str, bool, str]]:
    """Setup diagnostics as (label, ok, detail) rows. Spends no quota.

    Mirrors codex_bridge.status_rows' shape so server.py can render claude rows
    with the same formatter.
    """
    rows: list[tuple[str, bool, str]] = []

    ver = claude_version()
    if ver is None:
        rows.append(
            ("claude CLI", False, f"not found on PATH (set CLAUDE_BIN; tried {CLAUDE_BIN!r})")
        )
    else:
        rows.append(("claude CLI", True, ver))

    logged_in, detail = claude_auth_status()
    rows.append(("claude auth", logged_in, detail))

    rows.extend(claude_os_status())

    rows.append(("projects dir", PROJECTS_DIR.exists(), str(PROJECTS_DIR)))

    with _PIN_LOCK:
        n_pins = len(_PINNED)
    rows.append(("pinned sessions", True, f"{n_pins} workspace(s) pinned this run"))

    return rows
