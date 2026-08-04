"""Offline unit tests for the pure logic in claude_bridge.py.

Like test_codex.py these use temp fixtures and monkeypatching and never invoke the
real `claude` / `claude-os`, so they cost no Anthropic/DeepSeek/Kimi quota. The
live round-trip lives in the smoke test.

    pytest test_claude.py
"""

import json
import os
import subprocess
import sys
import time

import pytest

import claude_bridge
import server

SAMPLE_SID = "019ef10a-fc0e-7180-b22f-5bd19fe8fc5b"


def _write_models(tmp_path, lines):
    """Point CLAUDE_OS_CONFIG_DIR at tmp_path with a models.txt fixture."""
    d = tmp_path / "claude-os"
    d.mkdir()
    (d / "models.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


# --------------------------------------------------------------------------
# validate_sandbox / defaults
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", claude_bridge.SANDBOX_MODES)
def test_validate_sandbox_accepts_valid(mode):
    assert claude_bridge.validate_sandbox(mode) == mode


def test_validate_sandbox_rejects_unknown():
    with pytest.raises(ValueError):
        claude_bridge.validate_sandbox("yolo")


def test_default_sandbox_is_default():
    # The claude backend defaults to automode, NOT read-only (deliberate: the user
    # runs Claude Code in automode and expects delegated sub-agents to match).
    assert claude_bridge.DEFAULT_SANDBOX == "default"
    assert "default" in claude_bridge.SANDBOX_MODES


# --------------------------------------------------------------------------
# normalize_workspace
# --------------------------------------------------------------------------


def test_normalize_workspace_none_is_cwd():
    assert claude_bridge.normalize_workspace(None) == os.getcwd()


def test_normalize_workspace_abspath(tmp_path):
    assert claude_bridge.normalize_workspace(str(tmp_path)) == os.path.abspath(str(tmp_path))


# --------------------------------------------------------------------------
# _permission_flags — the sandbox -> permission-mode mapping
# --------------------------------------------------------------------------


def test_permission_flags_default_is_explicit_automode():
    assert claude_bridge._permission_flags("default") == ["--permission-mode", "auto"]


def test_permission_flags_read_only_allowlists_read_tools():
    flags = claude_bridge._permission_flags("read-only")
    assert flags[:2] == ["--permission-mode", "dontAsk"]
    assert "--allowedTools" in flags
    tools = flags[flags.index("--allowedTools") + 1 :]
    assert "Bash" not in tools  # no shell access at all
    assert "Write" not in tools and "Edit" not in tools  # no writes
    assert "Read" in tools and "Grep" in tools


def test_permission_flags_workspace_write_has_no_unrestricted_bash():
    flags = claude_bridge._permission_flags("workspace-write")
    assert flags[:2] == ["--permission-mode", "acceptEdits"]
    assert "Bash" not in flags  # only scoped "Bash(git ...)" rules are allowed
    assert any(a.startswith("Bash(git ") for a in flags)


def test_permission_flags_danger_full_access():
    assert claude_bridge._permission_flags("danger-full-access") == [
        "--dangerously-skip-permissions"
    ]


def test_permission_flags_invalid_raises():
    with pytest.raises(ValueError):
        claude_bridge._permission_flags("yolo")


# --------------------------------------------------------------------------
# _mcp_guard_flags — recursion guard
# --------------------------------------------------------------------------


def test_mcp_guard_on_by_default():
    assert claude_bridge._mcp_guard_flags() == [
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
    ]


def test_mcp_guard_off_with_inherit_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_BRIDGE_INHERIT_MCP", "1")
    assert claude_bridge._mcp_guard_flags() == []


# --------------------------------------------------------------------------
# launcher routing — plain claude vs the claude-os harness
# --------------------------------------------------------------------------


def test_resolve_launcher_none_uses_claude():
    assert claude_bridge._resolve_launcher(None) == (claude_bridge.CLAUDE_BIN, {}, None)


@pytest.mark.parametrize("model", ["fable", "opus", "sonnet", "haiku", "claude-fable-5"])
def test_resolve_launcher_anthropic_first_party(model):
    bin, env, model_arg = claude_bridge._resolve_launcher(model)
    assert bin == claude_bridge.CLAUDE_BIN
    assert env == {}
    assert model_arg == model


def test_resolve_launcher_harness_alias(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CLAUDE_OS_CONFIG_DIR",
        str(
            _write_models(
                tmp_path,
                [
                    "endpoint|deepseek-v4-pro|pro|https://api.deepseek.com/anthropic|DEEPSEEK_API_KEY",
                    "endpoint|deepseek-v4-flash|flash|https://api.deepseek.com/anthropic|DEEPSEEK_API_KEY",
                ],
            )
        ),
    )
    bin, env, model_arg = claude_bridge._resolve_launcher("ds-flash")
    assert bin == claude_bridge.CLAUDE_OS_BIN
    assert env == {"CLAUDE_OS_MODEL": "deepseek-v4-flash"}
    assert model_arg is None  # harness env wins; no --model


def test_resolve_launcher_harness_alias_named_claude_ds(tmp_path, monkeypatch):
    # Regression: "claude-ds" starts with "claude-" and must NOT be mistaken for
    # an Anthropic first-party model — it is a harness alias for deepseek-v4-pro.
    monkeypatch.setenv(
        "CLAUDE_OS_CONFIG_DIR",
        str(
            _write_models(
                tmp_path,
                [
                    "endpoint|deepseek-v4-pro|pro|https://api.deepseek.com/anthropic|DEEPSEEK_API_KEY"
                ],
            )
        ),
    )
    bin, env, model_arg = claude_bridge._resolve_launcher("claude-ds")
    assert bin == claude_bridge.CLAUDE_OS_BIN
    assert env == {"CLAUDE_OS_MODEL": "deepseek-v4-pro"}
    assert model_arg is None


def test_resolve_launcher_bare_claude_os_uses_last_model(tmp_path, monkeypatch):
    d = _write_models(
        tmp_path,
        ["endpoint|deepseek-v4-flash|flash|https://api.deepseek.com/anthropic|DEEPSEEK_API_KEY"],
    )
    (d / "last-model").write_text("deepseek-v4-flash\n", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_OS_CONFIG_DIR", str(d))
    bin, env, model_arg = claude_bridge._resolve_launcher("claude-os")
    assert bin == claude_bridge.CLAUDE_OS_BIN
    assert env == {"CLAUDE_OS_MODEL": "deepseek-v4-flash"}
    assert model_arg is None


def test_resolve_launcher_bare_claude_os_without_last_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CLAUDE_OS_CONFIG_DIR",
        str(
            _write_models(
                tmp_path,
                [
                    "endpoint|deepseek-v4-flash|flash|https://api.deepseek.com/anthropic|DEEPSEEK_API_KEY"
                ],
            )
        ),
    )
    with pytest.raises(ValueError):
        claude_bridge._resolve_launcher("claude-os")


def test_resolve_launcher_unknown_id_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CLAUDE_OS_CONFIG_DIR",
        str(
            _write_models(
                tmp_path,
                [
                    "endpoint|deepseek-v4-pro|pro|https://api.deepseek.com/anthropic|DEEPSEEK_API_KEY",
                ],
            )
        ),
    )
    with pytest.raises(ValueError):
        claude_bridge._resolve_launcher("not-a-model")


def test_resolve_launcher_permissive_without_models_file(tmp_path, monkeypatch):
    # No models.txt -> claude-os falls back to its compiled defaults, which we
    # can't read, so unknown ids pass through (the harness treats them as Ollama).
    monkeypatch.setenv("CLAUDE_OS_CONFIG_DIR", str(tmp_path / "claude-os"))
    bin, env, _ = claude_bridge._resolve_launcher("deepseek-v4-flash")
    assert bin == claude_bridge.CLAUDE_OS_BIN
    assert env == {"CLAUDE_OS_MODEL": "deepseek-v4-flash"}


def test_validate_model_fail_fast(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CLAUDE_OS_CONFIG_DIR",
        str(
            _write_models(
                tmp_path,
                [
                    "endpoint|deepseek-v4-pro|pro|https://api.deepseek.com/anthropic|DEEPSEEK_API_KEY",
                ],
            )
        ),
    )
    assert claude_bridge.validate_model(None) is None
    assert claude_bridge.validate_model("sonnet") == "sonnet"
    assert claude_bridge.validate_model("ds") == "ds"
    with pytest.raises(ValueError):
        claude_bridge.validate_model("bogus")


# --------------------------------------------------------------------------
# build_args — the heart of the bridge: fresh vs resume vs stream argv shape
# --------------------------------------------------------------------------


def test_build_args_fresh_basic():
    args = claude_bridge.build_args("hello", "C:\\ws", "default", None, None)
    assert args[0] == claude_bridge.CLAUDE_BIN
    assert args[1] == "-p"
    assert args[args.index("--output-format") + 1] == "json"
    assert "--permission-mode" in args and args[args.index("--permission-mode") + 1] == "auto"
    assert "--strict-mcp-config" in args
    assert args[-1] == "hello"  # prompt is positional and last


def test_build_args_fresh_dash_prompt_is_last():
    args = claude_bridge.build_args("-write a poem", "ws", "default", None, None)
    assert args[-1] == "-write a poem"


def test_build_args_anthropic_model_adds_model_flag():
    args = claude_bridge.build_args("p", "ws", "default", "sonnet", None)
    assert args[args.index("--model") + 1] == "sonnet"


def test_build_args_harness_model_uses_claude_os_bin():
    args = claude_bridge.build_args("p", "ws", "default", "ds-flash", None)
    assert args[0] == claude_bridge.CLAUDE_OS_BIN
    assert "--model" not in args  # harness env controls the model


def test_build_args_resume_uses_pinned_sid():
    args = claude_bridge.build_args("again", "ws", "default", None, SAMPLE_SID)
    assert args[args.index("--resume") + 1] == SAMPLE_SID
    assert "--continue" not in args
    assert args[-1] == "again"


def test_build_args_continue_fallback():
    args = claude_bridge.build_args("again", "ws", "default", None, claude_bridge._CONTINUE)
    assert "--continue" in args
    assert "--resume" not in args


def test_build_args_stream_adds_stream_json():
    args = claude_bridge.build_args("p", "ws", "default", None, None, json_stream=True)
    assert args[args.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in args
    assert "--include-partial-messages" in args


def test_build_args_effort_passthrough():
    args = claude_bridge.build_args("p", "ws", "default", None, None, effort="xhigh")
    assert args[args.index("--effort") + 1] == "xhigh"


def test_build_args_no_effort_by_default():
    args = claude_bridge.build_args("p", "ws", "default", None, None)
    assert "--effort" not in args


@pytest.mark.parametrize("sandbox", claude_bridge.SANDBOX_MODES)
@pytest.mark.parametrize("model", [None, "sonnet", "ds-flash"])
@pytest.mark.parametrize("resume_session", [None, claude_bridge._CONTINUE, SAMPLE_SID])
@pytest.mark.parametrize("effort", [None, "high"])
@pytest.mark.parametrize("json_stream", [False, True])
def test_build_args_mcp_config_value_never_immediately_precedes_prompt(
    sandbox, model, resume_session, effort, json_stream
):
    """Regression for a claude CLI (v2.1.221) arg-parser bug: when `--mcp-config`'s
    JSON value is the token immediately before the trailing positional prompt, with
    no other recognized flag in between, claude mis-resolves the mcp-config value
    and tries to treat the prompt text as (part of) a file path, failing with:

        Error: Invalid MCP configuration:
        MCP config file not found: <cwd>/<prompt text>

    Reproduced directly at the shell, bypassing this bridge and claude-os entirely,
    with the exact minimal argv this bridge used to build for harness models with
    no --model/--effort/--resume flag (the most common harness-model case):

        claude -p --output-format json --permission-mode auto --strict-mcp-config \\
            --mcp-config '{"mcpServers":{}}' 'ping'
        # => Error: Invalid MCP configuration: MCP config file not found: <cwd>/ping

    Adding any recognized flag between the mcp-config value and the prompt (e.g.
    --model, or reordering --mcp-config earlier so --permission-mode follows it)
    avoids the bug. We can't invoke the real claude binary from a unit test (no
    network/quota here), so this test locks the structural invariant instead:
    across every sandbox/model/effort/resume/stream combination, --mcp-config's
    value must never be the immediate predecessor of the trailing prompt arg.
    """
    args = claude_bridge.build_args(
        "prompt text", "ws", sandbox, model, resume_session, json_stream=json_stream, effort=effort
    )
    assert args[-1] == "prompt text"
    if "--mcp-config" not in args:
        return  # CLAUDE_BRIDGE_INHERIT_MCP=1 path — no mcp-config flag at all
    mcp_value_idx = args.index("--mcp-config") + 1
    assert mcp_value_idx != len(args) - 2, (
        "--mcp-config's value sits immediately before the trailing prompt arg, "
        f"the exact shape that triggers the claude CLI parser bug: {args}"
    )


# --------------------------------------------------------------------------
# _parse_json_result — banner tolerance + error paths
# --------------------------------------------------------------------------


def test_parse_json_result_plain():
    out = json.dumps({"type": "result", "subtype": "success", "result": "hi"})
    assert claude_bridge._parse_json_result(out)["result"] == "hi"


def test_parse_json_result_tolerates_claude_os_banner():
    out = (
        "Launching → claude via deepseek-v4-flash https://api.deepseek.com/anthropic\n"
        "\n" + json.dumps({"type": "result", "subtype": "success", "result": "hi"})
    )
    assert claude_bridge._parse_json_result(out)["result"] == "hi"


def test_parse_json_result_rejects_non_json():
    with pytest.raises(RuntimeError):
        claude_bridge._parse_json_result("no json here")


def test_result_is_error_detects_error_flags():
    assert claude_bridge._result_is_error({"is_error": True, "subtype": "success"}) is True
    assert claude_bridge._result_is_error({"subtype": "error_max_turns"}) is True
    assert claude_bridge._result_is_error({"subtype": "success", "result": "ok"}) is False


# --------------------------------------------------------------------------
# _FAKE_CLAUDE infra: subprocess.Popen -> a script that emits the result JSON
# --------------------------------------------------------------------------

_FAKE_CLAUDE = """
import sys, json
mode = sys.argv[1]
if mode == "fail":
    sys.stderr.write("boom\\n")
    sys.exit(1)
if mode == "error":
    print(json.dumps({"type":"result","subtype":"error_max_turns",
                      "session_id":"019ef10a-fc0e-7180-b22f-5bd19fe8fc5b",
                      "is_error":True,"result":"claude reported an in-run error"}))
    sys.exit(0)
if mode == "empty":
    print(json.dumps({"type":"result","subtype":"success",
                      "session_id":"019ef10a-fc0e-7180-b22f-5bd19fe8fc5b",
                      "is_error":False,"result":"   "}))
    sys.exit(0)
print(json.dumps({"type":"result","subtype":"success",
                  "session_id":"019ef10a-fc0e-7180-b22f-5bd19fe8fc5b",
                  "is_error":False,"result":"FAKE ANSWER"}))
sys.exit(0)
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    """Redirect claude_bridge.subprocess.Popen to a fake claude script.

    Captures the argv/cwd/env of the real Popen call for assertions, then launches
    the fake script (mode selected by FAKE_CLAUDE_MODE env).
    """
    fake = tmp_path / "fake_claude.py"
    fake.write_text(_FAKE_CLAUDE, encoding="utf-8")
    real_popen = subprocess.Popen
    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = list(args)
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        mode = os.environ.get("FAKE_CLAUDE_MODE", "normal")
        return real_popen(
            [sys.executable, str(fake), mode],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    monkeypatch.setattr(claude_bridge.subprocess, "Popen", fake_popen)
    return captured


def test_run_claude_fresh_returns_answer_and_pins(tmp_path, monkeypatch, fake_claude):
    monkeypatch.setattr(claude_bridge, "_PINNED", {})
    ws = str(tmp_path / "ws")
    ans = claude_bridge.run_claude("hello", ws, "default", None, False, 30)
    assert ans == "FAKE ANSWER"
    assert claude_bridge.get_pinned(ws) == SAMPLE_SID


def test_run_claude_pin_false_does_not_pin(tmp_path, monkeypatch, fake_claude):
    monkeypatch.setattr(claude_bridge, "_PINNED", {})
    ws = str(tmp_path / "ws")
    claude_bridge.run_claude("hello", ws, "default", None, False, 30, pin=False)
    assert claude_bridge.get_pinned(ws) is None


def test_run_claude_continue_does_not_repin(tmp_path, monkeypatch, fake_claude):
    monkeypatch.setattr(claude_bridge, "_PINNED", {})
    ws = str(tmp_path / "ws")
    claude_bridge.run_claude("hello", ws, "default", None, True, 30)
    assert claude_bridge.get_pinned(ws) is None  # continue keeps the original sid


def test_run_claude_continue_passes_continue_flag(tmp_path, monkeypatch, fake_claude):
    monkeypatch.setattr(claude_bridge, "_PINNED", {})
    claude_bridge.run_claude("hello", str(tmp_path / "ws"), "default", None, True, 30)
    assert "--continue" in fake_claude["args"]
    assert "--resume" not in fake_claude["args"]


def test_run_claude_resume_passes_pinned_sid(tmp_path, monkeypatch, fake_claude):
    monkeypatch.setattr(claude_bridge, "_PINNED", {})
    claude_bridge._pin(str(tmp_path / "ws"), SAMPLE_SID)
    claude_bridge.run_claude("hello", str(tmp_path / "ws"), "default", None, True, 30)
    assert fake_claude["args"][fake_claude["args"].index("--resume") + 1] == SAMPLE_SID


def test_run_claude_harness_sets_claude_os_model_env(tmp_path, monkeypatch, fake_claude):
    monkeypatch.setenv(
        "CLAUDE_OS_CONFIG_DIR",
        str(
            _write_models(
                tmp_path,
                [
                    "endpoint|deepseek-v4-flash|flash|https://api.deepseek.com/anthropic|DEEPSEEK_API_KEY",
                ],
            )
        ),
    )
    monkeypatch.setattr(claude_bridge, "_PINNED", {})
    ws = str(tmp_path / "ws")
    claude_bridge.run_claude("hello", ws, "default", "ds-flash", False, 30)
    assert fake_claude["args"][0] == claude_bridge.CLAUDE_OS_BIN
    assert fake_claude["env"]["CLAUDE_OS_MODEL"] == "deepseek-v4-flash"


def test_run_claude_effort_flag_passed(tmp_path, monkeypatch, fake_claude):
    claude_bridge.run_claude(
        "hello", str(tmp_path / "ws"), "default", None, False, 30, effort="xhigh"
    )
    assert fake_claude["args"][fake_claude["args"].index("--effort") + 1] == "xhigh"


def test_run_claude_sets_cwd_to_workspace(tmp_path, monkeypatch, fake_claude):
    ws = str(tmp_path / "ws")
    claude_bridge.run_claude("hello", ws, "default", None, False, 30)
    assert os.path.normpath(fake_claude["cwd"]) == os.path.normpath(ws)


def test_run_claude_in_run_error_raises(tmp_path, monkeypatch, fake_claude):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "error")
    with pytest.raises(RuntimeError, match="in-run error"):
        claude_bridge.run_claude("hello", str(tmp_path / "ws"), "default", None, False, 30)


def test_run_claude_empty_result_raises(tmp_path, monkeypatch, fake_claude):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "empty")
    with pytest.raises(RuntimeError, match="empty result"):
        claude_bridge.run_claude("hello", str(tmp_path / "ws"), "default", None, False, 30)


def test_run_claude_nonzero_exit_raises(tmp_path, monkeypatch, fake_claude):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "fail")
    with pytest.raises(RuntimeError, match="exited 1"):
        claude_bridge.run_claude("hello", str(tmp_path / "ws"), "default", None, False, 30)


def test_run_claude_rejects_bad_sandbox(tmp_path, fake_claude):
    with pytest.raises(ValueError):
        claude_bridge.run_claude("hello", str(tmp_path / "ws"), "yolo", None, False, 30)


# --------------------------------------------------------------------------
# run_claude_streaming completes on PROCESS EXIT, not stdout EOF: claude can
# leave a child holding the stdout pipe open after the turn, so a stdout-EOF loop
# would hang until the watchdog. Fake CLI reproduces that lingering child; the
# run must still return promptly with the answer from the final result event.
# --------------------------------------------------------------------------

_FAKE_CLAUDE_STREAM = """
import sys, json, subprocess
SID = "019ef10a-fc0e-7180-b22f-5bd19fe8fc5b"
def E(t, **kw):
    d = {"type": t}; d.update(kw); return d
def SEV(text):
    return E("stream_event", event={"type": "content_block_delta",
                                    "delta": {"type": "text_delta", "text": text}})
for e in [E("system", subtype="init", session_id=SID),
          SEV("Fake "), SEV("answer"),
          E("result", subtype="success", session_id=SID,
            is_error=False, result="FAKE ANSWER")]:
    sys.stdout.write(json.dumps(e) + "\\n")
sys.stdout.flush()
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(8)"])
sys.exit(0)
"""


def test_run_claude_streaming_completes_on_process_exit_not_stdout_eof(tmp_path, monkeypatch):
    fake = tmp_path / "fake_claude_stream.py"
    fake.write_text(_FAKE_CLAUDE_STREAM, encoding="utf-8")
    real_popen = subprocess.Popen

    def fake_popen(args, **kwargs):
        return real_popen(
            [sys.executable, str(fake)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    monkeypatch.setattr(claude_bridge.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(claude_bridge, "_PINNED", {})
    events = []
    t = time.time()
    ans = claude_bridge.run_claude_streaming(
        "p", str(tmp_path / "ws"), "default", None, False, 30, on_event=events.append, pin=False
    )
    dt = time.time() - t
    assert ans == "FAKE ANSWER"
    assert dt < 4.0, f"must return on process exit (~1s), not wait for the 8s child; took {dt:.1f}s"
    assert any(e.get("type") == "result" for e in events)  # events streamed to on_event


# --------------------------------------------------------------------------
# status / auth
# --------------------------------------------------------------------------


def test_claude_version_none_when_binary_missing(monkeypatch):
    monkeypatch.setattr(claude_bridge, "CLAUDE_BIN", "definitely-not-a-real-claude-bin")
    monkeypatch.setattr(claude_bridge.subprocess, "run", _raise_oserror)
    assert claude_bridge.claude_version() is None


def _raise_oserror(*a, **k):
    raise OSError("boom")


def test_claude_auth_status_ok(monkeypatch):
    out = '{"loggedIn": true, "authMethod": "oauth_token", "apiProvider": "firstParty"}'

    class _R:
        returncode = 0
        stdout = out
        stderr = ""

    monkeypatch.setattr(claude_bridge.subprocess, "run", lambda *a, **k: _R())
    ok, detail = claude_bridge.claude_auth_status()
    assert ok is True
    assert "oauth_token" in detail


def test_claude_auth_status_not_logged_in(monkeypatch):
    class _R:
        returncode = 1
        stdout = ""
        stderr = "not logged in"

    monkeypatch.setattr(claude_bridge.subprocess, "run", lambda *a, **k: _R())
    ok, detail = claude_bridge.claude_auth_status()
    assert ok is False


def test_status_rows_claude_missing(monkeypatch):
    monkeypatch.setattr(claude_bridge, "claude_version", lambda: None)
    monkeypatch.setattr(claude_bridge, "claude_auth_status", lambda: (False, "not logged in"))
    monkeypatch.setattr(claude_bridge, "_claude_os_bin_path", lambda: None)
    rows = {label: (ok, detail) for label, ok, detail in claude_bridge.status_rows()}
    assert rows["claude CLI"][0] is False
    assert rows["claude auth"][0] is False
    assert rows["claude-os harness"][0] is False


def test_status_rows_claude_ok(monkeypatch):
    monkeypatch.setattr(claude_bridge, "claude_version", lambda: "2.1.220")
    monkeypatch.setattr(claude_bridge, "claude_auth_status", lambda: (True, "oauth_token"))
    monkeypatch.setattr(claude_bridge, "_claude_os_bin_path", lambda: "/usr/local/bin/claude-os")
    monkeypatch.setattr(
        claude_bridge,
        "_load_harness_models",
        lambda: {
            "deepseek-v4-flash": {
                "kind": "endpoint",
                "desc": "flash",
                "token_env": "DEEPSEEK_API_KEY",
            },
        },
    )
    rows = {label: (ok, detail) for label, ok, detail in claude_bridge.status_rows()}
    assert rows["claude CLI"] == (True, "2.1.220")
    assert rows["claude auth"][0] is True
    assert rows["claude-os harness"][0] is True


# --------------------------------------------------------------------------
# read_history — v2.1.x transcript parsing
# --------------------------------------------------------------------------


def _write_transcript(d, sid, events):
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sid}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return p


def test_read_history_parses_user_and_assistant_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_bridge, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(claude_bridge, "_PINNED", {})
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_transcript(
        tmp_path / "projects" / claude_bridge._encode_cwd(str(ws)),
        SAMPLE_SID,
        [
            {"type": "user", "message": {"content": "first Q"}, "origin": {"kind": "human"}},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "..."},
                        {"type": "text", "text": "first A"},
                    ]
                },
            },
            # a tool-result user record has NO human origin -> must be skipped
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "tool_result", "content": "ignored"},
                    ]
                },
                "origin": {"kind": "tool_result"},
            },
            {"type": "user", "message": {"content": "second Q"}, "origin": {"kind": "human"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "second A"}]}},
        ],
    )
    assert claude_bridge.read_history(str(ws), True) == [
        {"role": "user", "content": "first Q"},
        {"role": "assistant", "content": "first A"},
        {"role": "user", "content": "second Q"},
        {"role": "assistant", "content": "second A"},
    ]


def test_read_history_finds_transcript_by_pinned_sid(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_bridge, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(claude_bridge, "_PINNED", {})
    # deliberately mismatch the encoded dir name vs the real cwd to prove sid scan
    _write_transcript(
        tmp_path / "projects" / "-totally-different-name",
        SAMPLE_SID,
        [{"type": "user", "message": {"content": "only Q"}, "origin": {"kind": "human"}}],
    )
    claude_bridge._pin("C:\\proj", SAMPLE_SID)
    assert claude_bridge.read_history("C:\\proj", True) == [{"role": "user", "content": "only Q"}]


def test_read_history_empty_for_fresh_or_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_bridge, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(claude_bridge, "_PINNED", {})
    assert claude_bridge.read_history("C:\\proj", False) == []
    assert claude_bridge.read_history("C:\\nowhere", True) == []


def test_encode_cwd_matches_claude_convention():
    assert claude_bridge._encode_cwd("/home/andrew/dev/consulting/invoices") == (
        "-home-andrew-dev-consulting-invoices"
    )


# --------------------------------------------------------------------------
# watch mode: stream-json event -> watch-line mapping (server-side converter)
# --------------------------------------------------------------------------


def test_watch_lines_assistant_narration():
    ev = {"type": "assistant", "message": {"content": [{"type": "text", "text": "hello\nworld"}]}}
    assert server._claude_event_to_watch_lines(ev) == [("narration", "hello")]


def test_watch_lines_tool_use_start():
    ev = {
        "type": "stream_event",
        "event": {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
        },
    }
    kind, text = server._claude_event_to_watch_lines(ev)[0]
    assert kind == "command" and "Read" in text


def test_watch_lines_tool_use_start_tracks_index():
    tool_indices = set()
    ev = {
        "type": "stream_event",
        "event": {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
        },
    }
    server._claude_event_to_watch_lines(ev, tool_indices)
    assert 2 in tool_indices


def test_watch_lines_tool_use_stop_only_for_tracked_tool_block():
    tool_indices = {0}  # recorded from a prior tool_use content_block_start
    ev = {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}}
    assert server._claude_event_to_watch_lines(ev, tool_indices) == [("result", "done")]
    assert 0 not in tool_indices  # consumed


def test_watch_lines_text_block_stop_is_not_done():
    # a text block's stop carries no block type; without a tracked tool index it
    # must NOT read as a fake "done"
    ev = {"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}}
    assert server._claude_event_to_watch_lines(ev) == []
    assert server._claude_event_to_watch_lines(ev, {1}) == []  # untracked index


def test_watch_lines_ignores_noise():
    assert server._claude_event_to_watch_lines({"type": "system", "subtype": "init"}) == []
    assert (
        server._claude_event_to_watch_lines(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "x"},
                },
            }
        )
        == []
    )


def test_watch_lines_error_event():
    ev = {"type": "error", "message": "something broke"}
    assert server._claude_event_to_watch_lines(ev) == [("result", "error: something broke")]


def test_watch_lines_result_is_error():
    ev = {"type": "result", "is_error": True, "subtype": "error_max_turns"}
    assert server._claude_event_to_watch_lines(ev) == [("result", "error")]
