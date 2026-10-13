# Codex Adapter

`CodexAdapter` runs one Codex process per Band room over stdio, each in its own
workspace. Runnable scripts: [examples/codex/](../../examples/codex/).

Automatic idle release is unavailable pending native safety qualification;
see [idle room resource retention](../idle-room-release.md).

For existing integrations, follow the [SDK 4.x to 5.x migration](../acp.md#upgrading-from-sdk-4x-to-5x).

- **Where settings go.** Runtime settings live in `CodexAdapterConfig`, which
  forbids unknown fields, so a misplaced `emit=` or a typo fails construction.
  `emit=`, `capabilities=` and `additional_tools=` go on `CodexAdapter`.
- **Room approvals need a non-default `approval_policy`.** It is forwarded to
  Codex unchanged, and the default `"never"` means Codex never asks.
  `approval_mode` only decides how Band answers the requests Codex does send.
  `"on-request"` with `"read-only"` does not guarantee a request: Codex may
  attempt the command inside the sandbox and receive a denial without asking.
  The model may need to request escalation explicitly. For commands that need
  approval without model-requested escalation, the app-server RPC supports
  `"untrusted"` (verified with CLI 0.149.0 and 0.160.0). Commands must use
  default sandbox permissions: Codex rejects explicit `require_escalated`
  overrides under this policy before requesting approval. The model must still
  attempt the tool call; the policy cannot force it to act. This is distinct from
  the [retired user/project TOML setting](https://learn.chatgpt.com/docs/agent-approvals-security);
  see the [released RPC schema](https://github.com/openai/codex/blob/rust-v0.149.0/codex-rs/app-server-protocol/schema/typescript/v2/AskForApproval.ts).
- **`sandbox_policy` pins the sandbox.** While it is set, the `/sandbox` room
  command is refused.
- **One workspace per room.** The default is `./.band-workspaces/<room-id>`, and
  a custom `workspace_for_room` must return a distinct absolute path for every
  live room. `cwd`, a non-stdio `transport` and a custom `client_factory` are
  rejected at construction because they cannot guarantee per-room isolation.
- **Ownership survives recovery and failed cleanup.** The workspace stays
  claimed until the room leaves and its process is closed. Failed startup,
  close errors and close timeouts retain the claim for retry. A timed-out or
  cancelled caller does not abandon subprocess cleanup; retry room cleanup to
  finish releasing ownership. Shutdown attempts every room and reports failures.
- **POSIX cleanup settles the original owned process group.** After a normal
  leader exit or an exit timeout, close sends TERM, waits up to one second,
  then sends KILL and waits up to one more second for `ESRCH`. An unconfirmed
  group raises `CleanupIncompleteError` and retains the client and workspace
  claim for retry. The group ID comes from the original subprocess handle,
  never a lookup of a possibly recycled PID. POSIX groups lack a
  generation-bound handle: reuse between the original group emptying and the
  final probe or signal remains a residual race. Descendants that leave the
  owned group are outside this guarantee. Windows retains best-effort
  `taskkill /F /T` on leader timeout; it cannot confirm orphaned descendants
  after normal leader exit or descendants that escape that tree.
- **Band tools own replies and declines.** Native completed text is optional
  thought telemetry without mentions; it never settles a turn. A native-only
  turn fails the shared missing-reply verdict. `Emit.THOUGHTS` controls visibility.
  Remove retired `fallback_send_agent_text` and `assistant_text_mode` config keys
  and `CODEX_FALLBACK_SEND_AGENT_TEXT` / `CODEX_ASSISTANT_TEXT_MODE` environment
  variables. Explicit configs reject retired keys; the environment keys are ignored.
- **Tool filters apply to fresh threads.** Include/category/exclude filters narrow
  the dynamic platform tools advertised when starting a thread. Resumed threads
  retain their saved registrations: start a fresh thread after changing filters.
  First-prompt context refreshes communication instructions, not saved tool schemas.
- **`reasoning_effort` is not validated.** The valid values depend on the model
  and the Codex CLI version, and the backend rejects unknown ones.
- **`skill_roots` need Codex CLI 0.136.0 or newer.** Codex has no config key
  for extra skill folders, so the adapter sends them to each room's app-server
  with `skills/extraRoots/set` before the room's thread starts. An older CLI
  rejects the request, and the room fails to start rather than running without
  the skills. Roots must be absolute paths; Codex accepts ones that don't exist.

While an accepted turn awaits approval, newer ordinary room work remains
outstanding. The runtime can deliver approval and status controls ahead of that
pending work through Codex's existing authorization rules. The active turn keeps
its original tools; an ordinary input rejected before provider invocation is
retried by the shared room scheduler after that turn releases the session.

## Runtime MCP servers

Pass the keyword-only `resolve_mcp_servers` argument to `CodexAdapter`, not to
`CodexAdapterConfig`. It is an async callable receiving the room's absolute
workspace path and returning `Mapping[str, CodexMcpServer]`. Import
`CodexMcpStdioServer`, `CodexMcpHttpServer`, `CodexMcpServer` and
`CodexMcpStartupError` from `band.integrations.codex.mcp`.

- `CodexMcpStdioServer(command, args=(), env={}, cwd=None)` describes a stdio
  server. Arguments and environment values are copied and frozen. Relative
  commands and arguments are interpreted in the server's `cwd`. Nonempty `env`
  uses the SDK's Python interpreter to restore generated environment aliases
  before replacing itself with the server process; native protected environment
  names are refused.
- `CodexMcpHttpServer(url, bearer_token=None, headers={})` describes streamable
  HTTP. SSE and other descriptor types are refused before constructing a client.
  Names must match `[A-Za-z0-9_-]+`, be unique, and not be the native reserved name
  `codex_apps`.
- Resolution happens once for each new client, after any unusable client has
  closed. Reusing an initialized client does not resolve again. To rotate or revoke
  a grant on an already running client, clean up the room first; the next startup
  gets a fresh snapshot. An empty mapping adds no servers or readiness requests.
- The returned mapping is frozen before startup. Neither adapter configuration
  nor `codex_env` is changed. Validated native TOML overrides are appended after
  existing Codex arguments, including after a relay's `--` boundary. Each supplied
  server is enabled, required, and uses connection readiness. Extra native servers
  are not changed.
- Bearer, header and stdio environment values go into a private child environment
  through generated variable names, never into the generated TOML or argv.
  Generated names also reserve inherited environment names.
  Secret fields are excluded from descriptor repr, equality and supported
  Pydantic serialization. These runtime objects are not persistence records.
- Before publishing initialization or starting a thread, the adapter pages
  `mcpServerStatus/list` under one 30-second deadline. Every supplied server must
  have server information and capabilities, no tool-discovery error, and a null
  pre-thread runtime status or `connected` status. Tool-less servers are allowed;
  auth configuration alone and an empty catalog are not readiness proof.
  Malformed pages, duplicate names, repeated cursors and negative states fail.
  Deadline expiry is checked even if a transport consumes its cancellation.
- Cancellation and retirement fence startup after each stage and before every
  readiness request, including thread-error recovery. A resolver may request
  cleanup of its own room without waiting on itself. One invocation owner adopts
  an already-delivered cancellation and disarms readiness deadlines on cleanup.
  Settlement beyond the cleanup grace reports incomplete cleanup without another
  cancellation; the client, workspace claim and turn release remain owned until
  settlement finishes. Retry room cleanup to close and release the settled client.
  Failed close retains the client, workspace claim and diagnostic redactor.
  Failed worker registration unwinds the gated worker and its ownership records.
  Invocation-only cleanup timers are closed before control returns to a caller,
  so they cannot cancel unrelated work that caller performs later.
  `CodexMcpStartupError.stage` identifies a secret-free startup or validation
  category.
- With adopted servers, thread start/resume errors trigger a bounded readiness
  check, rather than classification by error-message wording. Failed readiness
  refuses the history fallback and marks the client unusable; the next attempt
  closes it and resolves a fresh grant. Positive readiness preserves genuine
  missing-history fallback and unrelated thread errors.

Scoped redaction covers process stderr, malformed RPC diagnostics and forwarded
RPC errors. Error frames are sanitized before routing to a waiting request, so
consumption after owned teardown cannot expose the released secret scope.
Model selection, skill roots, native
sandbox/approval settings, real resume and Band's `dynamicTools` path are unchanged.
