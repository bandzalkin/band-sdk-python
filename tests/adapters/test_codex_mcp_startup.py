"""Runtime MCP startup ownership and positive readiness regressions."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from band.adapters.codex import CodexAdapter, CodexAdapterConfig
from band.integrations.codex import CodexJsonRpcError
from band.integrations.codex import mcp as codex_mcp
from band.integrations.codex.mcp import (
    CodexMcpHttpServer,
    CodexMcpLaunch,
    CodexMcpServer,
    CodexMcpStartupError,
    CodexMcpStdioServer,
)
from band.integrations.codex.types import CodexRequestMethod, CodexSessionState
from band.testing import FakeAgentTools, reported_failures
from tests.adapters.codexturns import FakeCodexClient
from tests.adapters.test_codex_adapter import make_platform_message
from tests.conftest import eager_tasks, requires_eager_tasks


async def test_failed_retirement_keeps_primary_startup_error(tmp_path: Path) -> None:
    primary = RuntimeError("startup failed")

    class FailedClient(FakeCodexClient):
        async def connect(self) -> None:
            raise primary

        async def close(self) -> None:
            raise RuntimeError("retirement failed")

    client = FailedClient()
    adapter = CodexAdapter(
        CodexAdapterConfig(workspace_for_room=lambda _: str(tmp_path))
    )
    room = adapter._room_client("a")
    adapter._active_room.set("a")
    room.client = client
    with pytest.raises(RuntimeError) as raised:
        await adapter._ensure_client_ready()
    assert raised.value is primary
    assert room.client is client
    assert room.unusable
    assert adapter._room_clients["a"] is room


async def test_retired_room_cannot_start_a_client(tmp_path: Path) -> None:
    client = FakeCodexClient()
    adapter = CodexAdapter(
        CodexAdapterConfig(workspace_for_room=lambda _: str(tmp_path))
    )
    room = adapter._room_client("a")
    adapter._active_room.set("a")
    adapter._closing_rooms.add("a")
    built = []

    def build(_launch: object) -> FakeCodexClient:
        built.append(client)
        return client

    adapter._build_client = build
    with pytest.raises(asyncio.CancelledError):
        await adapter._ensure_client_ready()
    assert built == []
    assert room.client is None
    assert not room.initialized


def ready_row(name: str = "private") -> dict[str, Any]:
    return {
        "name": name,
        "serverInfo": {"name": "fixture", "version": "1"},
        "serverCapabilities": {},
        "toolsError": None,
        "runtimeStatus": None,
        "tools": {},
        "authStatus": "bearerToken",
    }


class StatusClient(FakeCodexClient):
    def __init__(
        self,
        pages: list[dict[str, Any]] | None = None,
        checkpoint: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self.pages = list(pages if pages is not None else [{"data": [ready_row()]}])
        self.checkpoint = checkpoint

    async def pause(self, stage: str) -> None:
        if self.checkpoint is not None:
            await self.checkpoint(stage)

    async def connect(self) -> None:
        await self.pause("connect")
        await super().connect()

    async def initialize(self, **kwargs: Any) -> dict[str, Any]:
        await self.pause("initialize")
        return await super().initialize(**kwargs)

    async def request(
        self, method: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        if method == "mcpServerStatus/list":
            await self.pause("readiness")
            self.requests.append((method, params or {}))
            return self.pages.pop(0)
        await self.pause(
            "skills" if method == CodexRequestMethod.SKILLS_EXTRA_ROOTS_SET else "model"
        )
        return await super().request(method, params, **kwargs)


def mcp_adapter(
    tmp_path: Path,
    clients: list[FakeCodexClient],
    resolver: Callable[[str], Awaitable[Mapping[str, CodexMcpServer]]],
    **config: Any,
) -> tuple[CodexAdapter, list[CodexMcpLaunch | None]]:
    adapter = CodexAdapter(
        CodexAdapterConfig(
            workspace_for_room=lambda room: str(tmp_path / room), **config
        ),
        resolve_mcp_servers=resolver,
    )
    launches: list[CodexMcpLaunch | None] = []

    def build(launch: CodexMcpLaunch | None) -> FakeCodexClient:
        launches.append(launch)
        return clients[len(launches) - 1]

    adapter._build_client = build
    return adapter, launches


async def start(adapter: CodexAdapter, room_id: str = "a") -> None:
    adapter._room_client(room_id)
    adapter._active_room.set(room_id)
    await adapter._ensure_client_ready()


async def private_servers(_workspace: str) -> Mapping[str, CodexMcpServer]:
    return {
        "private": CodexMcpHttpServer(
            "http://localhost/mcp", bearer_token="secret-sentinel"
        )
    }


async def test_reuse_skips_resolution_and_retirement_resolves_fresh(
    tmp_path: Path,
) -> None:
    calls = []

    async def resolve(workspace: str) -> Mapping[str, CodexMcpServer]:
        calls.append(workspace)
        return await private_servers(workspace)

    first, second = StatusClient(), StatusClient()
    adapter, launches = mcp_adapter(tmp_path, [first, second], resolve)
    await start(adapter)
    original = adapter._room_clients["a"].client
    await start(adapter)
    assert adapter._room_clients["a"].client is original
    assert len(calls) == 1
    assert len(launches) == 1
    adapter._room_clients["a"].unusable = True
    await start(adapter)
    assert first.closed
    assert adapter._room_clients["a"].client is second
    assert calls == [str(tmp_path / "a"), str(tmp_path / "a")]
    await adapter.cleanup_all()


async def test_held_resolver_does_not_block_another_room(tmp_path: Path) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def resolve(workspace: str) -> Mapping[str, CodexMcpServer]:
        if Path(workspace).name == "a":
            entered.set()
            await release.wait()
        return await private_servers(workspace)

    adapter, launches = mcp_adapter(tmp_path, [StatusClient(), StatusClient()], resolve)
    waiting = asyncio.create_task(start(adapter, "a"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(start(adapter, "b"), timeout=1)
        assert adapter._room_clients["b"].initialized
        assert not adapter._room_clients["a"].initialized
        assert adapter._room_clients["a"].client is None
        assert not waiting.done()
        assert len(launches) == 1
        release.set()
        await asyncio.wait_for(waiting, timeout=1)
        assert adapter._room_clients["a"].initialized
        assert len(launches) == 2
    finally:
        release.set()
        await waiting
        await adapter.cleanup_all()


@pytest.mark.parametrize(
    "stage", ["resolution", "connect", "initialize", "skills", "model", "readiness"]
)
@pytest.mark.parametrize("swallow", [False, True])
async def test_cancelled_startup_cannot_publish(
    tmp_path: Path, stage: str, swallow: bool
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def checkpoint(current: str) -> None:
        if current == stage:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                if not swallow:
                    raise

    async def resolve(workspace: str) -> Mapping[str, CodexMcpServer]:
        await checkpoint("resolution")
        return await private_servers(workspace)

    client = StatusClient(checkpoint=checkpoint)
    adapter, launches = mcp_adapter(
        tmp_path, [client], resolve, skill_roots=[str(tmp_path)]
    )
    running = asyncio.create_task(start(adapter))
    try:
        await entered.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not adapter._room_clients["a"].initialized
        assert adapter._room_clients["a"].client is None
        assert adapter._room_threads == {}
        if stage == "resolution":
            assert launches == []
        else:
            assert client.closed
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await adapter.cleanup_all()


async def test_cancelled_at_admission_never_resolves(tmp_path: Path) -> None:
    calls = []

    async def resolve(workspace: str) -> Mapping[str, CodexMcpServer]:
        calls.append(workspace)
        return {}

    adapter, launches = mcp_adapter(tmp_path, [StatusClient()], resolve)

    async def cancelled() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await start(adapter)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.create_task(cancelled())
    assert calls == []
    assert launches == []
    await adapter.cleanup_all()


async def test_resolver_self_cleanup_unwinds_without_self_wait(tmp_path: Path) -> None:
    adapter: CodexAdapter

    async def resolve(workspace: str) -> Mapping[str, CodexMcpServer]:
        await adapter.on_cleanup("a")
        return await private_servers(workspace)

    adapter, launches = mcp_adapter(tmp_path, [StatusClient()], resolve)

    async def locked_start() -> None:
        room = adapter._room_client("a")
        adapter._active_room.set("a")
        async with room.rpc_lock:
            await adapter._ensure_client_ready()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(locked_start(), timeout=1)
    await asyncio.wait_for(adapter.on_cleanup("a"), timeout=1)
    assert launches == []
    assert "a" not in adapter._room_clients
    assert "a" not in adapter._closing_rooms


async def test_failed_close_retains_secret_scope_claim_and_primary_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class FailedClient(StatusClient):
        refuse_close = True

        async def initialize(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("primary secret-sentinel")

        async def close(self) -> None:
            if self.refuse_close:
                raise RuntimeError("cleanup secret-sentinel")
            await super().close()

    first = FailedClient()
    adapter, _ = mcp_adapter(tmp_path, [first, StatusClient()], private_servers)
    with pytest.raises(CodexMcpStartupError) as error:
        await start(adapter)
    assert error.value.stage == "initialize"
    assert "secret-sentinel" not in str(error.value)
    assert "secret-sentinel" not in caplog.text
    room = adapter._room_clients["a"]
    assert room.client is first
    assert room.mcp_snapshot is not None
    assert room.unusable
    assert adapter._workspaces.claim("a", room.workspace) == room.workspace
    first.refuse_close = False
    await start(adapter)
    assert first.closed
    assert room.initialized
    assert room.client is not first
    await adapter.cleanup_all()


async def test_status_pagination_requires_all_expected_names(tmp_path: Path) -> None:
    async def resolve(_workspace: str) -> Mapping[str, CodexMcpServer]:
        return {
            "one": CodexMcpHttpServer("http://localhost/one"),
            "two": CodexMcpHttpServer("http://localhost/two"),
        }

    client = StatusClient(
        [
            {"data": [ready_row("one")], "nextCursor": "next"},
            {"data": [ready_row("two")], "nextCursor": None},
        ]
    )
    adapter, _ = mcp_adapter(tmp_path, [client], resolve)
    await start(adapter)
    assert adapter._room_clients["a"].initialized
    assert [
        params for method, params in client.requests if method == "mcpServerStatus/list"
    ] == [
        {"cursor": None, "limit": 100, "detail": "full"},
        {"cursor": "next", "limit": 100, "detail": "full"},
    ]
    await adapter.cleanup_all()


@pytest.mark.parametrize(
    "page",
    [
        {},
        {"data": None},
        {"data": {}},
        {"data": [None]},
        {"data": [{"name": 7}]},
        {"data": []},
        {"data": [ready_row(), ready_row()]},
        {"data": [ready_row()], "nextCursor": 1},
        {"data": [ready_row()], "nextCursor": ""},
        {"data": [{"name": "private", "authStatus": "bearerToken", "tools": {}}]},
        {"data": [{**ready_row(), "serverInfo": None}]},
        {"data": [{**ready_row(), "serverInfo": {}}]},
        {"data": [{**ready_row(), "serverCapabilities": None}]},
        {"data": [{**ready_row(), "toolsError": "secret-sentinel"}]},
    ],
)
async def test_bad_catalog_never_marks_room_ready(
    tmp_path: Path, page: dict[str, Any]
) -> None:
    client = StatusClient([page])
    adapter, _ = mcp_adapter(tmp_path, [client], private_servers)
    with pytest.raises(CodexMcpStartupError) as error:
        await start(adapter)
    assert error.value.stage == "readiness"
    assert "secret-sentinel" not in str(error.value)
    assert not adapter._room_clients["a"].initialized
    assert adapter._room_threads == {}
    assert client.closed
    await adapter.cleanup_all()


@pytest.mark.parametrize(
    "status",
    [
        "notStarted",
        "starting",
        "failed",
        "authenticationRequired",
        "authRequired",
        "disabled",
        "cancelled",
        "garbage",
        False,
    ],
)
async def test_negative_runtime_status_is_not_readiness(
    tmp_path: Path, status: Any
) -> None:
    client = StatusClient([{"data": [{**ready_row(), "runtimeStatus": status}]}])
    adapter, _ = mcp_adapter(tmp_path, [client], private_servers)
    with pytest.raises(CodexMcpStartupError):
        await start(adapter)
    assert client.closed
    assert not adapter._room_clients["a"].initialized
    await adapter.cleanup_all()


async def test_repeated_cursor_is_refused(tmp_path: Path) -> None:
    client = StatusClient(
        [
            {"data": [ready_row()], "nextCursor": "same"},
            {"data": [], "nextCursor": "same"},
        ]
    )
    adapter, _ = mcp_adapter(tmp_path, [client], private_servers)
    with pytest.raises(CodexMcpStartupError):
        await start(adapter)
    assert client.closed
    assert (
        len(
            [
                method
                for method, _ in client.requests
                if method == "mcpServerStatus/list"
            ]
        )
        == 2
    )
    await adapter.cleanup_all()


@pytest.mark.parametrize(
    "method", [CodexRequestMethod.THREAD_START, CodexRequestMethod.THREAD_RESUME]
)
async def test_required_mcp_failure_never_starts_history_fallback(
    tmp_path: Path, method: CodexRequestMethod
) -> None:
    class RequiredFailureClient(StatusClient):
        failed = False

        async def request(
            self, sent: str, params: dict[str, Any] | None = None, **kwargs: Any
        ) -> dict[str, Any]:
            if sent == method:
                self.failed = True
                self.requests.append((sent, params or {}))
                raise CodexJsonRpcError(
                    code=-32002,
                    message="Required MCP startup failed; thread not found secret-sentinel",
                    data={"token": "secret-sentinel"},
                )
            if sent == "mcpServerStatus/list" and self.failed:
                self.requests.append((sent, params or {}))
                return {"data": [{**ready_row(), "runtimeStatus": "failed"}]}
            return await super().request(sent, params, **kwargs)

    client = RequiredFailureClient()
    adapter, _ = mcp_adapter(tmp_path, [client], private_servers)
    await start(adapter)
    history = (
        CodexSessionState(thread_id="saved")
        if method == CodexRequestMethod.THREAD_RESUME
        else CodexSessionState()
    )
    with pytest.raises(CodexMcpStartupError) as error:
        await adapter._ensure_thread(
            room_id="a",
            history=history,
            tools=FakeAgentTools(),
            is_session_bootstrap=True,
        )
    assert error.value.stage == method
    assert "secret-sentinel" not in str(error.value)
    assert "a" not in adapter._needs_history_injection
    assert adapter._room_threads == {}
    if method == CodexRequestMethod.THREAD_RESUME:
        assert CodexRequestMethod.THREAD_START not in client.request_methods
    await adapter.cleanup_all()


async def test_callback_exception_metadata_cannot_forge_startup_stage(
    tmp_path: Path,
) -> None:
    async def resolve(_workspace: str) -> Mapping[str, CodexMcpServer]:
        raise CodexMcpStartupError("connect")

    adapter, launches = mcp_adapter(tmp_path, [StatusClient()], resolve)
    with pytest.raises(CodexMcpStartupError) as error:
        await start(adapter)
    assert error.value.stage == "resolution"
    assert launches == []
    await adapter.cleanup_all()


async def test_empty_snapshot_has_no_mcp_readiness_requirement(tmp_path: Path) -> None:
    async def resolve(_workspace: str) -> Mapping[str, CodexMcpServer]:
        return {}

    client = FakeCodexClient()
    adapter, launches = mcp_adapter(tmp_path, [client], resolve)
    await start(adapter)
    assert adapter._room_clients["a"].initialized
    assert "mcpServerStatus/list" not in client.request_methods
    assert launches[0] is not None
    assert "-c" not in launches[0].command
    await adapter.cleanup_all()


async def test_snapshot_is_stable_while_connect_suspends(tmp_path: Path) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    servers: dict[str, CodexMcpServer] = {
        "private": CodexMcpHttpServer(
            "http://localhost/original", bearer_token="original-sentinel"
        )
    }

    async def checkpoint(stage: str) -> None:
        if stage == "connect":
            entered.set()
            await release.wait()

    async def resolve(_workspace: str) -> Mapping[str, CodexMcpServer]:
        return servers

    adapter, launches = mcp_adapter(
        tmp_path, [StatusClient(checkpoint=checkpoint)], resolve
    )
    running = asyncio.create_task(start(adapter))
    try:
        await entered.wait()
        servers.clear()
        servers["replacement"] = CodexMcpHttpServer(
            "http://localhost/replacement", bearer_token="replacement-sentinel"
        )
        release.set()
        await running
        room = adapter._room_clients["a"]
        assert room.mcp_snapshot is not None
        assert room.mcp_snapshot.expected_names == frozenset({"private"})
        launch = launches[0]
        assert launch is not None and launch.env is not None
        assert "original-sentinel" in launch.env.values()
        assert "replacement-sentinel" not in launch.env.values()
        assert room.initialized
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
        await adapter.cleanup_all()


@pytest.mark.looptime
async def test_readiness_pages_share_one_total_deadline(tmp_path: Path) -> None:
    class SlowPages(StatusClient):
        async def request(
            self, method: str, params: dict[str, Any] | None = None, **kwargs: Any
        ) -> dict[str, Any]:
            if method == "mcpServerStatus/list":
                self.requests.append((method, params or {}))
                await asyncio.sleep(20)
                cursor = params.get("cursor") if params is not None else None
                return {
                    "data": [ready_row()] if cursor is None else [],
                    "nextCursor": "next" if cursor is None else None,
                }
            return await super().request(method, params, **kwargs)

    client = SlowPages()
    adapter, _ = mcp_adapter(tmp_path, [client], private_servers)
    with pytest.raises(CodexMcpStartupError) as error:
        await start(adapter)
    assert error.value.stage == "readiness"
    assert not adapter._room_clients["a"].initialized
    assert client.closed
    await adapter.cleanup_all()


async def test_missing_history_still_resumes_via_fallback_with_mcp(
    tmp_path: Path,
) -> None:
    class MissingHistory(StatusClient):
        async def request(
            self, method: str, params: dict[str, Any] | None = None, **kwargs: Any
        ) -> dict[str, Any]:
            if method == CodexRequestMethod.THREAD_RESUME:
                self.requests.append((method, params or {}))
                raise CodexJsonRpcError(code=-32002, message="Thread not found")
            if method == "mcpServerStatus/list":
                self.requests.append((method, params or {}))
                return {"data": [ready_row()]}
            return await super().request(method, params, **kwargs)

    client = MissingHistory()
    adapter, _ = mcp_adapter(tmp_path, [client], private_servers)
    await start(adapter)
    thread = await adapter._ensure_thread(
        room_id="a",
        history=CodexSessionState(thread_id="missing"),
        tools=FakeAgentTools(),
        is_session_bootstrap=True,
    )
    assert thread == "thr-1"
    assert "a" in adapter._needs_history_injection
    params = client.params_of(CodexRequestMethod.THREAD_START)[0]
    assert "dynamicTools" in params
    assert params["cwd"] == str(tmp_path / "a")
    await adapter.cleanup_all()


class DuplicateServers(Mapping[str, Any]):
    """A caller mapping can yield the same name twice from items()."""

    def __getitem__(self, key: str) -> Any:
        if key != "private":
            raise KeyError(key)
        return CodexMcpHttpServer("http://localhost/mcp")

    def __iter__(self) -> Iterator[str]:
        return iter(("private", "private"))

    def __len__(self) -> int:
        return 2


class SseServer:
    transport = "sse"
    url = "http://localhost/sse"


@pytest.mark.parametrize(
    ("servers", "stage"),
    [
        ({"private": SseServer()}, "unsupported-transport"),
        ({"invalid.name": CodexMcpHttpServer("http://localhost/mcp")}, "invalid-name"),
        (DuplicateServers(), "duplicate-name"),
        ({"codex_apps": CodexMcpHttpServer("http://localhost/mcp")}, "reserved-name"),
    ],
)
async def test_invalid_resolver_result_refuses_before_construction(
    tmp_path: Path, servers: Mapping[str, Any], stage: str
) -> None:
    async def resolve(_workspace: str) -> Mapping[str, CodexMcpServer]:
        return servers

    adapter, launches = mcp_adapter(tmp_path, [StatusClient()], resolve)
    with pytest.raises(CodexMcpStartupError) as error:
        await start(adapter)
    assert error.value.stage == stage
    assert launches == []
    assert adapter._room_clients["a"].client is None
    await adapter.cleanup_all()


async def test_windows_secret_stdio_refuses_before_client_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(codex_mcp, "os", SimpleNamespace(name="nt", environ=os.environ))
    secret = "private-stdio-platform-grant"

    async def resolve(_workspace: str) -> Mapping[str, CodexMcpServer]:
        return {"private": CodexMcpStdioServer("runner", env={"TOKEN": secret})}

    adapter, launches = mcp_adapter(tmp_path, [StatusClient()], resolve)
    try:
        with pytest.raises(CodexMcpStartupError) as error:
            await start(adapter)
        assert error.value.stage == "unsupported-platform"
        assert secret not in str(error.value)
        assert launches == []
        assert adapter._room_clients["a"].client is None
    finally:
        await adapter.cleanup_all()


@requires_eager_tasks
async def test_cancelled_admission_refuses_before_eager_worker(tmp_path: Path) -> None:
    resolved = []

    async def resolve(workspace: str) -> Mapping[str, CodexMcpServer]:
        resolved.append(workspace)
        return {}

    adapter, launches = mcp_adapter(tmp_path, [], resolve)

    async def cancelled_caller() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        with pytest.raises(asyncio.CancelledError):
            await adapter.on_message(
                msg=make_platform_message(room_id="a"),
                tools=FakeAgentTools(),
                history=CodexSessionState(),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="a",
            )

    with eager_tasks():
        caller = asyncio.create_task(cancelled_caller())
        await caller
    assert resolved == []
    assert launches == []
    assert adapter._room_clients == {}
    assert adapter._turn_tasks == {}


async def test_callback_cancellation_cannot_leak_its_reason(tmp_path: Path) -> None:
    async def resolve(_workspace: str) -> Mapping[str, CodexMcpServer]:
        raise asyncio.CancelledError("private-callback-sentinel")

    adapter, launches = mcp_adapter(tmp_path, [], resolve)
    with pytest.raises(asyncio.CancelledError) as error:
        await start(adapter)
    assert error.value.args == ()
    assert launches == []
    await adapter.cleanup_all()


async def test_rpc_failure_forwarded_to_band_is_redacted(tmp_path: Path) -> None:
    class FailingTurnClient(StatusClient):
        async def request(
            self, method: str, params: dict[str, Any] | None = None, **kwargs: Any
        ) -> dict[str, Any]:
            if method == CodexRequestMethod.TURN_START:
                raise CodexJsonRpcError(
                    code=-32001,
                    message="Provider rejected secret-sentinel",
                    data={"nested": ["secret-sentinel"]},
                )
            return await super().request(method, params, **kwargs)

    adapter, _ = mcp_adapter(tmp_path, [FailingTurnClient()], private_servers)
    tools = FakeAgentTools()
    await adapter.on_started("Fixture", "RPC diagnostic fixture")
    with pytest.raises(CodexJsonRpcError) as error:
        await adapter.on_message(
            msg=make_platform_message(room_id="a"),
            tools=tools,
            history=CodexSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="a",
        )
    assert "secret-sentinel" not in str(error.value)
    assert error.value.data == {"nested": ["[REDACTED]"]}
    failures = reported_failures(tools)
    assert len(failures) == 1
    assert failures[0]["provider"] == "codex"
    assert "Provider rejected [REDACTED]" in failures[0]["message"]
    assert "secret-sentinel" not in str(failures)
    await adapter.cleanup_all()
