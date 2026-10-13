"""Secret-safe Codex transport diagnostics and forwarded RPC failures."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from band.integrations.codex import CodexJsonRpcError, CodexStdioClient
from band.integrations.codex.mcp import CodexMcpHttpServer, freeze_mcp_servers
from band.integrations.codex.rpc_base import BaseJsonRpcClient

SECRET = 'sentinel-"quoted"-\\escaped-credential'


def redactor() -> Any:
    return freeze_mcp_servers(
        {"private": CodexMcpHttpServer("http://localhost/mcp", bearer_token=SECRET)}
    ).redactor


async def test_stderr_and_malformed_rpc_keep_context_without_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="band.integrations.codex.stdio_client")
    caplog.set_level(logging.DEBUG, logger="band.integrations.codex.rpc_base")
    client = CodexStdioClient(redactor=redactor())
    reader = asyncio.StreamReader()
    reader.feed_data(f"connection failed {SECRET}\n".encode())
    reader.feed_eof()
    client._proc = SimpleNamespace(stderr=reader)
    await client._read_stderr_loop()
    await client._dispatch_rpc_message("malformed " + SECRET)
    await client._dispatch_rpc_message(json.dumps([SECRET]))
    await client._dispatch_rpc_message(json.dumps({"unexpected": SECRET}))
    rendered = caplog.text
    assert SECRET not in rendered
    assert json.dumps(SECRET)[1:-1] not in rendered
    assert "connection failed" in rendered
    assert "malformed" in rendered
    assert "unexpected" in rendered


async def test_rpc_error_message_and_nested_data_are_redacted() -> None:
    class Peer(BaseJsonRpcClient):
        async def _send_json(self, payload: dict[str, Any]) -> None:
            await self._dispatch_rpc_message(
                json.dumps(
                    {
                        "id": payload["id"],
                        "error": {
                            "code": -32000,
                            "message": "server denied " + SECRET,
                            "data": {SECRET: ["Bearer " + SECRET, {"cause": SECRET}]},
                        },
                    }
                )
            )

    peer = Peer(redactor=redactor())
    with pytest.raises(CodexJsonRpcError) as error:
        await peer.request("thread/start", {})
    assert error.value.code == -32000
    assert "server denied" in str(error.value)
    assert SECRET not in str(error.value)
    assert SECRET not in repr(error.value.data)
    assert json.dumps(SECRET)[1:-1] not in json.dumps(error.value.data)


async def test_success_results_and_tool_arguments_are_not_redacted() -> None:
    class Peer(BaseJsonRpcClient):
        async def _send_json(self, payload: dict[str, Any]) -> None:
            assert payload["params"] == {"arguments": {"token": SECRET}}
            await self._dispatch_rpc_message(
                json.dumps({"id": payload["id"], "result": {"content": [SECRET]}})
            )

    peer = Peer(redactor=redactor())
    result = await peer.request("tools/call", {"arguments": {"token": SECRET}})
    assert result == {"content": [SECRET]}
    await peer._dispatch_rpc_message(
        json.dumps(
            {
                "id": "server-request",
                "method": "item/tool/call",
                "params": {"arguments": {"token": SECRET}},
            }
        )
    )
    event = await peer.recv_event(timeout_s=1)
    assert event.params == {"arguments": {"token": SECRET}}
    assert event.raw["params"] == {"arguments": {"token": SECRET}}


async def test_rpc_error_delivered_before_close_is_safe_when_consumed_after() -> None:
    scope = redactor()
    client = CodexStdioClient(redactor=scope)
    sent = asyncio.Event()
    consume = asyncio.Event()
    outgoing: list[dict[str, Any]] = []

    def write_request(data: bytes) -> None:
        outgoing.append(json.loads(data))
        sent.set()

    async def drain() -> None:
        await consume.wait()

    async def wait() -> None:
        return None

    client._proc = SimpleNamespace(
        stdin=SimpleNamespace(write=write_request, drain=drain, close=lambda: None),
        wait=wait,
    )
    request = asyncio.create_task(client.request("thread/start", {}))
    try:
        await asyncio.wait_for(sent.wait(), timeout=1)
        await client._dispatch_rpc_message(
            json.dumps(
                {
                    "id": outgoing[0]["id"],
                    "error": {
                        "code": -32000,
                        "message": "server denied " + SECRET,
                        "data": {SECRET: ["Bearer " + SECRET, {"cause": SECRET}]},
                    },
                }
            )
        )
        assert not request.done()
        await client.close()
        assert scope.text(SECRET) == SECRET
        assert not request.done()
        consume.set()
        with pytest.raises(CodexJsonRpcError) as error:
            await asyncio.wait_for(request, timeout=1)
        assert error.value.code == -32000
        assert error.value.message == "server denied [REDACTED]"
        assert error.value.data == {
            "[REDACTED]": ["Bearer [REDACTED]", {"cause": "[REDACTED]"}]
        }
        assert SECRET not in str(error.value)
        assert json.dumps(SECRET)[1:-1] not in json.dumps(error.value.data)
    finally:
        consume.set()
        await asyncio.gather(request, return_exceptions=True)
        await client.close()


async def test_secret_scope_is_retained_until_physical_teardown() -> None:
    scope = redactor()
    client = CodexStdioClient(env={"GENERATED": SECRET}, redactor=scope)
    exited = asyncio.Event()

    class Process:
        stdin = None

        async def wait(self) -> None:
            await exited.wait()

    client._proc = Process()
    closing = asyncio.create_task(client.close())
    try:
        while client._close_task is None:
            await asyncio.sleep(0)
        assert scope.text(SECRET) != SECRET
        assert client.env is not None and client.env["GENERATED"] == SECRET
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert scope.text(SECRET) != SECRET
        exited.set()
        await client.close()
        assert client.env is None
        assert scope.text(SECRET) == SECRET
    finally:
        exited.set()
        await asyncio.gather(closing, return_exceptions=True)
        await client.close()
