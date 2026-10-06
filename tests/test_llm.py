"""T005 验证：LLM 客户端（超时、重试、错误归一化）+ T006 依赖的 trace 落点。

覆盖 spec 3.0（本层只管调用与错误归一化）、6.2 E4（失败可解释、
重试耗尽后不泄漏裸堆栈）。
"""

from __future__ import annotations

import io
import socket
import unittest
import urllib.error

from tests.helpers import (
    AgentTestCase,
    chat_completion_payload,
    make_client,
    tool_call,
)
from agent.llm import (
    ChatMessage,
    LLMAuthError,
    LLMBadRequestError,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMServerError,
    LLMTimeoutError,
    urllib_transport,
)
from agent.trace import Tracer


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


class TestSuccessfulCall(AgentTestCase):
    def test_returns_content(self):
        client, _ = self.make_client(
            [(200, chat_completion_payload(content="你好"))]
        )
        response = client.chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(response.content, "你好")
        self.assertFalse(response.has_tool_calls)
        self.assertEqual(response.attempts, 1)

    def test_parses_tool_calls(self):
        payload = chat_completion_payload(
            tool_calls=[tool_call("calculator", {"expression": "1+1"})],
            finish_reason="tool_calls",
        )
        client, _ = self.make_client([(200, payload)])
        response = client.chat([ChatMessage(role="user", content="算 1+1")])
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.tool_calls[0]["function"]["name"], "calculator")
        self.assertEqual(response.finish_reason, "tool_calls")

    def test_null_content_becomes_empty_string(self):
        payload = chat_completion_payload(tool_calls=[tool_call("search", {"q": "x"})])
        del payload["choices"][0]["message"]["content"]
        client, _ = self.make_client([(200, payload)])
        response = client.chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(response.content, "")

    def test_request_carries_model_messages_and_auth_header(self):
        client, transport = self.make_client(
            [(200, chat_completion_payload(content="ok"))]
        )
        client.chat([ChatMessage(role="system", content="S"), ChatMessage(role="user", content="U")])
        sent = transport.calls[0]
        self.assertEqual(sent["url"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(sent["body"]["model"], "test-model")
        self.assertEqual(
            [m["role"] for m in sent["body"]["messages"]], ["system", "user"]
        )
        self.assertEqual(sent["headers"]["Authorization"], "Bearer test-key-not-real")
        self.assertNotIn("tools", sent["body"])  # 不传工具时不应带 tools 字段

    def test_tools_are_passed_with_auto_choice(self):
        """spec 3.3：LLM 基于注册表生成的 Schema 自主决策。"""
        client, transport = self.make_client(
            [(200, chat_completion_payload(content="ok"))]
        )
        client.chat([ChatMessage(role="user", content="hi")], tools=SAMPLE_TOOLS)
        body = transport.calls[0]["body"]
        self.assertEqual(body["tools"], SAMPLE_TOOLS)
        self.assertEqual(body["tool_choice"], "auto")

    def test_timeout_setting_is_forwarded(self):
        client, transport = self.make_client(
            [(200, chat_completion_payload(content="ok"))], timeout_seconds=7.5
        )
        client.chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(transport.calls[0]["timeout"], 7.5)


class TestErrorNormalization(AgentTestCase):
    """spec 6.2 E4：各类失败都归一化为统一的 LLMError 家族。"""

    def test_401_is_auth_error_and_not_retriable(self):
        client, transport = self.make_client(
            [(401, {"error": {"message": "invalid api key"}})]
        )
        with self.assertRaises(LLMAuthError) as ctx:
            client.chat([ChatMessage(role="user", content="hi")])
        self.assertIn("invalid api key", str(ctx.exception))
        self.assertFalse(ctx.exception.retriable)
        self.assertEqual(len(transport.calls), 1)  # 鉴权失败不重试

    def test_400_is_bad_request_and_not_retriable(self):
        client, transport = self.make_client(
            [(400, {"error": {"message": "bad param"}})]
        )
        with self.assertRaises(LLMBadRequestError):
            client.chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(len(transport.calls), 1)

    def test_429_is_retriable_and_exhausts_retries(self):
        client, transport = self.make_client(
            [
                (429, {"error": {"message": "rate limited"}}),
                (429, {"error": {"message": "rate limited"}}),
                (429, {"error": {"message": "rate limited"}}),
            ],
            max_retries=2,
        )
        with self.assertRaises(LLMRateLimitError):
            client.chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(len(transport.calls), 3)  # 1 次 + 2 次重试

    def test_500_is_server_error_and_retriable(self):
        client, transport = self.make_client(
            [
                (500, {"error": {"message": "boom"}}),
                (200, chat_completion_payload(content="恢复了")),
            ],
            max_retries=1,
        )
        response = client.chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(response.content, "恢复了")
        self.assertEqual(response.attempts, 2)

    def test_timeout_raises_timeout_error(self):
        client, _ = self.make_client([LLMTimeoutError("LLM 请求超时（>5.0s）")])
        with self.assertRaises(LLMTimeoutError) as ctx:
            client.chat([ChatMessage(role="user", content="hi")])
        self.assertTrue(ctx.exception.retriable)

    def test_network_error_raises_network_error(self):
        client, _ = self.make_client([LLMNetworkError("LLM 网络错误：拒绝连接")])
        with self.assertRaises(LLMNetworkError):
            client.chat([ChatMessage(role="user", content="hi")])

    def test_retry_then_success_recovers(self):
        client, transport = self.make_client(
            [
                LLMTimeoutError("timeout"),
                (200, chat_completion_payload(content="第二次成功")),
            ],
            max_retries=2,
        )
        response = client.chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(response.content, "第二次成功")
        self.assertEqual(response.attempts, 2)
        self.assertEqual(len(transport.calls), 2)

    def test_all_errors_share_common_base_class(self):
        """上层只需捕获 LLMError 一个类型。"""
        for exc in (
            LLMAuthError("a"),
            LLMTimeoutError("b"),
            LLMNetworkError("c"),
            LLMRateLimitError("d"),
            LLMServerError("e"),
            LLMBadRequestError("f"),
            LLMResponseFormatError("g"),
        ):
            self.assertIsInstance(exc, LLMError)


class TestMalformedResponses(AgentTestCase):
    def test_non_object_payload(self):
        client, _ = self.make_client([(200, ["not", "an", "object"])])
        with self.assertRaises(LLMResponseFormatError):
            client.chat([ChatMessage(role="user", content="hi")])

    def test_missing_choices(self):
        client, _ = self.make_client([(200, {"model": "m"})])
        with self.assertRaises(LLMResponseFormatError):
            client.chat([ChatMessage(role="user", content="hi")])

    def test_empty_choices(self):
        client, _ = self.make_client([(200, {"choices": []})])
        with self.assertRaises(LLMResponseFormatError):
            client.chat([ChatMessage(role="user", content="hi")])

    def test_missing_message(self):
        client, _ = self.make_client([(200, {"choices": [{"finish_reason": "stop"}]})])
        with self.assertRaises(LLMResponseFormatError):
            client.chat([ChatMessage(role="user", content="hi")])

    def test_tool_calls_not_a_list(self):
        payload = chat_completion_payload(content="x")
        payload["choices"][0]["message"]["tool_calls"] = {"bad": True}
        client, _ = self.make_client([(200, payload)])
        with self.assertRaises(LLMResponseFormatError):
            client.chat([ChatMessage(role="user", content="hi")])


class TestTraceIntegration(AgentTestCase):
    """spec 3.7 / 6.1 N11：每次 LLM 调用都留下可回溯记录。"""

    def _jsonl_tracer(self):
        buffer = io.StringIO()
        return Tracer(stream=buffer, fmt="jsonl"), buffer

    def test_success_records_llm_call(self):
        tracer, buffer = self._jsonl_tracer()
        client, _ = self.make_client(
            [(200, chat_completion_payload(content="ok"))], tracer=tracer
        )
        client.chat([ChatMessage(role="user", content="hi")], trace_id="t1")
        self.assertIn('"event": "llm_call"', buffer.getvalue())
        record = tracer.records[0]
        self.assertEqual(record.trace_id, "t1")
        self.assertEqual(record.payload["model"], "test-model")
        self.assertIn("duration_ms", record.payload)

    def test_failure_records_error_with_attempt_info(self):
        tracer, buffer = self._jsonl_tracer()
        client, _ = self.make_client(
            [(500, {"error": {"message": "boom"}})], tracer=tracer, max_retries=0
        )
        with self.assertRaises(LLMServerError):
            client.chat([ChatMessage(role="user", content="hi")], trace_id="t2")
        self.assertIn("llm_call_failed", buffer.getvalue())
        payload = tracer.records[0].payload
        self.assertEqual(payload["error_type"], "LLMServerError")
        self.assertEqual(payload["attempt"], "1/1")
        self.assertTrue(payload["retriable"])  # 5xx 属可重试，但重试次数已耗尽


class TestUrllibTransport(unittest.TestCase):
    """真实传输层的异常映射——用 monkeypatch 覆盖，不触发真实网络。"""

    def setUp(self):
        self._original = urllib.request.urlopen

    def tearDown(self):
        urllib.request.urlopen = self._original

    def _raise(self, exc):
        def fake_urlopen(*args, **kwargs):
            raise exc

        urllib.request.urlopen = fake_urlopen

    def test_socket_timeout_maps_to_timeout_error(self):
        self._raise(socket.timeout("timed out"))
        with self.assertRaises(LLMTimeoutError):
            urllib_transport("https://x.invalid", {}, b"{}", 1.0)

    def test_url_error_with_timeout_reason_maps_to_timeout_error(self):
        self._raise(urllib.error.URLError(socket.timeout("timed out")))
        with self.assertRaises(LLMTimeoutError):
            urllib_transport("https://x.invalid", {}, b"{}", 1.0)

    def test_url_error_maps_to_network_error(self):
        self._raise(urllib.error.URLError("connection refused"))
        with self.assertRaises(LLMNetworkError):
            urllib_transport("https://x.invalid", {}, b"{}", 1.0)

    def test_os_error_maps_to_network_error(self):
        self._raise(OSError("broken pipe"))
        with self.assertRaises(LLMNetworkError):
            urllib_transport("https://x.invalid", {}, b"{}", 1.0)

    def test_http_error_returns_status_and_parsed_body(self):
        self._raise(
            urllib.error.HTTPError(
                "https://x.invalid",
                429,
                "Too Many Requests",
                {},
                io.BytesIO(b'{"error":{"message":"slow down"}}'),
            )
        )
        status, payload = urllib_transport("https://x.invalid", {}, b"{}", 1.0)
        self.assertEqual(status, 429)
        self.assertEqual(payload["error"]["message"], "slow down")

    def test_non_json_body_is_wrapped_not_raised(self):
        self._raise(
            urllib.error.HTTPError(
                "https://x.invalid", 500, "Server Error", {}, io.BytesIO(b"<html>")
            )
        )
        status, payload = urllib_transport("https://x.invalid", {}, b"{}", 1.0)
        self.assertEqual(status, 500)
        self.assertEqual(payload["raw"], "<html>")


if __name__ == "__main__":
    unittest.main(verbosity=2)
