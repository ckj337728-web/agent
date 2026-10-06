"""T004 验证：配置管理。

覆盖 spec 6.2 E7（缺少配置时给出明确提示）与 C-7/C-8（密钥不落仓库）。
"""

from __future__ import annotations

import unittest

from tests.helpers import AgentTestCase  # noqa: F401  (导入即完成 sys.path 设置)
from agent.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MAX_RETRIES,
    ENV_MODEL,
    ENV_TIMEOUT,
    ConfigError,
    LLMConfig,
)


class TestFromEnv(unittest.TestCase):
    def test_reads_all_values_from_env(self):
        config = LLMConfig.from_env(
            {
                ENV_API_KEY: "sk-abc",
                ENV_BASE_URL: "https://api.example.com/v1/",
                ENV_MODEL: "gpt-x",
                ENV_TIMEOUT: "12.5",
                ENV_MAX_RETRIES: "3",
            }
        )
        self.assertEqual(config.api_key, "sk-abc")
        self.assertEqual(config.base_url, "https://api.example.com/v1/")
        self.assertEqual(config.model, "gpt-x")
        self.assertEqual(config.timeout_seconds, 12.5)
        self.assertEqual(config.max_retries, 3)

    def test_defaults_when_optional_missing(self):
        config = LLMConfig.from_env({ENV_API_KEY: "sk-abc"})
        self.assertEqual(config.base_url, DEFAULT_BASE_URL)
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.timeout_seconds, DEFAULT_TIMEOUT_SECONDS)
        self.assertEqual(config.max_retries, DEFAULT_MAX_RETRIES)

    def test_blank_optional_values_fall_back_to_defaults(self):
        config = LLMConfig.from_env(
            {ENV_API_KEY: "sk-abc", ENV_MODEL: "  ", ENV_TIMEOUT: ""}
        )
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.timeout_seconds, DEFAULT_TIMEOUT_SECONDS)

    def test_values_are_trimmed(self):
        config = LLMConfig.from_env({ENV_API_KEY: "  sk-abc  "})
        self.assertEqual(config.api_key, "sk-abc")


class TestMissingConfig(unittest.TestCase):
    """spec 6.2 E7：缺少配置必须明确报错，而非静默失败。"""

    def test_missing_api_key_raises_config_error(self):
        with self.assertRaises(ConfigError):
            LLMConfig.from_env({})

    def test_blank_api_key_raises_config_error(self):
        with self.assertRaises(ConfigError):
            LLMConfig.from_env({ENV_API_KEY: "   "})

    def test_error_message_is_actionable(self):
        with self.assertRaises(ConfigError) as ctx:
            LLMConfig.from_env({})
        message = str(ctx.exception)
        self.assertIn(ENV_API_KEY, message)          # 指明缺哪个变量
        self.assertIn("$env:LLM_API_KEY", message)   # 给出可照抄的示例
        self.assertIn(".env.example", message)       # 指向示例文件

    def test_invalid_number_raises_config_error(self):
        with self.assertRaises(ConfigError) as ctx:
            LLMConfig.from_env({ENV_API_KEY: "sk-abc", ENV_TIMEOUT: "abc"})
        self.assertIn(ENV_TIMEOUT, str(ctx.exception))

    def test_negative_retries_rejected(self):
        with self.assertRaises(ConfigError):
            LLMConfig.from_env({ENV_API_KEY: "sk-abc", ENV_MAX_RETRIES: "-1"})

    def test_zero_timeout_rejected(self):
        with self.assertRaises(ConfigError):
            LLMConfig.from_env({ENV_API_KEY: "sk-abc", ENV_TIMEOUT: "0"})


class TestDerivedValues(unittest.TestCase):
    def test_chat_completions_url(self):
        config = LLMConfig.from_env(
            {ENV_API_KEY: "sk-abc", ENV_BASE_URL: "https://api.example.com/v1"}
        )
        self.assertEqual(
            config.chat_completions_url,
            "https://api.example.com/v1/chat/completions",
        )

    def test_trailing_slash_does_not_double_up(self):
        config = LLMConfig.from_env(
            {ENV_API_KEY: "sk-abc", ENV_BASE_URL: "https://api.example.com/v1/"}
        )
        self.assertEqual(
            config.chat_completions_url,
            "https://api.example.com/v1/chat/completions",
        )

    def test_safe_summary_masks_api_key(self):
        """C-7/C-8：任何可打印摘要都不得泄漏完整密钥。"""
        secret = "sk-super-secret-value-1234"
        config = LLMConfig.from_env({ENV_API_KEY: secret})
        summary = config.safe_summary()
        self.assertNotIn(secret, summary)
        self.assertIn("1234", summary)  # 仅保留末 4 位便于确认身份

    def test_short_key_fully_masked(self):
        config = LLMConfig.from_env({ENV_API_KEY: "abc"})
        self.assertNotIn("abc", config.safe_summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
