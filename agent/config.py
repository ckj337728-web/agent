"""配置管理：LLM API 参数一律从环境变量读取。

对应任务 T004，约束来自 spec.md：

- C-7 / C-8：仓库不得包含任何真实密钥，密钥只能经环境变量注入；
- 6.2 E7：缺少配置时必须给出**明确提示**，而不是静默失败或抛出裸堆栈。

本模块不依赖任何第三方库，也不会自动读取 .env 文件（保持零依赖）；
调用方需先把环境变量导出到进程环境中。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

# 环境变量名（与 .env.example 保持一致）
ENV_API_KEY = "LLM_API_KEY"
ENV_BASE_URL = "LLM_BASE_URL"
ENV_MODEL = "LLM_MODEL"
ENV_TIMEOUT = "LLM_TIMEOUT_SECONDS"
ENV_MAX_RETRIES = "LLM_MAX_RETRIES"

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2


class ConfigError(Exception):
    """配置缺失或非法。

    携带面向用户的完整提示文本，入口只需打印该文本即可，无需暴露堆栈。
    """


@dataclass(frozen=True)
class LLMConfig:
    """一次 LLM 调用所需的全部配置。"""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ConfigError(_missing_api_key_message())
        if not self.base_url or not self.base_url.strip():
            raise ConfigError(
                f"配置项 {ENV_BASE_URL} 不能为空。示例：{DEFAULT_BASE_URL}"
            )
        if self.timeout_seconds <= 0:
            raise ConfigError(
                f"配置项 {ENV_TIMEOUT} 必须为正数，当前值：{self.timeout_seconds}"
            )
        if self.max_retries < 0:
            raise ConfigError(
                f"配置项 {ENV_MAX_RETRIES} 不能为负数，当前值：{self.max_retries}"
            )

    @property
    def chat_completions_url(self) -> str:
        """OpenAI 兼容的 chat completions 端点。"""
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def safe_summary(self) -> str:
        """可安全打印的配置摘要——绝不包含密钥明文。"""
        return (
            f"model={self.model} base_url={self.base_url} "
            f"timeout={self.timeout_seconds}s max_retries={self.max_retries} "
            f"api_key={_mask_secret(self.api_key)}"
        )

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "LLMConfig":
        """从环境变量构造配置；缺失必填项时抛出 ConfigError。"""
        source = os.environ if env is None else env
        return cls(
            api_key=_read_str(source, ENV_API_KEY, default=""),
            base_url=_read_str(source, ENV_BASE_URL, default=DEFAULT_BASE_URL),
            model=_read_str(source, ENV_MODEL, default=DEFAULT_MODEL),
            timeout_seconds=_read_float(
                source, ENV_TIMEOUT, default=DEFAULT_TIMEOUT_SECONDS
            ),
            max_retries=_read_int(
                source, ENV_MAX_RETRIES, default=DEFAULT_MAX_RETRIES
            ),
        )


def _missing_api_key_message() -> str:
    return (
        "缺少必填配置：未设置环境变量 "
        f"{ENV_API_KEY}。\n"
        "本项目的密钥只能通过环境变量注入，不会写入仓库。请先设置后重试：\n"
        '  Windows PowerShell : $env:LLM_API_KEY="sk-..."\n'
        '  Linux / macOS      : export LLM_API_KEY="sk-..."\n'
        "可参考仓库根目录的 .env.example 查看全部可用配置项。"
    )


def _mask_secret(value: str) -> str:
    """仅保留末 4 位，其余打码；过短的密钥整体打码。"""
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


def _read_str(source: Mapping[str, str], name: str, default: str) -> str:
    """读取字符串配置；未设置或仅为空白时回退到默认值。"""
    raw = source.get(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def _read_float(source: Mapping[str, str], name: str, default: float) -> float:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(
            f"配置项 {name} 必须是数字，当前值：{raw!r}"
        ) from exc


def _read_int(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(
            f"配置项 {name} 必须是整数，当前值：{raw!r}"
        ) from exc
