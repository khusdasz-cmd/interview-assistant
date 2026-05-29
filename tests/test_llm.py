"""Tests for llm.py — API calls and context building."""

from interview_assistant import llm
from interview_assistant.config import get_profile


def test_build_context_includes_profile():
    ctx = llm.build_context()
    profile = get_profile()
    if profile:
        assert profile[:20] in ctx


def test_build_context_with_extra():
    ctx = llm.build_context(extra="测试额外内容")
    assert "测试额外内容" in ctx


def test_call_deepseek_error_returns_string():
    """When network fails, returns an error string starting with [API错误:."""
    result = llm.call_deepseek([{"role": "user", "content": "hi"}], timeout=0.001)
    assert isinstance(result, str)
    assert result.startswith("[API错误:")
