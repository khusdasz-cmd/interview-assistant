"""LLM API calls — supports DeepSeek, OpenAI-compatible, and Ollama."""

import json
from urllib.request import Request, urlopen

from interview_assistant.config import load_config, get_profile

_CONFIG = load_config()

# API config (config.toml → module-level)
DEEPSEEK_API_KEY = _CONFIG.get("api", {}).get("key", "")
DEEPSEEK_BASE_URL = _CONFIG.get("api", {}).get("url", "https://api.deepseek.com")

# Runtime provider config (overridable via CLI)
PROVIDER = "deepseek"
PROVIDER_MODEL = _CONFIG.get("api", {}).get("model", "deepseek-chat")
PROVIDER_KEY = DEEPSEEK_API_KEY
PROVIDER_URL = DEEPSEEK_BASE_URL


def call_deepseek(messages, timeout=20.0):
    """Call LLM (supports deepseek/openai/ollama backends)."""
    data = json.dumps({
        "model": PROVIDER_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 800,
    }).encode()
    req = Request(
        f"{PROVIDER_URL}/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {PROVIDER_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        resp = urlopen(req, timeout=timeout)
        result = json.loads(resp.read().decode())
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[API错误: {e}]"


def build_context(extra=""):
    """Build system prompt with project experience."""
    ctx = f"""你的个人背景：
{get_profile()}
{extra}

规则：
1. 回答必须基于用户的真实项目经历，不要编造
2. 语言自然口语化，适合面试场景
3. 控制在合理长度内"""
    return ctx
