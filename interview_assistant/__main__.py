"""Entry point: python -m interview_assistant"""

import argparse
import os

from interview_assistant.llm import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
)
from interview_assistant.app import InterviewAssistant


def main():
    parser = argparse.ArgumentParser(description="面试助手 v2")
    parser.add_argument("--loopback", action="store_true",
                        help="使用立体声混音（仅捕获系统音频）")
    parser.add_argument("--provider", default="deepseek",
                        choices=["deepseek", "openai", "ollama"],
                        help="LLM后端: deepseek(默认), openai(兼容), ollama(本地)")
    parser.add_argument("--model", default=None,
                        help="模型名，默认: deepseek-chat / gpt-4o-mini / qwen2.5")
    parser.add_argument("--api-key", default=None,
                        help="API密钥（ollama不需要）")
    parser.add_argument("--base-url", default=None,
                        help="API地址（ollama默认 http://localhost:11434/v1）")
    args = parser.parse_args()

    from interview_assistant import llm
    llm.PROVIDER = args.provider
    if args.provider == "deepseek":
        llm.PROVIDER_MODEL = args.model or "deepseek-chat"
        llm.PROVIDER_KEY = args.api_key or DEEPSEEK_API_KEY
        llm.PROVIDER_URL = args.base_url or DEEPSEEK_BASE_URL
    elif args.provider == "openai":
        llm.PROVIDER_MODEL = args.model or "gpt-4o-mini"
        llm.PROVIDER_KEY = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        llm.PROVIDER_URL = args.base_url or "https://api.openai.com"
    elif args.provider == "ollama":
        llm.PROVIDER_MODEL = args.model or "qwen2.5"
        llm.PROVIDER_KEY = args.api_key or ""
        llm.PROVIDER_URL = args.base_url or "http://localhost:11434/v1"

    print(f"🔧 {args.provider} / {llm.PROVIDER_MODEL}")
    app = InterviewAssistant(loopback=args.loopback)
    app.run()


if __name__ == "__main__":
    main()
