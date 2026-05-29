"""Configuration loading and shared constants."""

import os

_CONFIG = None


def load_config():
    """Load config.toml, return dict. Returns {} on failure."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml")
    try:
        import tomllib
        with open(path, "rb") as f:
            _CONFIG = tomllib.load(f)
    except FileNotFoundError:
        print(f"[警告] 未找到 {path}，使用默认配置")
        _CONFIG = {}
    except Exception as e:
        print(f"[警告] 读取 config.toml 失败: {e}")
        _CONFIG = {}
    return _CONFIG


_PROFILE_CACHE = None


def load_profile():
    """Load profile.txt, return string. Returns '' on failure."""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profile.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            _PROFILE_CACHE = f.read()
    except FileNotFoundError:
        print(f"[警告] 未找到 {path}，个人背景信息为空")
        _PROFILE_CACHE = ""
    except Exception as e:
        print(f"[警告] 读取 profile.txt 失败: {e}")
        _PROFILE_CACHE = ""
    return _PROFILE_CACHE


def get_profile():
    return load_profile()


# VAD constants
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480
SILENCE_TIMEOUT = 1.5
MIN_RECORD_SEC = 0.5
MAX_RECORD_SEC = 25
