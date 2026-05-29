"""Tests for config.py — configuration loading."""

import os
import tempfile

from interview_assistant import config


def test_load_config_returns_dict():
    cfg = config.load_config()
    assert isinstance(cfg, dict)


def test_get_profile_returns_string():
    profile = config.get_profile()
    assert isinstance(profile, str)


def test_vad_constants():
    assert config.VAD_SAMPLE_RATE == 16000
    assert config.VAD_FRAME_MS == 30
    assert config.VAD_FRAME_SIZE == 480  # 16000 * 30 / 1000
    assert config.MIN_RECORD_SEC == 0.5
    assert config.MAX_RECORD_SEC == 25
