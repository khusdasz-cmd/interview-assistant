"""Tests for audio.py — VAD recorder and utilities.

Heavy audio libraries (sounddevice, webrtcvad) are not available in CI.
Tests that need them are marked with pytest.mark.skip.
"""

import numpy as np
import pytest

from interview_assistant.config import VAD_FRAME_SIZE


class TestFrameEnergy:
    """_frame_energy is a pure function — no audio hardware needed."""

    def test_silence_frame_low_energy(self):
        from interview_assistant.audio import VADRecorder
        r = VADRecorder()
        # All zeros → energy ≈ 0
        zeros = b'\x00\x00' * VAD_FRAME_SIZE
        e = r._frame_energy(zeros)
        assert e < 1.0

    def test_loud_frame_high_energy(self):
        from interview_assistant.audio import VADRecorder
        r = VADRecorder()
        # Max amplitude int16 → high energy
        loud = b'\xff\x7f' * VAD_FRAME_SIZE
        e = r._frame_energy(loud)
        assert e > 10000

    def test_half_amplitude(self):
        from interview_assistant.audio import VADRecorder
        r = VADRecorder()
        half = b'\x00\x40' * VAD_FRAME_SIZE
        e = r._frame_energy(half)
        assert 1000 < e < 50000


class TestSetThresholds:
    def test_set_both(self):
        from interview_assistant.audio import VADRecorder
        r = VADRecorder()
        r.set_thresholds(energy_threshold=50, user_speech_energy=1000)
        assert r.energy_threshold == 50
        assert r.user_speech_energy == 1000

    def test_set_energy_only(self):
        from interview_assistant.audio import VADRecorder
        r = VADRecorder()
        r.set_thresholds(energy_threshold=30)
        assert r.energy_threshold == 30
        assert r.user_speech_energy == 800  # unchanged

    def test_min_clamp(self):
        from interview_assistant.audio import VADRecorder
        r = VADRecorder()
        r.set_thresholds(energy_threshold=0, user_speech_energy=0)
        assert r.energy_threshold == 1
        assert r.user_speech_energy == 10


class TestPTTMode:
    """PTT (push-to-talk) mode doesn't need VAD/audio hardware."""

    @pytest.fixture
    def recorder(self):
        from interview_assistant.audio import VADRecorder
        r = VADRecorder()
        r.ptt_mode = True
        return r

    def test_start_stop_manual(self, recorder):
        assert recorder.ptt_recording is False
        recorder.start_manual()
        assert recorder.ptt_recording is True
        assert recorder.recording is True
        recorder.stop_manual()
        assert recorder.ptt_recording is False

    def test_double_start_ignored(self, recorder):
        recorder.start_manual()
        recorder.start_manual()  # should not crash
        assert recorder.ptt_recording is True

    def test_double_stop_ignored(self, recorder):
        recorder.stop_manual()  # nothing started
        assert recorder.ptt_recording is False

    def test_ptt_does_not_call_vad(self, recorder):
        """In PTT mode, _process returns early without touching VAD."""
        recorder.start_manual()
        frame = b'\x00\x00' * VAD_FRAME_SIZE
        recorder.feed(frame)
        assert len(recorder.buffer) > 0

    def test_buffer_cleared_after_stop(self, recorder):
        recorder.start_manual()
        recorder.feed(b'\x00\x00' * VAD_FRAME_SIZE * 3)
        recorder.stop_manual()
        # Buffer should be cleared after stop_manual
        assert recorder.buffer == b""


class TestStartStopManualFlow:
    """End-to-end PTT with callback."""

    def test_callback_fired_on_stop(self):
        from interview_assistant.audio import VADRecorder
        results = []

        def cb(data, energy):
            results.append((data, energy))

        r = VADRecorder()
        r.ptt_mode = True
        r.on_audio_ready = cb
        r.start_manual()
        # Feed enough frames
        r.feed(b'\x01\x00' * VAD_FRAME_SIZE * 10)
        r.stop_manual()

        assert len(results) == 1
        audio, energy = results[0]
        assert len(audio) > 0
        assert energy > 0

    def test_no_callback_on_empty_buffer(self):
        from interview_assistant.audio import VADRecorder
        fired = []

        def cb(data, energy):
            fired.append(True)

        r = VADRecorder()
        r.ptt_mode = True
        r.on_audio_ready = cb
        r.start_manual()
        # Stop without feeding anything
        r.stop_manual()
        assert len(fired) == 0
