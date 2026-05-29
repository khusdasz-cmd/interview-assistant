"""Audio capture, VAD, and recording control."""

import numpy as np

from interview_assistant.config import (
    VAD_SAMPLE_RATE,
    VAD_FRAME_MS,
    VAD_FRAME_SIZE,
    SILENCE_TIMEOUT,
    MIN_RECORD_SEC,
    MAX_RECORD_SEC,
)


# ── Device scanning ─────────────────────────────────────────────────────────


def scan_devices():
    """Scan available audio input devices (dedup, test multiple sample rates)."""
    import sounddevice as sd
    seen = set()
    devs = []
    for i, d in enumerate(sd.query_devices()):
        name = d["name"]
        if d["max_input_channels"] > 0 and name not in seen:
            seen.add(name)
            for sr in [16000, 44100, 48000]:
                try:
                    sd.check_input_settings(device=i, samplerate=sr)
                    devs.append((i, name))
                    break
                except Exception:
                    continue
    return devs


def default_input_device():
    """Get system default input device index."""
    import sounddevice as sd
    try:
        return sd.default.device[0]
    except Exception:
        return None


# ── Capture classes ─────────────────────────────────────────────────────────


class MicCapture:
    """Microphone capture via sounddevice."""

    def __init__(self, callback, device=None):
        self.callback = callback
        self.device = device
        self.running = False
        self.stream = None

    def start(self):
        import sounddevice as sd
        self.running = True

        def audio_cb(indata, frames, time_info, status):
            if self.running:
                audio = (indata[:, 0] * 32767).astype(np.int16)
                self.callback(audio.tobytes())

        self.stream = sd.InputStream(
            device=self.device,
            samplerate=VAD_SAMPLE_RATE, channels=1,
            dtype="float32", callback=audio_cb,
            blocksize=VAD_FRAME_SIZE * 2,
        )
        self.stream.start()

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


class StereoMixCapture:
    """Stereo mix capture via pyaudio."""

    def __init__(self, callback, device_index=None):
        self.callback = callback
        self.running = False
        self.stream = None
        self.resample_buf = b""

    def start(self):
        import pyaudio as pa
        self.running = True
        p = pa.PyAudio()

        target_sr = VAD_SAMPLE_RATE
        device_sr = 44100
        ratio = target_sr / device_sr

        def pa_callback(in_data, frame_count, time_info, status):
            if not self.running:
                return (None, pa.paComplete)
            self.resample_buf += in_data
            out = b""
            bytes_needed = int(len(self.resample_buf) / 2 * ratio) * 2
            if bytes_needed >= VAD_FRAME_SIZE * 2:
                samples = np.frombuffer(self.resample_buf, dtype=np.int16)
                target_len = int(len(samples) * ratio)
                indices = np.linspace(0, len(samples) - 1, target_len)
                resampled = np.interp(indices, np.arange(len(samples)),
                                      samples.astype(float)).astype(np.int16)
                out = resampled.tobytes()
                self.resample_buf = b""
                self.callback(out)
            return (in_data, pa.paContinue)

        try:
            self.stream = p.open(
                format=pa.paInt16, channels=1, rate=device_sr,
                input=True, input_device_index=self.device_index,
                frames_per_buffer=480, stream_callback=pa_callback,
            )
            self.stream.start_stream()
        except Exception as e:
            self.running = False
            raise RuntimeError(
                '立体声混音捕获失败: ' + str(e) + '\n'
                '请先在Windows声音设置中启用立体声混音:\n'
                '  右键托盘喇叭→声音→录制→右键空白处→\n'
                '  显示禁用的设备→启用立体声混音'
            ) from e
        self._pa = p

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass


# ── VAD Recorder ────────────────────────────────────────────────────────────


class VADRecorder:
    """Voice Activity Detection + auto/manual recording.

    PTT mode: manual start/stop via start_manual()/stop_manual().
    Auto mode: VAD detection with energy gating.
    """

    def __init__(self, suppress_user_speech=False):
        import webrtcvad as wvad
        self.vad = wvad.Vad(2)
        self.suppress_user_speech = suppress_user_speech

        self.energy_threshold = 25
        self.user_speech_energy = 800

        self.recording = False
        self.buffer = b""
        self.silence_frames = 0
        self.speech_frames = 0
        self.rec_frames = 0

        self.silence_max = int(SILENCE_TIMEOUT / (VAD_FRAME_MS / 1000))
        self.min_frames = int(MIN_RECORD_SEC / (VAD_FRAME_MS / 1000))
        self.max_frames = int(MAX_RECORD_SEC / (VAD_FRAME_MS / 1000))

        self.on_audio_ready = None

        self.ptt_mode = False
        self.ptt_recording = False
        self.current_energy = 0.0

    def set_thresholds(self, energy_threshold=None, user_speech_energy=None):
        if energy_threshold is not None:
            self.energy_threshold = max(1, int(energy_threshold))
        if user_speech_energy is not None:
            self.user_speech_energy = max(10, int(user_speech_energy))

    def start_manual(self):
        """PTT: start manual recording."""
        if self.ptt_recording:
            return
        self.ptt_recording = True
        self.recording = True
        self.buffer = b""
        self.speech_frames = 0
        self.silence_frames = 0
        self.rec_frames = 0

    def stop_manual(self):
        """PTT: stop manual recording and fire callback."""
        if not self.ptt_recording:
            return
        self.ptt_recording = False
        if self.recording and len(self.buffer) > VAD_FRAME_SIZE * 2:
            self.recording = False
            energy_peak = float(np.max(np.abs(
                np.frombuffer(self.buffer, dtype=np.int16))))
            cb = self.on_audio_ready
            data = self.buffer
            self.buffer = b""
            if cb:
                cb(data, energy_peak)
        else:
            self.recording = False
            self.buffer = b""

    def feed(self, pcm_bytes):
        offset = 0
        while offset + VAD_FRAME_SIZE * 2 <= len(pcm_bytes):
            frame = pcm_bytes[offset: offset + VAD_FRAME_SIZE * 2]
            offset += VAD_FRAME_SIZE * 2
            self._process(frame)

    def _frame_energy(self, frame):
        samples = np.frombuffer(frame, dtype=np.int16).astype(float)
        return float(np.sqrt(np.mean(samples ** 2)))

    def _process(self, frame):
        if self.ptt_mode:
            if self.ptt_recording and self.recording:
                self.buffer += frame
                self.current_energy = self._frame_energy(frame)
            return

        energy = self._frame_energy(frame)
        self.current_energy = energy

        if energy < self.energy_threshold:
            is_speech = False
        else:
            is_speech = self.vad.is_speech(frame, VAD_SAMPLE_RATE)

        if self.suppress_user_speech and energy > self.user_speech_energy:
            if self.recording:
                self._finish()
            return

        if not self.recording:
            if is_speech:
                self.recording = True
                self.buffer = frame
                self.speech_frames = 1
                self.silence_frames = 0
                self.rec_frames = 1
            return

        self.buffer += frame
        self.rec_frames += 1
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        if self.silence_frames >= self.silence_max and self.speech_frames >= self.min_frames:
            self._finish()
        elif self.rec_frames >= self.max_frames:
            self._finish()

    def _finish(self):
        data = self.buffer
        self.recording = False
        self.buffer = b""
        self.speech_frames = 0
        self.silence_frames = 0
        self.rec_frames = 0
        if self.on_audio_ready and len(data) > VAD_FRAME_SIZE * 2:
            energy_peak = float(np.max(np.abs(
                np.frombuffer(data, dtype=np.int16))))
            self.on_audio_ready(data, energy_peak)

    def reset(self):
        self.recording = False
        self.buffer = b""
        self.speech_frames = 0
        self.silence_frames = 0
        self.rec_frames = 0
