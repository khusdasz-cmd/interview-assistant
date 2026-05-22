"""
面试辅助工具 v2 — 双模式（模拟面试 + 实战辅助）

模式A - 模拟面试：粘贴JD → AI面试官提问 → 你语音回答 → AI评分反馈
模式B - 实战辅助：实时监听面试官提问 → 基于你的项目经历生成参考回答

语音隔离策略：
  默认用麦克风捕获（面试官声音从外放传入麦克风）
  检测到高能量音频（你的回答）时自动抑制，只转录安静时段的内容
  如需完全隔离，启用"立体声混音"后加 --loopback 参数

启动：
  python interview-assistant.py

热键:
  Esc - 退出
"""

import argparse
import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from urllib.request import Request, urlopen

import numpy as np

# ═══════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════

def load_config():
    path = os.path.join(os.path.dirname(__file__), "config.toml")
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        print(f"[警告] 未找到 {path}，使用默认配置")
        return {}
    except Exception as e:
        print(f"[警告] 读取 config.toml 失败: {e}")
        return {}

_CONFIG = load_config()

# API 配置（config.toml → 全局变量）
DEEPSEEK_API_KEY = _CONFIG.get("api", {}).get("key", "")
DEEPSEEK_BASE_URL = _CONFIG.get("api", {}).get("url", "https://api.deepseek.com")

# 运行时 provider 配置（CLI args 可在 main() 中覆盖）
PROVIDER = "deepseek"
PROVIDER_MODEL = _CONFIG.get("api", {}).get("model", "deepseek-chat")
PROVIDER_KEY = DEEPSEEK_API_KEY
PROVIDER_URL = DEEPSEEK_BASE_URL

WHISPER_MODEL = _CONFIG.get("audio", {}).get("whisper_model", "base")

# VAD 技术常量
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_FRAME_SIZE = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480
SILENCE_TIMEOUT = 1.5
MIN_RECORD_SEC = 0.5
MAX_RECORD_SEC = 25

_PROFILE_CACHE = None

def _load_profile():
    path = os.path.join(os.path.dirname(__file__), "profile.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[警告] 未找到 {path}，个人背景信息为空")
        return ""
    except Exception as e:
        print(f"[警告] 读取 profile.txt 失败: {e}")
        return ""

def get_profile():
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        _PROFILE_CACHE = _load_profile()
    return _PROFILE_CACHE


# ═══════════════════════════════════════════════════════════
# 2. DeepSeek API
# ═══════════════════════════════════════════════════════════

def call_deepseek(messages, timeout=20.0):
    """调用 LLM（支持多后端: deepseek/openai/ollama）。"""
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
    """构建包含项目经历的 system prompt。"""
    ctx = f"""你的个人背景：
{get_profile()}
{extra}

规则：
1. 回答必须基于用户的真实项目经历，不要编造
2. 语言自然口语化，适合面试场景
3. 控制在合理长度内"""
    return ctx


# ═══════════════════════════════════════════════════════════
# 3. 音频捕获
# ═══════════════════════════════════════════════════════════

def scan_devices():
    """扫描可用音频输入设备(去重,试多采样率)。"""
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
    """获取系统默认输入设备索引。"""
    import sounddevice as sd
    try:
        return sd.default.device[0]
    except Exception:
        return None

class MicCapture:
    """麦克风捕获（sounddevice）。"""

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
    """立体声混音捕获（pyaudio）。"""

    def __init__(self, callback, device_index=None):
        self.callback = callback
        self.running = False
        self.stream = None
        self.resample_buf = b""

    def start(self):
        import pyaudio as pa
        self.running = True
        p = pa.PyAudio()

        # 目标采样率 16000, 设备原生 44100
        target_sr = VAD_SAMPLE_RATE
        device_sr = 44100
        ratio = target_sr / device_sr

        def pa_callback(in_data, frame_count, time_info, status):
            if not self.running:
                return (None, pa.paComplete)
            # 拼接 + 重采样 (简单线性插值)
            self.resample_buf += in_data
            out = b""
            bytes_needed = int(len(self.resample_buf) / 2 * ratio) * 2
            if bytes_needed >= VAD_FRAME_SIZE * 2:
                samples = np.frombuffer(self.resample_buf, dtype=np.int16)
                target_len = int(len(samples) * ratio)
                indices = np.linspace(0, len(samples) - 1, target_len)
                resampled = np.interp(indices, np.arange(len(samples)), samples.astype(float)).astype(np.int16)
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


# ═══════════════════════════════════════════════════════════
# 4. VAD 录音控制器（含语音隔离）
# ═══════════════════════════════════════════════════════════

class VADRecorder:
    """
    语音活动检测 + 自动录音。
    PTT模式: 手动控制录音起止
    自动模式: VAD检测 + 能量门控
    """

    def __init__(self, suppress_user_speech=False):
        import webrtcvad as wvad
        self.vad = wvad.Vad(2)
        self.suppress_user_speech = suppress_user_speech

        # 可调阈值（实例变量，运行时修改立即生效）
        self.energy_threshold = 25      # 低于此值视为静音
        self.user_speech_energy = 800   # 助听模式高于此值丢弃

        self.recording = False
        self.buffer = b""
        self.silence_frames = 0
        self.speech_frames = 0
        self.rec_frames = 0

        self.silence_max = int(SILENCE_TIMEOUT / (VAD_FRAME_MS / 1000))
        self.min_frames = int(MIN_RECORD_SEC / (VAD_FRAME_MS / 1000))
        self.max_frames = int(MAX_RECORD_SEC / (VAD_FRAME_MS / 1000))

        self.on_audio_ready = None  # callback(audio_bytes, energy_peak)

        # PTT / 调试
        self.ptt_mode = False            # True=手动 False=自动(VAD)
        self.ptt_recording = False       # PTT模式下正在录音
        self.current_energy = 0.0        # 最新一帧能量（供UI轮询）

    def set_thresholds(self, energy_threshold=None, user_speech_energy=None):
        if energy_threshold is not None:
            self.energy_threshold = max(1, int(energy_threshold))
        if user_speech_energy is not None:
            self.user_speech_energy = max(10, int(user_speech_energy))

    def start_manual(self):
        """PTT: 开始手动录音。"""
        if self.ptt_recording:
            return
        self.ptt_recording = True
        self.recording = True
        self.buffer = b""
        self.speech_frames = 0
        self.silence_frames = 0
        self.rec_frames = 0

    def stop_manual(self):
        """PTT: 停止手动录音并处理。"""
        if not self.ptt_recording:
            return
        self.ptt_recording = False
        if self.recording and len(self.buffer) > VAD_FRAME_SIZE * 2:
            self.recording = False
            energy_peak = float(np.max(np.abs(np.frombuffer(self.buffer, dtype=np.int16))))
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
        # PTT模式: 空闲完全闭麦, 录音时纯缓存无VAD
        if self.ptt_mode:
            if self.ptt_recording and self.recording:
                self.buffer += frame
                self.current_energy = self._frame_energy(frame)
            return

        energy = self._frame_energy(frame)
        self.current_energy = energy

        # === 自动(VAD)模式 ===
        if energy < self.energy_threshold:
            is_speech = False
        else:
            is_speech = self.vad.is_speech(frame, VAD_SAMPLE_RATE)

        # 助听模式: 高能量帧（用户自己说话）丢弃
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
            energy_peak = float(np.max(np.abs(np.frombuffer(data, dtype=np.int16))))
            self.on_audio_ready(data, energy_peak)

    def reset(self):
        self.recording = False
        self.buffer = b""
        self.speech_frames = 0
        self.silence_frames = 0
        self.rec_frames = 0


# ═══════════════════════════════════════════════════════════
# 5. Whisper 转写
# ═══════════════════════════════════════════════════════════

_whisper = None


def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        cache = os.path.join(os.path.dirname(__file__), ".whisper_cache")
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", download_root=cache)
    return _whisper


def transcribe(pcm_bytes):
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    model = get_whisper()
    segments, _ = model.transcribe(
        audio, language="zh", beam_size=3,
        initial_prompt=""
    )
    text = " ".join(s.text for s in segments).strip()
    try:
        from zhconv import convert
        text = convert(text, "zh-cn")
    except Exception:
        pass
    return text


# ═══════════════════════════════════════════════════════════
# 6. 主界面
# ═══════════════════════════════════════════════════════════

class InterviewAssistant:
    MODES = ("mock", "assist")

    def __init__(self, loopback=False):
        self.loopback = loopback

        self.capture = None
        self.recorder = None

        # 模拟面试状态
        self.mock_questions = []
        self.mock_index = 0
        self.mock_history = []  # [(q, a, feedback), ...]
        self.mock_running = False
        self.mock_messages = []  # LLM对话历史（模拟面试）
        self.assist_messages = []  # LLM对话历史（实战辅助）
        self._processing = False  # 防并发处理锁
        self.audio_devices = scan_devices()
        self.audio_device_idx = self.audio_devices[0][0] if self.audio_devices else None
        self.wave_buffer = [0] * 100  # 波形可视化的采样缓冲

        # 先建根窗口，再创建 tk 变量
        self.root = tk.Tk()
        self.current_mode = tk.StringVar(value="mock")

        self._build_ui()
        self._start_audio()
        # 默认手动模式
        self.root.after(500, lambda: self._toggle_ptt())

        # 后台预加载 Whisper 模型，避免首次录音等待
        threading.Thread(target=get_whisper, daemon=True).start()

    # ── UI ──
    def _build_ui(self):
        self.root.title("面试助手 v2")
        self.root.geometry("720x700+120+50")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1a1a2e")

        # 模式切换栏
        mode_frame = tk.Frame(self.root, bg="#16213e")
        mode_frame.pack(fill=tk.X)
        for text, val in [("🎯 模拟面试", "mock"), ("👂 实战辅助", "assist")]:
            rb = tk.Radiobutton(
                mode_frame, text=text, variable=self.current_mode,
                value=val, command=self._on_mode_switch,
                font=("Microsoft YaHei UI", 12, "bold"),
                bg="#16213e", fg="#e0e0e0", selectcolor="#1a1a2e",
                indicatoron=False, padx=20, pady=8, relief=tk.FLAT,
            )
            rb.pack(side=tk.LEFT, padx=2)
        tk.Label(mode_frame, text="空格=录音 | F3=静音 | Esc=退出",
                 font=("Microsoft YaHei UI", 8), fg="#666", bg="#16213e",
                 padx=8, pady=8).pack(side=tk.RIGHT)

        # 主容器（用来切换不同模式的内容）
        self.main = tk.Frame(self.root, bg="#1a1a2e")
        self.main.pack(fill=tk.BOTH, expand=True)

        # -- 模拟面试界面 --
        self.mock_frame = tk.Frame(self.main, bg="#1a1a2e")

        # JD 输入
        jd_header = tk.Frame(self.mock_frame, bg="#2d2d44")
        jd_header.pack(fill=tk.X, padx=8, pady=(8, 0))
        tk.Label(jd_header, text="📋 粘贴职位描述 (JD)", font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#ccc", bg="#2d2d44", padx=8, pady=4).pack(side=tk.LEFT)
        self.jd_btn = tk.Button(jd_header, text="开始面试", command=self._start_mock,
                                bg="#00ff88", fg="#000", font=("Microsoft YaHei UI", 9, "bold"),
                                padx=12, relief=tk.FLAT, cursor="hand2")
        self.jd_btn.pack(side=tk.RIGHT, padx=4, pady=4)

        self.jd_text = scrolledtext.ScrolledText(
            self.mock_frame, height=4, font=("Microsoft YaHei UI", 10),
            bg="#2d2d44", fg="#ddd", relief=tk.FLAT, padx=8, pady=6,
        )
        self.jd_text.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.jd_text.insert(tk.END, "在此粘贴岗位描述...")

        # 对话区域
        chat_frame = tk.Frame(self.mock_frame, bg="#1a1a2e")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 面试官
        tk.Label(chat_frame, text="🤵 AI 面试官", font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#ffd700", bg="#1a1a2e").pack(anchor=tk.W)
        self.interviewer_text = scrolledtext.ScrolledText(
            chat_frame, height=3, font=("Microsoft YaHei UI", 11),
            bg="#2d2d44", fg="#fff", relief=tk.FLAT, padx=8, pady=6,
        )
        self.interviewer_text.pack(fill=tk.X, pady=(0, 6))

        # 我的回答
        tk.Label(chat_frame, text="🙋 你的回答", font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#00ff88", bg="#1a1a2e").pack(anchor=tk.W)
        self.my_answer_text = scrolledtext.ScrolledText(
            chat_frame, height=3, font=("Microsoft YaHei UI", 11),
            bg="#1e2a3a", fg="#eee", relief=tk.FLAT, padx=8, pady=6,
        )
        self.my_answer_text.pack(fill=tk.X, pady=(0, 6))

        # AI 评价
        tk.Label(chat_frame, text="📊 AI 评价 & 建议", font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#ff9f43", bg="#1a1a2e").pack(anchor=tk.W)
        self.feedback_text = scrolledtext.ScrolledText(
            chat_frame, height=5, font=("Microsoft YaHei UI", 11),
            bg="#1a1a2e", fg="#e0e0e0", relief=tk.FLAT, padx=8, pady=6,
        )
        self.feedback_text.pack(fill=tk.BOTH, expand=True)

        # 保存报告按钮
        btn_row = tk.Frame(self.mock_frame, bg="#1a1a2e")
        btn_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Button(btn_row, text="💾 保存报告", command=self._save_report,
                  bg="#2d2d44", fg="#aaa", font=("Microsoft YaHei UI", 8),
                  padx=8, relief=tk.FLAT, cursor="hand2").pack(side=tk.RIGHT)

        self.mock_frame.pack(fill=tk.BOTH, expand=True)

        # -- 实战辅助界面 --
        self.assist_frame = tk.Frame(self.main, bg="#1a1a2e")

        tk.Label(self.assist_frame, text="🎤 面试官问题", font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#ccc", bg="#2d2d44", padx=8, pady=4, anchor=tk.W).pack(fill=tk.X)
        self.assist_question = scrolledtext.ScrolledText(
            self.assist_frame, height=4, font=("Microsoft YaHei UI", 11),
            bg="#2d2d44", fg="#fff", relief=tk.FLAT, padx=8, pady=6,
        )
        self.assist_question.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(self.assist_frame, text="💡 参考回答", font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#00ff88", bg="#1a1a2e", padx=8, pady=4, anchor=tk.W).pack(fill=tk.X)
        self.assist_answer = scrolledtext.ScrolledText(
            self.assist_frame, height=10, font=("Microsoft YaHei UI", 11),
            bg="#1a1a2e", fg="#e0e0e0", relief=tk.FLAT, padx=8, pady=6, wrap=tk.WORD,
        )
        self.assist_answer.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 面试情景
        self.assist_ctx_frame = tk.Frame(self.assist_frame, bg="#1e2a3a")
        self.assist_ctx_frame.pack(fill=tk.X, padx=8, pady=2)
        ctx_hdr = tk.Frame(self.assist_ctx_frame, bg="#1e2a3a")
        ctx_hdr.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(ctx_hdr, text="🏢 面试情景", font=("Microsoft YaHei UI", 8, "bold"),
                 fg="#ffd700", bg="#1e2a3a").pack(side=tk.LEFT)
        self.assist_ctx_btn = tk.Button(ctx_hdr, text="收起 ▲", command=self._toggle_assist_ctx,
                                        bg="#1e2a3a", fg="#888", font=("Microsoft YaHei UI", 7),
                                        padx=4, relief=tk.FLAT, cursor="hand2")
        self.assist_ctx_btn.pack(side=tk.RIGHT)
        self.assist_ctx_content = tk.Frame(self.assist_ctx_frame, bg="#1e2a3a")
        self.assist_ctx_content.pack(fill=tk.X, padx=6)
        row1 = tk.Frame(self.assist_ctx_content, bg="#1e2a3a")
        row1.pack(fill=tk.X, padx=0, pady=1)
        tk.Label(row1, text="公司", font=("Microsoft YaHei UI", 8), fg="#aaa", bg="#1e2a3a").pack(side=tk.LEFT)
        self.assist_company = tk.Entry(row1, font=("Microsoft YaHei UI", 9), bg="#2d2d44", fg="#eee", relief=tk.FLAT, width=20)
        self.assist_company.pack(side=tk.LEFT, padx=4)
        tk.Label(row1, text="岗位", font=("Microsoft YaHei UI", 8), fg="#aaa", bg="#1e2a3a").pack(side=tk.LEFT, padx=(10,0))
        self.assist_position = tk.Entry(row1, font=("Microsoft YaHei UI", 9), bg="#2d2d44", fg="#eee", relief=tk.FLAT, width=20)
        self.assist_position.pack(side=tk.LEFT, padx=4)
        tk.Label(self.assist_ctx_content, text="JD/备注", font=("Microsoft YaHei UI", 8), fg="#aaa", bg="#1e2a3a").pack(anchor=tk.W, padx=6)
        self.assist_jd = tk.Text(self.assist_ctx_content, height=2, font=("Microsoft YaHei UI", 9), bg="#2d2d44", fg="#eee", relief=tk.FLAT)
        self.assist_jd.pack(fill=tk.X, padx=6, pady=(0,4))

        # ── 底部控制栏 ──
        control_frame = tk.Frame(self.root, bg="#0d0d1a")
        control_frame.pack(fill=tk.X)

        # 左: PTT按钮
        self.ptt_mode = tk.BooleanVar(value=False)
        self.rec_btn = tk.Button(
            control_frame, text="🎤 自动(VAD)", command=self._toggle_ptt,
            bg="#2d2d44", fg="#ccc", font=("Microsoft YaHei UI", 9, "bold"),
            padx=10, relief=tk.FLAT, cursor="hand2",
        )
        self.rec_btn.pack(side=tk.LEFT, padx=6, pady=4)

        self.rec_indicator = tk.Label(
            control_frame, text="● 待命", font=("Microsoft YaHei UI", 9, "bold"),
            fg="#888", bg="#0d0d1a", padx=6,
        )
        self.rec_indicator.pack(side=tk.LEFT)

        # 中: 音频波形
        self.wave_canvas = tk.Canvas(control_frame, width=180, height=48,
                                     bg="#0a0a1a", highlightthickness=0)
        self.wave_canvas.pack(side=tk.LEFT, padx=6, pady=4)
        self.wave_canvas.create_line(0, 16, 160, 16, fill="#1a3a2a", tags="baseline")
        self.wave_line = self.wave_canvas.create_line(
            [0]*202, fill="#00ff88", width=2, tags="wave")

        # 右: 设置 + 退出
        self.settings_btn = tk.Button(
            control_frame, text="⚙ 阈值", command=self._toggle_settings,
            bg="#2d2d44", fg="#888", font=("Microsoft YaHei UI", 8),
            padx=8, relief=tk.FLAT, cursor="hand2",
        )
        self.settings_btn.pack(side=tk.RIGHT, padx=4, pady=4)


        # 状态文字
        self.status_var = tk.StringVar(value="⏳ 启动中...")
        tk.Label(control_frame, textvariable=self.status_var,
                 font=("Microsoft YaHei UI", 8), fg="#666", bg="#0d0d1a",
                 padx=6).pack(side=tk.RIGHT)

        # ── 设置面板 (可折叠) ──
        self.settings_frame = tk.Frame(self.root, bg="#16213e")
        self.settings_visible = False

        tk.Label(self.settings_frame, text="静音阈值", font=("Microsoft YaHei UI", 8),
                 fg="#aaa", bg="#16213e").grid(row=0, column=0, padx=8, pady=(8,0), sticky=tk.W)
        self.e_thresh_var = tk.IntVar(value=25)
        tk.Scale(self.settings_frame, from_=1, to=200, orient=tk.HORIZONTAL,
                 variable=self.e_thresh_var, command=self._on_threshold_change,
                 length=200, bg="#16213e", fg="#eee", highlightthickness=0,
                 font=("Microsoft YaHei UI", 7)).grid(row=0, column=1, padx=8, pady=(8,0))

        tk.Label(self.settings_frame, text="用户语音阈值(助听模式)", font=("Microsoft YaHei UI", 8),
                 fg="#aaa", bg="#16213e").grid(row=1, column=0, padx=8, sticky=tk.W)
        self.user_thresh_var = tk.IntVar(value=800)
        tk.Scale(self.settings_frame, from_=100, to=10000, orient=tk.HORIZONTAL,
                 variable=self.user_thresh_var, command=self._on_threshold_change,
                 length=200, bg="#16213e", fg="#eee", highlightthickness=0,
                 font=("Microsoft YaHei UI", 7)).grid(row=1, column=1, padx=8)

        # ── 音频输入 ──
        tk.Label(self.settings_frame, text="音频输入", font=("Microsoft YaHei UI", 8),
                 fg="#aaa", bg="#16213e").grid(row=2, column=0, padx=8, sticky=tk.W)
        dev_names = [n for _, n in self.audio_devices] if self.audio_devices else ["无设备"]
        self.audio_device_var = tk.StringVar(value=dev_names[0])
        self.audio_combo = ttk.Combobox(self.settings_frame, values=dev_names,
                                         textvariable=self.audio_device_var, state="readonly",
                                         font=("Microsoft YaHei UI", 8))
        self.audio_combo.grid(row=2, column=1, padx=8, pady=2, sticky=tk.EW)
        self.audio_combo.bind("<<ComboboxSelected>>", self._on_device_change)

        # ── LLM 设置 ──
        sep = tk.Frame(self.settings_frame, bg="#2d2d44", height=1)
        sep.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=8, pady=6)

        tk.Label(self.settings_frame, text="Model", font=("Microsoft YaHei UI", 8),
                 fg="#aaa", bg="#16213e").grid(row=4, column=0, padx=8, sticky=tk.W)
        self.model_et = tk.Entry(self.settings_frame, font=("Microsoft YaHei UI",9),
                                 bg="#2d2d44", fg="#eee", relief=tk.FLAT)
        self.model_et.insert(0, PROVIDER_MODEL)
        self.model_et.grid(row=4, column=1, padx=8, pady=2, sticky=tk.EW)

        tk.Label(self.settings_frame, text="API Key", font=("Microsoft YaHei UI", 8),
                 fg="#aaa", bg="#16213e").grid(row=5, column=0, padx=8, sticky=tk.W)
        self.key_et = tk.Entry(self.settings_frame, font=("Microsoft YaHei UI",9),
                               bg="#2d2d44", fg="#eee", relief=tk.FLAT, show="*")
        self.key_et.insert(0, PROVIDER_KEY)
        self.key_et.grid(row=5, column=1, padx=8, pady=2, sticky=tk.EW)

        tk.Label(self.settings_frame, text="API URL", font=("Microsoft YaHei UI", 8),
                 fg="#aaa", bg="#16213e").grid(row=6, column=0, padx=8, sticky=tk.W)
        self.url_et = tk.Entry(self.settings_frame, font=("Microsoft YaHei UI",9),
                               bg="#2d2d44", fg="#eee", relief=tk.FLAT)
        self.url_et.insert(0, PROVIDER_URL)
        self.url_et.grid(row=6, column=1, padx=8, pady=2, sticky=tk.EW)
        tk.Label(self.settings_frame, text="OpenAI 兼容格式 / Ollama 填 http://localhost:11434/v1",
                 font=("Microsoft YaHei UI", 7), fg="#666", bg="#16213e"
                 ).grid(row=7, column=1, padx=8, sticky=tk.W)

        tk.Button(self.settings_frame, text="应用", command=self._apply_llm_config,
                  bg="#00ff88", fg="#000", font=("Microsoft YaHei UI",8,"bold"),
                  padx=16, relief=tk.FLAT, cursor="hand2"
                  ).grid(row=8, column=1, padx=8, pady=(4,8), sticky=tk.E)

        self.settings_frame.columnconfigure(1, weight=1)

        # 热键
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<space>", self._on_space)
        self.root.bind("<F3>", self._toggle_mute)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        # 能量轮询
        self._poll_energy()

    # ── 音频 ──
    def _start_audio(self):
        """根据当前模式启动音频。"""
        # 模拟面试: 需要捕获用户声音 → 不抑制
        # 实战辅助+loopback: 只有面试官声音 → 不需要抑制
        # 实战辅助+mic: 用户声音需要抑制
        suppress = (self.current_mode.get() == "assist" and not self.loopback)

        try:
            self.recorder = VADRecorder(suppress_user_speech=suppress)
            audio_cfg = _CONFIG.get("audio", {})
            if "energy_threshold" in audio_cfg:
                self.recorder.energy_threshold = audio_cfg["energy_threshold"]
            if "user_speech_energy" in audio_cfg:
                self.recorder.user_speech_energy = audio_cfg["user_speech_energy"]
            self.recorder.on_audio_ready = self._on_audio_captured

            if self.loopback:
                self.capture = StereoMixCapture(self._audio_cb, device_index=self.audio_device_idx)
                self.status_var.set("🔊 系统音频捕获 (立体声混音)")
            else:
                self.capture = MicCapture(self._audio_cb, device=self.audio_device_idx)
                if suppress:
                    self.status_var.set("🎤 麦克风 (静音模式,期待面试官声音)")
                else:
                    self.status_var.set("🎤 麦克风录制中...")
            self.capture.start()
        except Exception as e:
            self.status_var.set(f"🔴 {e}")

    def _restart_audio(self):
        """模式切换时重启音频，保留 PTT 和静音状态。"""
        was_ptt = self.recorder.ptt_mode if self.recorder else False
        was_muted = getattr(self, "_muted", False)
        if self.capture:
            self.capture.stop()
        if self.recorder:
            self.recorder.reset()
        self._start_audio()
        # 强制同步指示灯
        if was_muted:
            self._muted = True
            self.rec_indicator.config(text="🔇 已静音", fg="#ff4444")
        elif was_ptt:
            self.recorder.reset()  # 清VAD残留
            self.recorder.ptt_mode = True
            self.rec_btn.config(text="🎤 手动(PTT)", bg="#e74c3c", fg="#fff")
            self.rec_indicator.config(text="⏹ 空闲", fg="#ff9f43")
        else:
            self.rec_btn.config(text="🎤 自动(VAD)", bg="#2d2d44", fg="#ccc")

    def _audio_cb(self, pcm_bytes):
        if self.recorder and not getattr(self, '_muted', False):
            self.recorder.feed(pcm_bytes)

    def _on_audio_captured(self, audio_bytes, energy_peak):
        """音频录制完成 → 转写 + 处理（防并发重入）。"""
        if self._processing:
            self.status_var.set(f"⏭ 已丢弃 (正在处理上一条, 能量:{energy_peak:.0f})")
            return
        self._processing = True
        self.status_var.set(f"📝 转写中... (能量: {energy_peak:.0f})")
        self.root.update_idletasks()

        # 整个过程放到线程里，不阻塞 VAD
        def process():
            try:
                text = transcribe(audio_bytes)
            except Exception as e:
                self._set_status_async(f"🔴 转写失败: {e}")
                self._processing = False
                return

            if not text or len(text) < 2:
                self._set_status_async(f"🎤 监听中 (能量峰值:{energy_peak:.0f})")
                self._processing = False
                return

            text = text.strip()
            self._set_status_async(f"💡 已转写 ({len(text)}字, 能量:{energy_peak:.0f})")

            if self.current_mode.get() == "mock":
                self._handle_mock_answer(text)
            else:
                self._handle_assist_question(text)
            self._processing = False

        threading.Thread(target=process, daemon=True).start()

    def _set_status_async(self, msg):
        """线程安全地更新状态栏。"""
        self.root.after(0, lambda: self.status_var.set(msg))


    def _start_mock(self):
        jd = self.jd_text.get(1.0, tk.END).strip()
        if not jd or jd == "在此粘贴岗位描述...":
            self._set_feedback("请先粘贴职位描述 (JD)")
            return

        self.jd_btn.config(state=tk.DISABLED, text="准备中...")
        self._set_feedback("正在分析 JD 并生成面试问题...")
        self.root.update_idletasks()

        def generate():
            try:
                # 初始化对话历史（system prompt）
                _prompt_cfg = _CONFIG.get("prompts", {})
                mock_role = _prompt_cfg.get("mock_role", "你是面试官，负责对候选人进行结构化面试")
                mock_rules = _prompt_cfg.get("mock_rules", "- 前 3 题围绕 JD 核心要求\n- 中 2 题结合候选人经历与 JD 的匹配点\n- 最后 1 题行为面试（STAR）\n- 每个问题要自然、口语化，像真实面试官在提问\n- 输出格式：每行一个问题，不要编号前缀以外的内容")
                system_prompt = f"""{mock_role}。根据以下 JD 和候选人的经历，生成 6 个面试问题。

JD：
{jd}

候选人经历：
{get_profile()}

要求：
{mock_rules}"""
                self.mock_messages = [{"role": "system", "content": system_prompt}]
                resp = call_deepseek(self.mock_messages, timeout=25)
                self.mock_questions = [q.strip() for q in resp.strip().split("\n") if q.strip() and len(q.strip()) > 5]
                if not self.mock_questions:
                    self.mock_questions = ["请先做个自我介绍。"]
                self.mock_index = 0
                self.mock_history = []
                self._show_mock_question()
            except Exception as e:
                self._set_feedback(f"生成问题失败: {e}")
            finally:
                self.jd_btn.config(state=tk.NORMAL, text="重新面试")

        threading.Thread(target=generate, daemon=True).start()

    def _show_mock_question(self):
        if self.mock_index >= len(self.mock_questions):
            self._end_mock()
            return

        q = self.mock_questions[self.mock_index]
        self._set_interviewer(f"第 {self.mock_index + 1}/{len(self.mock_questions)} 题\n\n{q}")
        self._set_answer("")
        self._set_feedback("🎤 请口头回答这个问题...")

        self.mock_running = True
        self.status_var.set(f"🎤 等待回答 (第{self.mock_index+1}/{len(self.mock_questions)})")

    def _handle_mock_answer(self, text):
        if not self.mock_running:
            return
        self.mock_running = False
        self._set_answer(text)
        self._set_feedback("📊 AI 评价中...")
        self.status_var.set("💡 生成反馈...")
        self.root.update_idletasks()

        q = self.mock_questions[self.mock_index]

        def evaluate():
            # 追加用户回答到对话历史
            self.mock_messages.append({"role": "user", "content": f"面试官问：{q}\n\n我的回答：{text}"})
            # 用完整历史评价
            prompt = self.mock_messages + [
                {"role": "system", "content": f"请对候选人上一条回答给出评分(1-10)、优点、改进建议和参考回答示范，简洁口语化。JD要求：{self.jd_text.get(1.0, tk.END).strip()[:300]}"},
            ]
            feedback = call_deepseek(prompt, timeout=20)
            self.mock_messages.append({"role": "assistant", "content": feedback})
            self._set_feedback(feedback)
            self.mock_history.append((q, text, feedback))
            self.status_var.set(f"✅ 第{self.mock_index+1}题完成")

            # 自动进入下一题
            self.mock_index += 1
            self.root.after(2000, self._show_mock_question)

        threading.Thread(target=evaluate, daemon=True).start()

    def _end_mock(self):
        self._set_interviewer("🎉 面试结束！生成总结报告中...")
        self.status_var.set("📊 生成总结报告")

        def summarize():
            prompt = self.mock_messages + [
                {"role": "user", "content": "请根据以上整个面试过程，给出总结评估报告：\n1. 总体评分\n2. 表现优势 (2-3条)\n3. 待改进 (2-3条)\n4. 针对该岗位的准备建议"},
            ]
            summary = call_deepseek(prompt, timeout=20)
            self.mock_summary = summary
            self._set_feedback(f"📋 总结报告\n\n{summary}")
            self.status_var.set("✅ 模拟面试完成")

        threading.Thread(target=summarize, daemon=True).start()

    def _save_report(self):
        """保存面试记录到文件。"""
        import datetime
        content = f"=== 面试记录 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n\n"
        for i, (q, a, fb) in enumerate(self.mock_history, 1):
            content += f"第{i}题：{q}\n回答：{a}\n评价：{fb}\n\n"
        if hasattr(self, 'mock_summary'):
            content += f"总结报告：\n{self.mock_summary}\n"
        path = os.path.join(os.path.dirname(__file__), f"面试报告_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self._set_feedback(f"💾 已保存: {os.path.basename(path)}")

    # ── 实战辅助 ──
    def _handle_assist_question(self, text):
        self._set_assist_question(text)
        self._set_status_async("💡 生成参考回答...")

        company = self.assist_company.get().strip()
        position = self.assist_position.get().strip()
        jd_note = self.assist_jd.get(1.0, tk.END).strip()
        scene = ""
        if company or position or jd_note:
            parts = []
            for k, v in [("公司", company), ("岗位", position), ("JD", jd_note)]:
                if v:
                    parts.append(f"{k}：{v}")
            scene = "\n面试情景：\n" + "\n".join(parts)

        # 构建消息列表：system + 历史对话 + 当前问题
        _prompt_cfg = _CONFIG.get("prompts", {})
        assist_role = _prompt_cfg.get("assist_role", "你是面试辅助助手。根据用户的真实项目经历和面试情景，给出参考回答。")
        assist_rules = _prompt_cfg.get("assist_rules", "基于用户经历不编造、贴合公司岗位、口语化、200字以内")
        messages = [{"role": "system", "content": f"""{build_context()}{scene}

{assist_role}
要求：{assist_rules}"""}]
        # 带上历史对话，实现上下文连贯
        messages.extend(self.assist_messages)
        messages.append({"role": "user", "content": f"面试官问：{text}"})

        answer = call_deepseek(messages, timeout=15)

        # 保存本轮 Q&A 到历史
        self.assist_messages.append({"role": "user", "content": f"面试官问：{text}"})
        self.assist_messages.append({"role": "assistant", "content": answer})
        # 限制历史长度，避免超出上下文窗口（保留最近约 10 轮）
        if len(self.assist_messages) > 20:
            self.assist_messages = self.assist_messages[-20:]

        self._set_assist_answer(f"问题：{text}\n\n参考回答：\n{answer}")
        self._set_status_async("🎤 监听中... 回答就绪")


    # ── 模式切换 ──
    def _on_mode_switch(self):
        mode = self.current_mode.get()
        self.mock_frame.pack_forget()
        self.assist_frame.pack_forget()
        if mode == "mock":
            self.mock_frame.pack(fill=tk.BOTH, expand=True)
        else:
            self.assist_frame.pack(fill=tk.BOTH, expand=True)
        self._restart_audio()
        # 修正录音指示灯
        self.root.after(200, lambda: self._sync_rec_indicator())

    def _sync_rec_indicator(self):
        """同步录音指示灯到实际状态。"""
        if not self.recorder:
            return
        if getattr(self, '_muted', False):
            self.rec_indicator.config(text="🔇 已静音", fg="#ff4444")
        elif self.recorder.ptt_recording:
            self.rec_indicator.config(text="🔴 手动录音", fg="#ff0000")
        elif self.recorder.ptt_mode:
            self.rec_indicator.config(text="⏹ 空闲", fg="#ff9f43")
        else:
            self.rec_indicator.config(text="● 待命", fg="#888")

    # ── UI 辅助 ──
    def _set_interviewer(self, text):
        self.interviewer_text.delete(1.0, tk.END)
        self.interviewer_text.insert(1.0, text)

    def _set_answer(self, text):
        self.my_answer_text.delete(1.0, tk.END)
        self.my_answer_text.insert(1.0, text)

    def _set_feedback(self, text):
        self.feedback_text.insert(1.0, text + "\n\n")
        self.feedback_text.see(1.0)

    def _set_assist_question(self, text):
        self.assist_question.delete(1.0, tk.END)
        self.assist_question.insert(1.0, text)

    def _set_assist_answer(self, text):
        self.assist_answer.delete(1.0, tk.END)
        self.assist_answer.insert(1.0, text)

    # ── 控制栏 ──
    def _toggle_assist_ctx(self):
        if self.assist_ctx_content.winfo_ismapped():
            self.assist_ctx_content.pack_forget()
            self.assist_ctx_btn.config(text="展开 ▼")
        else:
            self.assist_ctx_content.pack(fill=tk.X, padx=6)
            self.assist_ctx_btn.config(text="收起 ▲")

    def _toggle_mute(self, event=None):
        """F3: 临时静音/取消静音（讲话时用）。"""
        self._muted = not getattr(self, '_muted', False)
        if self._muted:
            if self.recorder:
                self.recorder.reset()
            self.rec_indicator.config(text="🔇 已静音", fg="#ff4444")
            self.status_var.set("🔇 按 F3 恢复")
        else:
            self.rec_indicator.config(text="● 待命", fg="#888")
            self.status_var.set("🎤 已恢复")

    def _on_device_change(self, event=None):
        """切换音频输入设备。"""
        name = self.audio_device_var.get()
        for idx, n in self.audio_devices:
            if n == name:
                self.audio_device_idx = idx
                break
        self.status_var.set(f"🔄 切换: {name}")
        self._restart_audio()

    def _toggle_ptt(self):
        """切换 自动(VAD)/手动(PTT) 模式。"""
        if not self.recorder:
            return
        new_mode = not self.recorder.ptt_mode
        self.recorder.ptt_mode = new_mode
        if new_mode:
            # 进入PTT: 清理VAD残留状态
            if self.recorder.recording:
                self.recorder.reset()
            self.rec_btn.config(text="🎤 手动(PTT)", bg="#e74c3c", fg="#fff")
            self.rec_indicator.config(text="⏹ 空闲", fg="#ff9f43")
            self.status_var.set("手动模式: 按空格键录制/停止")
        else:
            self.rec_btn.config(text="🎤 自动(VAD)", bg="#2d2d44", fg="#ccc")
            if self.recorder.ptt_recording:
                self.recorder.stop_manual()
            if self.recorder.recording:
                self.recorder.reset()
            self.rec_indicator.config(text="● 待命", fg="#888")
            self.status_var.set("自动模式: VAD检测语音")

    def _on_space(self, event):
        """空格键: 录制开关（任何模式下都可用）。"""
        if not self.recorder:
            return
        if self.recorder.ptt_recording:
            self.recorder.stop_manual()
            self.rec_indicator.config(text="⏹ 已停止", fg="#e74c3c")
            self.status_var.set("处理中...")
        else:
            if not self.recorder.ptt_mode:
                self.recorder.ptt_mode = True
                self.rec_btn.config(text="🎤 手动(PTT)", bg="#e74c3c", fg="#fff")
            self.recorder.start_manual()
            self.rec_indicator.config(text="🔴 录音中", fg="#ff0000")
            self.status_var.set("🎙 录音中... 再按空格停止")
        return "break"

    def _toggle_settings(self):
        """显示/隐藏阈值设置面板。"""
        if self.settings_visible:
            self.settings_frame.pack_forget()
            self.settings_visible = False
            self.settings_btn.config(text="⚙ 阈值")
        else:
            self.settings_frame.pack(fill=tk.X)
            self.settings_visible = True
            self.settings_btn.config(text="⚙ 收起")

    def _on_threshold_change(self, val):
        """阈值滑块变化时同步到 recorder。"""
        if self.recorder:
            self.recorder.set_thresholds(
                energy_threshold=self.e_thresh_var.get(),
                user_speech_energy=self.user_thresh_var.get(),
            )

    def _apply_llm_config(self):
        """应用 LLM 设置（model / key / url）。"""
        global PROVIDER_MODEL, PROVIDER_KEY, PROVIDER_URL
        PROVIDER_MODEL = self.model_et.get().strip()
        PROVIDER_KEY = self.key_et.get().strip()
        PROVIDER_URL = self.url_et.get().strip().rstrip("/")
        self._set_status_async(f"✅ 已切换: {PROVIDER_MODEL}")

    def _poll_energy(self):
        """每 200ms 更新音频波形。"""
        if not hasattr(self, 'wave_canvas'):
            return
        energy = self.recorder.current_energy if self.recorder else 0

        # PTT闭麦或空闲: 波形归零, 固定绿色
        ptt_idle = self.recorder and self.recorder.ptt_mode and not self.recorder.ptt_recording and not self.recorder.recording
        val = 0 if ptt_idle else min(energy, 2000)
        self.wave_buffer.pop(0)
        self.wave_buffer.append(val)

        # 画波形 (基线偏下 65%)
        cw, ch = 180, 48
        mid = int(ch * 0.65)
        points = []
        import math
        for i, v in enumerate(self.wave_buffer):
            norm = min(1.0, math.log(1 + v / 10) / math.log(1 + 2000 / 10))
            x = int(i * cw / len(self.wave_buffer))
            y = mid - int(norm * (mid - 2))
            points.extend([x, y])
        self.wave_canvas.coords(self.wave_line, *points)

        # 颜色: PTT空闲固定绿, 否则随振幅变色
        color = "#00ff88" if (ptt_idle or energy < 25) else ("#ffd700" if energy < 800 else "#ff4444")
        self.wave_canvas.itemconfig(self.wave_line, fill=color)

        # 录音状态指示：PTT 模式优先于 VAD 录音
        if self.recorder:
            if self.recorder.ptt_recording:
                self.rec_indicator.config(text="🔴 手动录音", fg="#ff0000")
            elif self.recorder.ptt_mode:
                self.rec_indicator.config(text="⏹ 空闲", fg="#ff9f43")
            elif self.recorder.recording:
                self.rec_indicator.config(text="● 录音中", fg="#ff6600")
            else:
                self.rec_indicator.config(text="● 待命", fg="#888")

        self.wave_canvas.itemconfig(self.wave_line, fill=color)
        self.root.after(200, self._poll_energy)

    # ── 生命周期 ──
    def quit(self):
        if self.capture:
            self.capture.stop()
        if self.recorder:
            self.recorder.reset()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════
# 7. 入口
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="面试助手 v2")
    parser.add_argument("--loopback", action="store_true",
                        help="使用立体声混音（仅捕获系统音频，需要先启用 Stereo Mix）")
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

    global PROVIDER, PROVIDER_MODEL, PROVIDER_KEY, PROVIDER_URL
    PROVIDER = args.provider
    if args.provider == "deepseek":
        PROVIDER_MODEL = args.model or "deepseek-chat"
        PROVIDER_KEY = args.api_key or DEEPSEEK_API_KEY
        PROVIDER_URL = args.base_url or DEEPSEEK_BASE_URL
    elif args.provider == "openai":
        PROVIDER_MODEL = args.model or "gpt-4o-mini"
        PROVIDER_KEY = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        PROVIDER_URL = args.base_url or "https://api.openai.com"
    elif args.provider == "ollama":
        PROVIDER_MODEL = args.model or "qwen2.5"
        PROVIDER_KEY = args.api_key or ""
        PROVIDER_URL = args.base_url or "http://localhost:11434/v1"

    print(f"🔧 {args.provider} / {PROVIDER_MODEL}")
    app = InterviewAssistant(loopback=args.loopback)
    app.run()


if __name__ == "__main__":
    main()
