"""Main application — tkinter UI for mock interview and real-time assistance."""

import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

from interview_assistant.config import load_config, get_profile
from interview_assistant.llm import (
    PROVIDER_MODEL, PROVIDER_KEY, PROVIDER_URL,
    call_deepseek, build_context,
)
from interview_assistant.audio import (
    scan_devices, MicCapture, StereoMixCapture, VADRecorder,
)
from interview_assistant.transcriber import transcribe, get_whisper

_CONFIG = load_config()


class InterviewAssistant:
    MODES = ("mock", "assist")

    def __init__(self, loopback=False):
        self.loopback = loopback
        self.capture = None
        self.recorder = None

        self.mock_questions = []
        self.mock_index = 0
        self.mock_history = []
        self.mock_running = False
        self.mock_messages = []
        self.assist_messages = []
        self._processing = False
        self.audio_devices = scan_devices()
        self.audio_device_idx = self.audio_devices[0][0] if self.audio_devices else None
        self.wave_buffer = [0] * 100

        self.root = tk.Tk()
        self.current_mode = tk.StringVar(value="mock")

        self._build_ui()
        self._start_audio()
        self.root.after(500, lambda: self._toggle_ptt())
        threading.Thread(target=get_whisper, daemon=True).start()

    # ── UI ──
    def _build_ui(self):
        self.root.title("面试助手 v2")
        self.root.geometry("720x700+120+50")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1a1a2e")

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

        self.main = tk.Frame(self.root, bg="#1a1a2e")
        self.main.pack(fill=tk.BOTH, expand=True)

        # -- mock interview --
        self.mock_frame = tk.Frame(self.main, bg="#1a1a2e")

        jd_header = tk.Frame(self.mock_frame, bg="#2d2d44")
        jd_header.pack(fill=tk.X, padx=8, pady=(8, 0))
        tk.Label(jd_header, text="📋 粘贴职位描述 (JD)",
                 font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#ccc", bg="#2d2d44", padx=8, pady=4).pack(side=tk.LEFT)
        self.jd_btn = tk.Button(jd_header, text="开始面试", command=self._start_mock,
                                bg="#00ff88", fg="#000",
                                font=("Microsoft YaHei UI", 9, "bold"),
                                padx=12, relief=tk.FLAT, cursor="hand2")
        self.jd_btn.pack(side=tk.RIGHT, padx=4, pady=4)

        self.jd_text = scrolledtext.ScrolledText(
            self.mock_frame, height=4, font=("Microsoft YaHei UI", 10),
            bg="#2d2d44", fg="#ddd", relief=tk.FLAT, padx=8, pady=6,
        )
        self.jd_text.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.jd_text.insert(tk.END, "在此粘贴岗位描述...")

        chat_frame = tk.Frame(self.mock_frame, bg="#1a1a2e")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        tk.Label(chat_frame, text="🤵 AI 面试官",
                 font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#ffd700", bg="#1a1a2e").pack(anchor=tk.W)
        self.interviewer_text = scrolledtext.ScrolledText(
            chat_frame, height=3, font=("Microsoft YaHei UI", 11),
            bg="#2d2d44", fg="#fff", relief=tk.FLAT, padx=8, pady=6,
        )
        self.interviewer_text.pack(fill=tk.X, pady=(0, 6))

        tk.Label(chat_frame, text="🙋 你的回答",
                 font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#00ff88", bg="#1a1a2e").pack(anchor=tk.W)
        self.my_answer_text = scrolledtext.ScrolledText(
            chat_frame, height=3, font=("Microsoft YaHei UI", 11),
            bg="#1e2a3a", fg="#eee", relief=tk.FLAT, padx=8, pady=6,
        )
        self.my_answer_text.pack(fill=tk.X, pady=(0, 6))

        tk.Label(chat_frame, text="📊 AI 评价 & 建议",
                 font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#ff9f43", bg="#1a1a2e").pack(anchor=tk.W)
        self.feedback_text = scrolledtext.ScrolledText(
            chat_frame, height=5, font=("Microsoft YaHei UI", 11),
            bg="#1a1a2e", fg="#e0e0e0", relief=tk.FLAT, padx=8, pady=6,
        )
        self.feedback_text.pack(fill=tk.BOTH, expand=True)

        btn_row = tk.Frame(self.mock_frame, bg="#1a1a2e")
        btn_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Button(btn_row, text="💾 保存报告", command=self._save_report,
                  bg="#2d2d44", fg="#aaa", font=("Microsoft YaHei UI", 8),
                  padx=8, relief=tk.FLAT, cursor="hand2").pack(side=tk.RIGHT)

        self.mock_frame.pack(fill=tk.BOTH, expand=True)

        # -- assist mode --
        self.assist_frame = tk.Frame(self.main, bg="#1a1a2e")

        tk.Label(self.assist_frame, text="🎤 面试官问题",
                 font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#ccc", bg="#2d2d44", padx=8, pady=4, anchor=tk.W
                 ).pack(fill=tk.X)
        self.assist_question = scrolledtext.ScrolledText(
            self.assist_frame, height=4, font=("Microsoft YaHei UI", 11),
            bg="#2d2d44", fg="#fff", relief=tk.FLAT, padx=8, pady=6,
        )
        self.assist_question.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(self.assist_frame, text="💡 参考回答",
                 font=("Microsoft YaHei UI", 10, "bold"),
                 fg="#00ff88", bg="#1a1a2e", padx=8, pady=4, anchor=tk.W
                 ).pack(fill=tk.X)
        self.assist_answer = scrolledtext.ScrolledText(
            self.assist_frame, height=10, font=("Microsoft YaHei UI", 11),
            bg="#1a1a2e", fg="#e0e0e0", relief=tk.FLAT, padx=8, pady=6,
            wrap=tk.WORD,
        )
        self.assist_answer.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.assist_ctx_frame = tk.Frame(self.assist_frame, bg="#1e2a3a")
        self.assist_ctx_frame.pack(fill=tk.X, padx=8, pady=2)
        ctx_hdr = tk.Frame(self.assist_ctx_frame, bg="#1e2a3a")
        ctx_hdr.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(ctx_hdr, text="🏢 面试情景",
                 font=("Microsoft YaHei UI", 8, "bold"),
                 fg="#ffd700", bg="#1e2a3a").pack(side=tk.LEFT)
        self.assist_ctx_btn = tk.Button(
            ctx_hdr, text="收起 ▲", command=self._toggle_assist_ctx,
            bg="#1e2a3a", fg="#888", font=("Microsoft YaHei UI", 7),
            padx=4, relief=tk.FLAT, cursor="hand2")
        self.assist_ctx_btn.pack(side=tk.RIGHT)
        self.assist_ctx_content = tk.Frame(self.assist_ctx_frame, bg="#1e2a3a")
        self.assist_ctx_content.pack(fill=tk.X, padx=6)
        row1 = tk.Frame(self.assist_ctx_content, bg="#1e2a3a")
        row1.pack(fill=tk.X, padx=0, pady=1)
        tk.Label(row1, text="公司", font=("Microsoft YaHei UI", 8),
                 fg="#aaa", bg="#1e2a3a").pack(side=tk.LEFT)
        self.assist_company = tk.Entry(
            row1, font=("Microsoft YaHei UI", 9), bg="#2d2d44",
            fg="#eee", relief=tk.FLAT, width=20)
        self.assist_company.pack(side=tk.LEFT, padx=4)
        tk.Label(row1, text="岗位", font=("Microsoft YaHei UI", 8),
                 fg="#aaa", bg="#1e2a3a").pack(side=tk.LEFT, padx=(10, 0))
        self.assist_position = tk.Entry(
            row1, font=("Microsoft YaHei UI", 9), bg="#2d2d44",
            fg="#eee", relief=tk.FLAT, width=20)
        self.assist_position.pack(side=tk.LEFT, padx=4)
        tk.Label(self.assist_ctx_content, text="JD/备注",
                 font=("Microsoft YaHei UI", 8), fg="#aaa",
                 bg="#1e2a3a").pack(anchor=tk.W, padx=6)
        self.assist_jd = tk.Text(
            self.assist_ctx_content, height=2, font=("Microsoft YaHei UI", 9),
            bg="#2d2d44", fg="#eee", relief=tk.FLAT)
        self.assist_jd.pack(fill=tk.X, padx=6, pady=(0, 4))

        # ── control bar ──
        control_frame = tk.Frame(self.root, bg="#0d0d1a")
        control_frame.pack(fill=tk.X)

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

        self.wave_canvas = tk.Canvas(
            control_frame, width=180, height=48,
            bg="#0a0a1a", highlightthickness=0)
        self.wave_canvas.pack(side=tk.LEFT, padx=6, pady=4)
        self.wave_canvas.create_line(0, 16, 160, 16, fill="#1a3a2a",
                                     tags="baseline")
        self.wave_line = self.wave_canvas.create_line(
            [0] * 202, fill="#00ff88", width=2, tags="wave")

        self.settings_btn = tk.Button(
            control_frame, text="⚙ 阈值", command=self._toggle_settings,
            bg="#2d2d44", fg="#888", font=("Microsoft YaHei UI", 8),
            padx=8, relief=tk.FLAT, cursor="hand2",
        )
        self.settings_btn.pack(side=tk.RIGHT, padx=4, pady=4)

        self.status_var = tk.StringVar(value="⏳ 启动中...")
        tk.Label(control_frame, textvariable=self.status_var,
                 font=("Microsoft YaHei UI", 8), fg="#666", bg="#0d0d1a",
                 padx=6).pack(side=tk.RIGHT)

        # ── settings panel ──
        self.settings_frame = tk.Frame(self.root, bg="#16213e")
        self.settings_visible = False

        tk.Label(self.settings_frame, text="静音阈值",
                 font=("Microsoft YaHei UI", 8), fg="#aaa",
                 bg="#16213e").grid(row=0, column=0, padx=8, pady=(8, 0),
                                   sticky=tk.W)
        self.e_thresh_var = tk.IntVar(value=25)
        tk.Scale(self.settings_frame, from_=1, to=200, orient=tk.HORIZONTAL,
                 variable=self.e_thresh_var, command=self._on_threshold_change,
                 length=200, bg="#16213e", fg="#eee", highlightthickness=0,
                 font=("Microsoft YaHei UI", 7)
                 ).grid(row=0, column=1, padx=8, pady=(8, 0))

        tk.Label(self.settings_frame, text="用户语音阈值(助听模式)",
                 font=("Microsoft YaHei UI", 8), fg="#aaa",
                 bg="#16213e").grid(row=1, column=0, padx=8, sticky=tk.W)
        self.user_thresh_var = tk.IntVar(value=800)
        tk.Scale(self.settings_frame, from_=100, to=10000,
                 orient=tk.HORIZONTAL, variable=self.user_thresh_var,
                 command=self._on_threshold_change, length=200, bg="#16213e",
                 fg="#eee", highlightthickness=0,
                 font=("Microsoft YaHei UI", 7)
                 ).grid(row=1, column=1, padx=8)

        tk.Label(self.settings_frame, text="音频输入",
                 font=("Microsoft YaHei UI", 8), fg="#aaa",
                 bg="#16213e").grid(row=2, column=0, padx=8, sticky=tk.W)
        dev_names = [n for _, n in self.audio_devices] if self.audio_devices else ["无设备"]
        self.audio_device_var = tk.StringVar(value=dev_names[0])
        self.audio_combo = ttk.Combobox(
            self.settings_frame, values=dev_names,
            textvariable=self.audio_device_var, state="readonly",
            font=("Microsoft YaHei UI", 8))
        self.audio_combo.grid(row=2, column=1, padx=8, pady=2, sticky=tk.EW)
        self.audio_combo.bind("<<ComboboxSelected>>", self._on_device_change)

        sep = tk.Frame(self.settings_frame, bg="#2d2d44", height=1)
        sep.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=8, pady=6)

        tk.Label(self.settings_frame, text="Model",
                 font=("Microsoft YaHei UI", 8), fg="#aaa",
                 bg="#16213e").grid(row=4, column=0, padx=8, sticky=tk.W)
        self.model_et = tk.Entry(self.settings_frame,
                                 font=("Microsoft YaHei UI", 9),
                                 bg="#2d2d44", fg="#eee", relief=tk.FLAT)
        self.model_et.insert(0, PROVIDER_MODEL)
        self.model_et.grid(row=4, column=1, padx=8, pady=2, sticky=tk.EW)

        tk.Label(self.settings_frame, text="API Key",
                 font=("Microsoft YaHei UI", 8), fg="#aaa",
                 bg="#16213e").grid(row=5, column=0, padx=8, sticky=tk.W)
        self.key_et = tk.Entry(self.settings_frame,
                               font=("Microsoft YaHei UI", 9),
                               bg="#2d2d44", fg="#eee", relief=tk.FLAT,
                               show="*")
        self.key_et.insert(0, PROVIDER_KEY)
        self.key_et.grid(row=5, column=1, padx=8, pady=2, sticky=tk.EW)

        tk.Label(self.settings_frame, text="API URL",
                 font=("Microsoft YaHei UI", 8), fg="#aaa",
                 bg="#16213e").grid(row=6, column=0, padx=8, sticky=tk.W)
        self.url_et = tk.Entry(self.settings_frame,
                               font=("Microsoft YaHei UI", 9),
                               bg="#2d2d44", fg="#eee", relief=tk.FLAT)
        self.url_et.insert(0, PROVIDER_URL)
        self.url_et.grid(row=6, column=1, padx=8, pady=2, sticky=tk.EW)

        tk.Button(self.settings_frame, text="应用", command=self._apply_llm_config,
                  bg="#00ff88", fg="#000",
                  font=("Microsoft YaHei UI", 8, "bold"),
                  padx=16, relief=tk.FLAT, cursor="hand2"
                  ).grid(row=8, column=1, padx=8, pady=(4, 8), sticky=tk.E)

        self.settings_frame.columnconfigure(1, weight=1)

        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<space>", self._on_space)
        self.root.bind("<F3>", self._toggle_mute)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self._poll_energy()

    # ── Audio lifecycle ──
    def _start_audio(self):
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
                self.capture = StereoMixCapture(self._audio_cb,
                                                device_index=self.audio_device_idx)
                self.status_var.set("🔊 系统音频捕获 (立体声混音)")
            else:
                self.capture = MicCapture(self._audio_cb,
                                          device=self.audio_device_idx)
                if suppress:
                    self.status_var.set("🎤 麦克风 (静音模式,期待面试官声音)")
                else:
                    self.status_var.set("🎤 麦克风录制中...")
            self.capture.start()
        except Exception as e:
            self.status_var.set(f"🔴 {e}")

    def _restart_audio(self):
        was_ptt = self.recorder.ptt_mode if self.recorder else False
        was_muted = getattr(self, "_muted", False)
        if self.capture:
            self.capture.stop()
        if self.recorder:
            self.recorder.reset()
        self._start_audio()
        if was_muted:
            self._muted = True
            self.rec_indicator.config(text="🔇 已静音", fg="#ff4444")
        elif was_ptt:
            self.recorder.reset()
            self.recorder.ptt_mode = True
            self.rec_btn.config(text="🎤 手动(PTT)", bg="#e74c3c", fg="#fff")
            self.rec_indicator.config(text="⏹ 空闲", fg="#ff9f43")
        else:
            self.rec_btn.config(text="🎤 自动(VAD)", bg="#2d2d44", fg="#ccc")

    def _audio_cb(self, pcm_bytes):
        if self.recorder and not getattr(self, '_muted', False):
            self.recorder.feed(pcm_bytes)

    def _on_audio_captured(self, audio_bytes, energy_peak):
        if self._processing:
            self.status_var.set(
                f"⏭ 已丢弃 (正在处理上一条, 能量:{energy_peak:.0f})")
            return
        self._processing = True
        self.status_var.set(f"📝 转写中... (能量: {energy_peak:.0f})")
        self.root.update_idletasks()

        def process():
            try:
                text = transcribe(audio_bytes)
            except Exception as e:
                self._set_status_async(f"🔴 转写失败: {e}")
                self._processing = False
                return
            if not text or len(text) < 2:
                self._set_status_async(
                    f"🎤 监听中 (能量峰值:{energy_peak:.0f})")
                self._processing = False
                return
            text = text.strip()
            self._set_status_async(
                f"💡 已转写 ({len(text)}字, 能量:{energy_peak:.0f})")
            if self.current_mode.get() == "mock":
                self._handle_mock_answer(text)
            else:
                self._handle_assist_question(text)
            self._processing = False

        threading.Thread(target=process, daemon=True).start()

    def _set_status_async(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    # ── Mock interview ──
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
                _prompt_cfg = _CONFIG.get("prompts", {})
                mock_role = _prompt_cfg.get(
                    "mock_role",
                    "你是面试官，负责对候选人进行结构化面试")
                mock_rules = _prompt_cfg.get(
                    "mock_rules",
                    "- 前 3 题围绕 JD 核心要求\n- 中 2 题结合候选人经历与 JD 的匹配点\n- 最后 1 题行为面试（STAR）\n- 每个问题要自然、口语化，像真实面试官在提问\n- 输出格式：每行一个问题，不要编号前缀以外的内容")
                system_prompt = f"""{mock_role}。根据以下 JD 和候选人的经历，生成 6 个面试问题。

JD：
{jd}

候选人经历：
{get_profile()}

要求：
{mock_rules}"""
                self.mock_messages = [
                    {"role": "system", "content": system_prompt}]
                resp = call_deepseek(self.mock_messages, timeout=25)
                self.mock_questions = [
                    q.strip() for q in resp.strip().split("\n")
                    if q.strip() and len(q.strip()) > 5]
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
        self._set_interviewer(
            f"第 {self.mock_index + 1}/{len(self.mock_questions)} 题\n\n{q}")
        self._set_answer("")
        self._set_feedback("🎤 请口头回答这个问题...")
        self.mock_running = True
        self.status_var.set(
            f"🎤 等待回答 (第{self.mock_index+1}/{len(self.mock_questions)})")

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
            self.mock_messages.append(
                {"role": "user",
                 "content": f"面试官问：{q}\n\n我的回答：{text}"})
            prompt = self.mock_messages + [
                {"role": "system",
                 "content": f"请对候选人上一条回答给出评分(1-10)、优点、改进建议和参考回答示范，简洁口语化。JD要求：{self.jd_text.get(1.0, tk.END).strip()[:300]}"},
            ]
            feedback = call_deepseek(prompt, timeout=20)
            self.mock_messages.append(
                {"role": "assistant", "content": feedback})
            self._set_feedback(feedback)
            self.mock_history.append((q, text, feedback))
            self.status_var.set(f"✅ 第{self.mock_index+1}题完成")
            self.mock_index += 1
            self.root.after(2000, self._show_mock_question)

        threading.Thread(target=evaluate, daemon=True).start()

    def _end_mock(self):
        self._set_interviewer("🎉 面试结束！生成总结报告中...")
        self.status_var.set("📊 生成总结报告")

        def summarize():
            prompt = self.mock_messages + [
                {"role": "user",
                 "content": "请根据以上整个面试过程，给出总结评估报告：\n1. 总体评分\n2. 表现优势 (2-3条)\n3. 待改进 (2-3条)\n4. 针对该岗位的准备建议"},
            ]
            summary = call_deepseek(prompt, timeout=20)
            self.mock_summary = summary
            self._set_feedback(f"📋 总结报告\n\n{summary}")
            self.status_var.set("✅ 模拟面试完成")

        threading.Thread(target=summarize, daemon=True).start()

    def _save_report(self):
        import datetime
        content = (
            f"=== 面试记录 "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n\n")
        for i, (q, a, fb) in enumerate(self.mock_history, 1):
            content += f"第{i}题：{q}\n回答：{a}\n评价：{fb}\n\n"
        if hasattr(self, 'mock_summary'):
            content += f"总结报告：\n{self.mock_summary}\n"
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            f"面试报告_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self._set_feedback(f"💾 已保存: {os.path.basename(path)}")

    # ── Assist mode ──
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
        _prompt_cfg = _CONFIG.get("prompts", {})
        assist_role = _prompt_cfg.get(
            "assist_role",
            "你是面试辅助助手。根据用户的真实项目经历和面试情景，给出参考回答。")
        assist_rules = _prompt_cfg.get(
            "assist_rules",
            "基于用户经历不编造、贴合公司岗位、口语化、200字以内")
        messages = [{"role": "system",
                     "content": f"""{build_context()}{scene}

{assist_role}
要求：{assist_rules}"""}]
        messages.extend(self.assist_messages)
        messages.append(
            {"role": "user", "content": f"面试官问：{text}"})
        answer = call_deepseek(messages, timeout=15)
        self.assist_messages.append(
            {"role": "user", "content": f"面试官问：{text}"})
        self.assist_messages.append(
            {"role": "assistant", "content": answer})
        if len(self.assist_messages) > 20:
            self.assist_messages = self.assist_messages[-20:]
        self._set_assist_answer(f"问题：{text}\n\n参考回答：\n{answer}")
        self._set_status_async("🎤 监听中... 回答就绪")

    # ── Mode switch ──
    def _on_mode_switch(self):
        mode = self.current_mode.get()
        self.mock_frame.pack_forget()
        self.assist_frame.pack_forget()
        if mode == "mock":
            self.mock_frame.pack(fill=tk.BOTH, expand=True)
        else:
            self.assist_frame.pack(fill=tk.BOTH, expand=True)
        self._restart_audio()
        self.root.after(200, lambda: self._sync_rec_indicator())

    def _sync_rec_indicator(self):
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

    # ── UI helpers ──
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

    def _toggle_assist_ctx(self):
        if self.assist_ctx_content.winfo_ismapped():
            self.assist_ctx_content.pack_forget()
            self.assist_ctx_btn.config(text="展开 ▼")
        else:
            self.assist_ctx_content.pack(fill=tk.X, padx=6)
            self.assist_ctx_btn.config(text="收起 ▲")

    def _toggle_mute(self, event=None):
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
        name = self.audio_device_var.get()
        for idx, n in self.audio_devices:
            if n == name:
                self.audio_device_idx = idx
                break
        self.status_var.set(f"🔄 切换: {name}")
        self._restart_audio()

    def _toggle_ptt(self):
        if not self.recorder:
            return
        new_mode = not self.recorder.ptt_mode
        self.recorder.ptt_mode = new_mode
        if new_mode:
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
        if not self.recorder:
            return "break"
        if self.recorder.ptt_recording:
            self.recorder.stop_manual()
            self.rec_indicator.config(text="⏹ 已停止", fg="#e74c3c")
            self.status_var.set("处理中...")
        else:
            if not self.recorder.ptt_mode:
                self.recorder.ptt_mode = True
                self.rec_btn.config(text="🎤 手动(PTT)", bg="#e74c3c",
                                    fg="#fff")
            self.recorder.start_manual()
            self.rec_indicator.config(text="🔴 录音中", fg="#ff0000")
            self.status_var.set("🎙 录音中... 再按空格停止")
        return "break"

    def _toggle_settings(self):
        if self.settings_visible:
            self.settings_frame.pack_forget()
            self.settings_visible = False
            self.settings_btn.config(text="⚙ 阈值")
        else:
            self.settings_frame.pack(fill=tk.X)
            self.settings_visible = True
            self.settings_btn.config(text="⚙ 收起")

    def _on_threshold_change(self, val):
        if self.recorder:
            self.recorder.set_thresholds(
                energy_threshold=self.e_thresh_var.get(),
                user_speech_energy=self.user_thresh_var.get(),
            )

    def _apply_llm_config(self):
        from interview_assistant import llm
        llm.PROVIDER_MODEL = self.model_et.get().strip()
        llm.PROVIDER_KEY = self.key_et.get().strip()
        llm.PROVIDER_URL = self.url_et.get().strip().rstrip("/")
        self._set_status_async(f"✅ 已切换: {llm.PROVIDER_MODEL}")

    def _poll_energy(self):
        if not hasattr(self, 'wave_canvas'):
            return
        energy = self.recorder.current_energy if self.recorder else 0
        ptt_idle = (self.recorder and self.recorder.ptt_mode
                    and not self.recorder.ptt_recording
                    and not self.recorder.recording)
        val = 0 if ptt_idle else min(energy, 2000)
        self.wave_buffer.pop(0)
        self.wave_buffer.append(val)

        import math
        cw, ch = 180, 48
        mid = int(ch * 0.65)
        points = []
        for i, v in enumerate(self.wave_buffer):
            norm = min(1.0, math.log(1 + v / 10)
                       / math.log(1 + 2000 / 10))
            x = int(i * cw / len(self.wave_buffer))
            y = mid - int(norm * (mid - 2))
            points.extend([x, y])
        self.wave_canvas.coords(self.wave_line, *points)

        color = ("#00ff88" if (ptt_idle or energy < 25)
                 else ("#ffd700" if energy < 800 else "#ff4444"))
        self.wave_canvas.itemconfig(self.wave_line, fill=color)

        if self.recorder:
            if self.recorder.ptt_recording:
                self.rec_indicator.config(text="🔴 手动录音", fg="#ff0000")
            elif self.recorder.ptt_mode:
                self.rec_indicator.config(text="⏹ 空闲", fg="#ff9f43")
            elif self.recorder.recording:
                self.rec_indicator.config(text="● 录音中", fg="#ff6600")
            else:
                self.rec_indicator.config(text="● 待命", fg="#888")

        self.root.after(200, self._poll_energy)

    def quit(self):
        if self.capture:
            self.capture.stop()
        if self.recorder:
            self.recorder.reset()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
