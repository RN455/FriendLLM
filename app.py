"""Simple local desktop app for preparing WhatsApp chats and training an adapter."""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core import analyze_dataset, prepare_dataset, scan_speakers
from telemetry import duration


APP_DIR = Path(__file__).resolve().parent
RECOMMENDED_BASE = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
RECOMMENDED_ADAPTER = "Chun121/qwen3-4B-rpg-roleplay"
RECOMMENDED_PRESET = "Recommended — Qwen3 4B roleplay starter"
CUSTOM_PRESET = "Custom Hugging Face base and adapter"
STATUS_PREFIX = "FRIENDLLM_STATUS "
DATA_USAGE_AGREEMENT_VERSION = "1"
DATA_USAGE_CONDITIONS = (
    "Use only conversations you have permission to use. Everyone whose messages are included must have consented to this use.",
    "Chat exports are processed on this PC. Prepared datasets, checkpoints and trained adapters are saved locally. Public model files may be downloaded; the app does not upload your chats.",
    "Training can retain or reproduce personal information. Protect the saved files and share data or trained models only with permission from the people involved.",
    "Do not use the model to deceive others or impersonate someone without permission. Follow the terms of the selected base model and adapter.",
)


class DataUsageDialog(tk.Toplevel):
    """Require a fresh acknowledgement before each training or resume action."""

    def __init__(self, parent, config):
        super().__init__(parent)
        self.result = None
        self.title("Data use and consent")
        self.configure(background="#f4f6fa")
        self.transient(parent)
        width = min(680, self.winfo_screenwidth() - 80)
        height = min(760, self.winfo_screenheight() - 100)
        self.geometry(f"{width}x{height}+{(self.winfo_screenwidth() - width) // 2}+{(self.winfo_screenheight() - height) // 2}")
        self.minsize(min(520, width), min(480, height))
        self.resuming = bool(config.get("resume_from_checkpoint"))
        heading = ttk.Frame(self, style="App.TFrame", padding=(20, 18, 20, 12))
        heading.pack(fill="x")
        ttk.Label(heading, text="Data use and consent", style="Section.TLabel").pack(anchor="w")
        ttk.Label(heading, text="Review these conditions before starting or resuming a training run.",
                  style="Sub.TLabel", wraplength=width - 50).pack(anchor="w", pady=(5, 0))

        viewport = ttk.Frame(self, style="App.TFrame")
        viewport.pack(fill="both", expand=True, padx=20)
        self.canvas = tk.Canvas(viewport, background="#ffffff", highlightthickness=0)
        scroll = ttk.Scrollbar(viewport, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        content = ttk.Frame(self.canvas, style="Card.TFrame", padding=16)
        window_id = self.canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        wrapped = []

        def resize(event):
            self.canvas.itemconfigure(window_id, width=event.width)
            for label in wrapped:
                label.configure(wraplength=max(200, event.width - 38))

        self.canvas.bind("<Configure>", resize)
        ttk.Label(content, text="This training run", style="CardTitle.TLabel").pack(anchor="w")
        summary = [f"Speaker: {config.get('target_username', 'selected speaker')}",
                   f"Base model: {config.get('base_model_id', RECOMMENDED_BASE)}"]
        if self.resuming:
            summary.append(f"Resume from: {Path(config['resume_from_checkpoint']).name}")
        for text in summary:
            label = ttk.Label(content, text=text, style="Muted.TLabel", wraplength=width - 100)
            label.pack(anchor="w", fill="x", pady=(4, 0))
            wrapped.append(label)
        ttk.Separator(content).pack(fill="x", pady=14)
        for number, condition in enumerate(DATA_USAGE_CONDITIONS, 1):
            label = ttk.Label(content, text=f"{number}. {condition}", style="Body.TLabel", justify="left",
                              wraplength=width - 100)
            label.pack(anchor="w", fill="x", pady=(0, 12))
            wrapped.append(label)
        ttk.Separator(content).pack(fill="x", pady=(0, 10))
        self.confirmations = []
        for text in (
            "I have permission and consent from everyone whose messages are included in this training data.",
            "I understand that the dataset and trained model may contain personal information, and I will protect them and obtain permission before sharing.",
            "I agree to these data-use conditions for this training run.",
        ):
            variable = tk.BooleanVar(self, value=False)
            self.confirmations.append(variable)
            control = tk.Checkbutton(content, text=text, variable=variable, anchor="w", justify="left",
                                     wraplength=width - 100, font=("Segoe UI", 10), background="#ffffff",
                                     foreground="#34445f", activebackground="#ffffff", selectcolor="#ffffff",
                                     highlightthickness=0, borderwidth=0, padx=0, pady=7)
            control.pack(anchor="w", fill="x")
            wrapped.append(control)
            variable.trace_add("write", self._sync_accept)

        buttons = ttk.Frame(self, style="App.TFrame", padding=(20, 14))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Cancel", command=self._cancel).pack(side="left")
        self.accept_button = ttk.Button(buttons, text="Agree & resume training" if self.resuming else "Agree & start training",
                                       command=self._accept, style="Accent.TButton", state="disabled")
        self.accept_button.pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _event: self._cancel())
        self.bind("<Return>", lambda _event: self._accept())
        self.bind("<MouseWheel>", self._scroll)
        self.bind("<Button-4>", self._scroll)
        self.bind("<Button-5>", self._scroll)
        self.update_idletasks()
        self.grab_set()
        self.focus_set()

    def _scroll(self, event):
        if self.canvas.yview() != (0.0, 1.0):
            number = getattr(event, "num", None)
            delta = getattr(event, "delta", 0)
            units = -1 if number == 4 or delta > 0 else 1
            self.canvas.yview_scroll(units, "units")
        return "break"

    def _sync_accept(self, *_args):
        self.accept_button.configure(state="normal" if all(value.get() for value in self.confirmations) else "disabled")

    def _accept(self):
        if all(value.get() for value in self.confirmations):
            self.result = {
                "version": DATA_USAGE_AGREEMENT_VERSION,
                "accepted_utc": datetime.now(timezone.utc).isoformat(),
                "permission_and_consent_confirmed": True,
                "privacy_and_sharing_confirmed": True,
                "conditions_accepted": True,
                "conditions": list(DATA_USAGE_CONDITIONS),
            }
            self.destroy()
        return "break"

    def _cancel(self):
        self.result = None
        self.destroy()
        return "break"


def run_training_worker(config_path: str):
    # PyInstaller's --windowed mode clears stdout; reconnect it to the parent pipe.
    if sys.stdout is None:
        sys.stdout = os.fdopen(1, "w", encoding="utf-8", buffering=1, closefd=False)
    if sys.stderr is None:
        sys.stderr = os.fdopen(2, "w", encoding="utf-8", buffering=1, closefd=False)
    from train import main as train_main

    try:
        train_main(config_path)
    except Exception:
        # A windowed PyInstaller bootloader opens a blocking dialog for uncaught
        # exceptions. Return the traceback through the parent's activity pipe.
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


class FriendLLMApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FriendLLM Desktop")
        width = min(1120, self.root.winfo_screenwidth() - 80)
        height = min(960, self.root.winfo_screenheight() - 100)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(800, width), min(640, height))
        self.root.configure(bg="#f4f6fa")
        self.events: queue.Queue = queue.Queue()
        self.input_paths: list[Path] = []
        self.dataset_dir: Path | None = None
        self.dataset_generation = 0
        self.busy = False
        self.process: subprocess.Popen | None = None
        self.stop_file: Path | None = None
        self.training_paused = False
        self.default_output = Path.home() / "FriendLLM"
        self.speaker_labels: dict[str, str] = {}
        self.scan_generation = 0
        self.estimate_generation = 0
        self.estimate_after: str | None = None
        self.scroll_canvases = {}
        self.telemetry_history = deque(maxlen=300)
        self.performance_finished = False

        self._make_styles()
        self._build_ui()
        self.output_var.trace_add("write", self._output_changed)
        self._refresh_resume_button()
        for variable in (self.format_var, self.date_var):
            variable.trace_add("write", self._parser_settings_changed)
        for variable in (
            self.alias_var, self.gap_var, self.segment_var, self.context_var, self.replies_var, self.seed_var,
        ):
            variable.trace_add("write", self._settings_changed)
        self.root.after(100, self._drain_events)

    def _make_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#f4f6fa")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#f4f6fa", foreground="#14213d", font=("Segoe UI", 23, "bold"))
        style.configure("Section.TLabel", background="#f4f6fa", foreground="#14213d", font=("Segoe UI", 17, "bold"))
        style.configure("Sub.TLabel", background="#f4f6fa", foreground="#52617a", font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#14213d", font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#34445f", font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#66748a", font=("Segoe UI", 9))
        style.configure("TButton", padding=(10, 7), font=("Segoe UI", 9), background="#edf2f8", foreground="#243149")
        style.map("TButton", background=[("active", "#e0e8f2")])
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8),
                        background="#2563eb", foreground="#ffffff")
        style.map("Accent.TButton", background=[("disabled", "#dce5f4"), ("active", "#1d4ed8")],
                  foreground=[("disabled", "#71839d"), ("!disabled", "#ffffff")])
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=4)
        style.configure("TProgressbar", background="#2563eb", troughcolor="#e9eef6", borderwidth=0)

    def _build_ui(self):
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(20, 15, 20, 12))
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x", pady=(0, 10))
        heading = ttk.Frame(header, style="App.TFrame")
        heading.pack(side="left", fill="x", expand=True)
        self.page_title = tk.StringVar(value="FriendLLM Desktop")
        self.page_subtitle = tk.StringVar(value="Choose your chats, pick a speaker, and train on this PC.")
        ttk.Label(heading, textvariable=self.page_title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(heading, textvariable=self.page_subtitle, style="Sub.TLabel").pack(anchor="w", pady=(3, 0))
        self.settings_button = ttk.Button(header, text="Settings", command=self._toggle_settings)
        self.settings_button.pack(side="right", anchor="n", pady=(5, 0), padx=(12, 0))

        self.page_container = ttk.Frame(outer, style="App.TFrame")
        self.page_container.pack(fill="both", expand=True)
        self.page_container.columnconfigure(0, weight=1)
        self.page_container.rowconfigure(0, weight=1)
        self.pages = {}
        for name in ("main", "settings"):
            page = ttk.Frame(self.page_container, style="App.TFrame")
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[name] = page
        self.main_content = self._scrollable_page(self.pages["main"])
        self.settings_content = self._scrollable_page(self.pages["settings"])
        self.status_var = tk.StringVar(value="Ready — add chat files to scan")
        self.eta_var = tk.StringVar(value="Remaining: starts estimating after training begins")
        self.progress_var = tk.DoubleVar(value=0)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self._build_main_inputs()
        self._build_settings_page()
        self._build_training_controls()
        self.monitor_row = ttk.Frame(self.main_content, style="App.TFrame")
        self.monitor_row.pack(fill="x", pady=(0, 9))
        self.monitor_row.columnconfigure(0, weight=1, uniform="monitor")
        self.monitor_row.columnconfigure(1, weight=1, uniform="monitor")
        self.performance_panel = ttk.Frame(self.monitor_row, style="Card.TFrame", padding=12)
        self.performance_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.activity_panel = ttk.Frame(self.monitor_row, style="Card.TFrame", padding=12)
        self.activity_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self._build_performance_panel()
        self._build_activity_panel()
        self._build_file_buttons()
        footer = ttk.Label(outer, textvariable=self.status_var, style="Sub.TLabel")
        footer.pack(fill="x", pady=(10, 0))
        outer.bind("<Configure>", lambda event: footer.configure(wraplength=max(300, event.width - 50)), add="+")
        self._show_page("main")

    def _show_page(self, name):
        self.current_page = name
        self.pages[name].tkraise()
        settings = name == "settings"
        self.page_title.set("Settings" if settings else "FriendLLM Desktop")
        self.page_subtitle.set("Model, chat parsing, context and training options." if settings
                               else "Choose your chats, pick a speaker, and train on this PC.")
        self.settings_button.configure(text="Back to training" if settings else "Settings")

    def _toggle_settings(self):
        self._show_page("main" if self.current_page == "settings" else "settings")

    def _scrollable_page(self, page):
        viewport = ttk.Frame(page, style="App.TFrame")
        viewport.pack(fill="both", expand=True)
        canvas = tk.Canvas(viewport, background="#f4f6fa", borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        content = ttk.Frame(canvas, style="App.TFrame", padding=(2, 8))
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))
        self.scroll_canvases[page] = canvas
        return content

    def _on_mousewheel(self, event):
        active_page = self.pages[self.current_page]
        canvas = self.scroll_canvases.get(active_page)
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not active_page:
            if widget.winfo_class() in {"Listbox", "Text", "TCombobox", "TSpinbox", "Scrollbar"}:
                return
            widget = getattr(widget, "master", None)
        if widget is None:
            return
        if getattr(event, "num", None) == 4:
            units = -1
        elif getattr(event, "num", None) == 5:
            units = 1
        else:
            delta = int(getattr(event, "delta", 0))
            units = int(-delta / 120)
            if units == 0 and delta:
                units = -1 if delta > 0 else 1
        if canvas is not None and units and canvas.yview() != (0.0, 1.0):
            canvas.yview_scroll(units, "units")
            return "break"

    def _card(self, parent, title: str, description: str):
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.pack(fill="x", pady=(0, 9))
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        if description:
            label = ttk.Label(card, text=description, style="Body.TLabel", wraplength=870)
            label.pack(anchor="w", fill="x", pady=(2, 8))
            card.bind("<Configure>", lambda event: label.configure(wraplength=max(200, event.width - 28)), add="+")
        return card

    def _build_main_inputs(self):
        files = self._card(
            self.main_content, "1. Upload chat exports",
            "Choose WhatsApp TXT files or ZIPs. Speakers are scanned automatically; your chats stay on this PC.",
        )
        row = ttk.Frame(files, style="Card.TFrame")
        row.pack(fill="x")
        controls = ttk.Frame(row, style="Card.TFrame")
        controls.pack(fill="x", pady=(0, 7))
        ttk.Button(controls, text="Add files…", command=self._add_files).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Add folder…", command=self._add_folder).pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Remove selected", command=self._remove_selected).pack(side="left")
        list_frame = ttk.Frame(row, style="Card.TFrame")
        list_frame.pack(fill="x")
        self.input_list = tk.Listbox(
            list_frame, height=2, selectmode="extended", font=("Segoe UI", 9), bg="#f8f9fc", fg="#243149",
            relief="flat", highlightthickness=0,
        )
        self.input_list.pack(side="left", fill="both", expand=True)
        file_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.input_list.yview)
        self.input_list.configure(yscrollcommand=file_scroll.set)
        file_scroll.pack(side="right", fill="y")

        target = self._card(
            self.main_content, "2. Select a speaker", "",
        )
        grid = ttk.Frame(target, style="Card.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        ttk.Label(grid, text="Speaker", style="Body.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=5)
        self.target_display_var = tk.StringVar()
        self.target_combo = ttk.Combobox(grid, textvariable=self.target_display_var, state="readonly")
        self.target_combo.grid(row=0, column=1, sticky="ew", pady=4)
        self.target_combo.bind("<<ComboboxSelected>>", lambda _event: self._selected_speaker_changed())
        self.scan_label = ttk.Label(target, text="Add files to find speakers.", style="Muted.TLabel")
        self.scan_label.pack(anchor="w", pady=(7, 0))
        summary = ttk.Frame(target, style="Card.TFrame")
        summary.pack(fill="x", pady=(5, 0))
        self.example_label = ttk.Label(summary, text="Possible examples: —", style="CardTitle.TLabel")
        self.example_label.pack(side="left")
        self.split_label = ttk.Label(summary, text="Estimated split: train —  |  validation —  |  test —", style="Muted.TLabel")
        self.split_label.pack(side="left", padx=(18, 0))
        self.example_note = ttk.Label(target, text="Choose a speaker to calculate examples from the current context settings.", style="Muted.TLabel", wraplength=870)
        self.example_note.pack(anchor="w", pady=(3, 0))
        target.bind("<Configure>", lambda event: self.example_note.configure(wraplength=max(200, event.width - 28)), add="+")

        output = self._card(self.main_content, "3. Select output", "")
        grid = ttk.Frame(output, style="Card.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        ttk.Label(grid, text="Output folder", style="Body.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        self.output_var = tk.StringVar(value=str(self.default_output))
        ttk.Entry(grid, textvariable=self.output_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(grid, text="Browse…", command=self._browse_output).grid(row=0, column=2, padx=(7, 0), pady=4)

    def _build_settings_page(self):
        parsing = self._card(self.settings_content, "Chat parsing",
                             "Change these if an export uses a different format or date order. Aliases are comma-separated names for the same speaker.")
        parse_row = ttk.Frame(parsing, style="Card.TFrame")
        parse_row.pack(fill="x")
        ttk.Label(parse_row, text="Format", style="Body.TLabel").pack(side="left")
        self.format_var = tk.StringVar(value="auto")
        ttk.Combobox(parse_row, textvariable=self.format_var, values=("auto", "whatsapp", "mini"), state="readonly", width=13).pack(side="left", padx=(6, 18))
        ttk.Label(parse_row, text="Date order", style="Body.TLabel").pack(side="left")
        self.date_var = tk.StringVar(value="MDY")
        ttk.Combobox(parse_row, textvariable=self.date_var, values=("MDY", "DMY"), state="readonly", width=8).pack(side="left", padx=(6, 0))
        alias_row = ttk.Frame(parsing, style="Card.TFrame")
        alias_row.pack(fill="x", pady=(8, 0))
        ttk.Label(alias_row, text="Speaker aliases (optional)", style="Body.TLabel").pack(side="left", padx=(0, 12))
        self.alias_var = tk.StringVar()
        ttk.Entry(alias_row, textvariable=self.alias_var).pack(side="left", fill="x", expand=True)
        starter = self._card(
            self.settings_content, "Starter model",
            "Qwen3 4B with the roleplay starter is recommended. A custom base must have a matching LoRA adapter.",
        )
        self.preset_var = tk.StringVar(value=RECOMMENDED_PRESET)
        ttk.Combobox(
            starter, textvariable=self.preset_var, values=(RECOMMENDED_PRESET, CUSTOM_PRESET), state="readonly", width=48
        ).pack(anchor="w", fill="x")
        model_grid = ttk.Frame(starter, style="Card.TFrame")
        model_grid.pack(fill="x", pady=(7, 0))
        model_grid.columnconfigure(1, weight=1)
        ttk.Label(model_grid, text="Base model ID", style="Body.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=3)
        self.base_model_var = tk.StringVar(value=RECOMMENDED_BASE)
        self.base_model_entry = ttk.Entry(model_grid, textvariable=self.base_model_var, state="disabled")
        self.base_model_entry.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(model_grid, text="LoRA adapter repo", style="Body.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=3)
        self.adapter_repo_var = tk.StringVar(value=RECOMMENDED_ADAPTER)
        self.adapter_repo_entry = ttk.Entry(model_grid, textvariable=self.adapter_repo_var, state="disabled")
        self.adapter_repo_entry.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(model_grid, text="Saved model label", style="Body.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=3)
        self.model_var = tk.StringVar(value="FriendLLM")
        ttk.Entry(model_grid, textvariable=self.model_var).grid(row=2, column=1, sticky="ew", pady=3)
        self.preset_var.trace_add("write", self._preset_changed)

        context = self._card(
            self.settings_content, "Dataset context",
            "These values control how much of each conversation can become an example. Changing them recalculates the estimate.",
        )
        self.gap_var = tk.StringVar(value="60")
        self.segment_var = tk.StringVar(value="100")
        self.context_var = tk.StringVar(value="5")
        self.replies_var = tk.StringVar(value="3")
        self.seed_var = tk.StringVar(value="3407")
        self._settings_grid(context, [
            ("Conversation gap (minutes)", self.gap_var, 1, 1440),
            ("Maximum messages per segment", self.segment_var, 10, 1000),
            ("Context messages per example", self.context_var, 1, 64),
            ("Target replies per example", self.replies_var, 1, 32),
            ("Shuffle seed", self.seed_var, 0, 2147483647),
        ])

        training = self._card(
            self.settings_content, "Training settings",
            "Sequence length is the token context window used by the model. Other settings tune the local LoRA training run.",
        )
        self.epochs_var = tk.StringVar(value="2")
        self.length_var = tk.StringVar(value="1024")
        self.batch_var = tk.StringVar(value="1")
        self.grad_accum_var = tk.StringVar(value="16")
        self.lr_var = tk.StringVar(value="0.00005")
        self.device_var = tk.StringVar(value="Auto — use supported GPU, otherwise CPU")
        ttk.Label(training, text="Training device", style="Body.TLabel").pack(anchor="w")
        ttk.Combobox(
            training, textvariable=self.device_var, state="readonly",
            values=("Auto — use supported GPU, otherwise CPU", "CPU only"),
        ).pack(fill="x", pady=(3, 7))
        hardware_note = ttk.Label(
            training, text="GPU use depends on the installed PyTorch backend and available memory. "
            "CPU training uses more RAM and is slower.", style="Muted.TLabel", wraplength=870,
        )
        hardware_note.pack(anchor="w", fill="x", pady=(0, 7))
        training.bind("<Configure>", lambda event: hardware_note.configure(wraplength=max(200, event.width - 28)), add="+")
        self._settings_grid(training, [
            ("Epochs", self.epochs_var, 1, 10),
            ("Sequence length (tokens)", self.length_var, 256, 4096),
            ("Batch size", self.batch_var, 1, 4),
            ("Gradient accumulation", self.grad_accum_var, 1, 128),
            ("Learning rate", self.lr_var, 0.0000001, 0.01),
        ], learning_rate=True)

        settings_note = ttk.Label(self.settings_content, text="Changes apply to your next preparation or training run. Resuming a checkpoint uses its saved settings.",
                                 style="Sub.TLabel", wraplength=870)
        settings_note.pack(anchor="w", fill="x", pady=8)
        self.settings_content.bind("<Configure>", lambda event: settings_note.configure(wraplength=max(200, event.width - 28)), add="+")

    def _build_training_controls(self):
        actions = ttk.Frame(self.main_content, style="App.TFrame")
        actions.pack(fill="x", pady=(0, 6))
        self.prepare_button = ttk.Button(actions, text="Prepare dataset", command=self._prepare)
        self.train_button = ttk.Button(actions, text="Start training", command=self._train, style="Accent.TButton", state="disabled")
        self.resume_button = ttk.Button(actions, text="Resume latest checkpoint", command=self._resume_latest, state="disabled")
        self.stop_button = ttk.Button(actions, text="Stop & save checkpoint", command=self._stop_training, state="disabled")
        for column, button in enumerate((self.prepare_button, self.train_button, self.resume_button, self.stop_button)):
            actions.columnconfigure(column, weight=1, uniform="actions")
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 6, 0))
        self.dataset_label = ttk.Label(self.main_content, text="Dataset: not prepared", style="Sub.TLabel", wraplength=900)
        self.dataset_label.pack(anchor="w", fill="x", pady=(0, 10))
        self.main_content.bind("<Configure>", lambda event: self.dataset_label.configure(wraplength=max(200, event.width - 24)), add="+")

    def _build_file_buttons(self):
        files = self._card(self.main_content, "Open saved files", "")
        row = ttk.Frame(files, style="Card.TFrame")
        row.pack(fill="x", pady=(8, 0))
        for column, (text, command) in enumerate((
            ("Output folder", self._open_output), ("Training saves", self._open_latest_run),
            ("Model files", self._open_latest_adapter), ("Checkpoints", self._open_latest_checkpoint),
            ("Base model files", self._open_base_model_cache),
        )):
            row.columnconfigure(column, weight=1, uniform="files")
            ttk.Button(row, text=text, command=command).grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 6, 0))

    def _build_activity_panel(self):
        activity = self.activity_panel
        ttk.Label(activity, text="Output activity", style="CardTitle.TLabel").pack(anchor="w")
        status = ttk.Label(activity, textvariable=self.status_var, style="Muted.TLabel", wraplength=440)
        status.pack(anchor="w", fill="x", pady=(5, 8))
        self.progress = ttk.Progressbar(activity, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x", pady=(0, 7))
        eta = ttk.Label(activity, textvariable=self.eta_var, style="Muted.TLabel", wraplength=440)
        eta.pack(anchor="w", fill="x", pady=(0, 8))
        log_frame = ttk.Frame(activity, style="Card.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=13, width=1, wrap="word", font=("Consolas", 9), bg="#f8f9fc", fg="#243149",
                           relief="flat", padx=8, pady=6, state="disabled", highlightthickness=0)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        activity.bind("<Configure>", lambda event: [label.configure(wraplength=max(180, event.width - 28)) for label in (status, eta)], add="+")

    def _build_performance_panel(self):
        card = self.performance_panel
        ttk.Label(card, text="Performance", style="CardTitle.TLabel").pack(anchor="w")
        note = ttk.Label(card, text="Live hardware usage · session average and peak", style="Muted.TLabel", wraplength=440)
        note.pack(anchor="w", fill="x", pady=(4, 8))
        self.performance_vars = {}
        values = []
        for key, title in (
            ("device", "Device"), ("elapsed", "Elapsed"),
            ("remaining", "Time remaining"), ("step", "Optimizer step"),
            ("cpu_percent", "CPU load"), ("ram_percent", "System RAM"),
            ("process_ram_gib", "Training RAM"), ("gpu_percent", "GPU load"),
            ("gpu_ram_gib", "GPU memory"),
        ):
            row = ttk.Frame(card, style="Card.TFrame")
            row.pack(fill="x", pady=2)
            row.columnconfigure(1, weight=1)
            ttk.Label(row, text=title, style="Body.TLabel", width=17).grid(row=0, column=0, sticky="nw")
            variable = tk.StringVar(value="—")
            self.performance_vars[key] = variable
            label = ttk.Label(row, textvariable=variable, style="Muted.TLabel", wraplength=280)
            label.grid(row=0, column=1, sticky="ew")
            values.append(label)
        ttk.Label(card, text="Usage over time", style="Body.TLabel").pack(anchor="w", pady=(10, 4))
        self.performance_chart = tk.Canvas(card, height=140, background="#f8f9fc", highlightthickness=0)
        self.performance_chart.pack(fill="x", pady=(0, 4))
        self.performance_chart.bind("<Configure>", lambda _event: self._draw_performance())
        legend = ttk.Label(card, text="CPU: blue · RAM: green · GPU: orange\nETA uses measured speed after the first few steps. Loading and final packaging can add time.",
                           style="Muted.TLabel", wraplength=440)
        legend.pack(anchor="w", fill="x")

        def resize(event):
            note.configure(wraplength=max(180, event.width - 28))
            legend.configure(wraplength=max(180, event.width - 28))
            for label in values:
                label.configure(wraplength=max(140, event.width - 155))

        card.bind("<Configure>", resize, add="+")

    def _draw_performance(self):
        canvas = self.performance_chart
        canvas.delete("all")
        width, height = max(100, canvas.winfo_width()), max(80, canvas.winfo_height())
        bottom, plot_height = height - 25, height - 45
        for percent in (0, 50, 100):
            y = bottom - plot_height * percent / 100
            canvas.create_line(36, y, width - 8, y, fill="#dde3ed")
            canvas.create_text(30, y, text=f"{percent}%", anchor="e", fill="#67758a")
        samples = list(self.telemetry_history)
        if len(samples) < 2:
            return
        start, end = samples[0]["elapsed_seconds"], samples[-1]["elapsed_seconds"]
        span = max(1, end - start)
        for key, color in (("cpu_percent", "#377bd3"), ("ram_percent", "#27966d"), ("gpu_percent", "#db8b25")):
            points = []
            for sample in samples:
                if key not in sample:
                    if len(points) >= 4:
                        canvas.create_line(*points, fill=color, width=2)
                    points = []
                    continue
                points.extend((36 + (sample["elapsed_seconds"] - start) / span * (width - 44),
                               bottom - max(0, min(100, sample[key])) * plot_height / 100))
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2)
        canvas.create_text(36, height - 10, text=duration(start), anchor="w", fill="#67758a")
        canvas.create_text(width - 8, height - 10, text=duration(end), anchor="e", fill="#67758a")

    def _update_performance(self, payload):
        self.telemetry_history.append(payload)
        remaining = duration(payload.get("eta_seconds"))
        if not self.performance_finished:
            self.eta_var.set(f"Remaining: ~{remaining}" if payload.get("eta_seconds") is not None
                             else "Remaining: calculating from measured training speed…")
        self.performance_vars["elapsed"].set(duration(payload.get("elapsed_seconds")))
        self.performance_vars["remaining"].set(remaining)
        self.performance_vars["device"].set(payload.get("device", "Detecting…"))
        seconds = payload.get("average_step_seconds")
        self.performance_vars["step"].set(f"{seconds:.2f} s · {payload.get('measured_steps', 0):,} measured steps"
                                          if seconds is not None else "Measuring first steps…")
        for key in ("cpu_percent", "ram_percent", "process_ram_gib", "gpu_percent", "gpu_ram_gib"):
            value = payload.get(key)
            if value is None:
                self.performance_vars[key].set("Not reported by this backend")
                continue
            stats = payload.get("statistics", {}).get(key, {})
            unit = "%" if key.endswith("percent") else " GiB"
            self.performance_vars[key].set(
                f"{value:.1f}{unit} · avg {stats.get('mean', value):.1f}{unit} · peak {stats.get('peak', value):.1f}{unit}")
        self._draw_performance()

    def _settings_grid(self, parent, rows, learning_rate=False):
        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)
        for index, (label, variable, low, high) in enumerate(rows):
            column = (index % 2) * 2
            row = index // 2
            ttk.Label(grid, text=label, style="Body.TLabel").grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
            if learning_rate and label == "Learning rate":
                control = ttk.Entry(grid, textvariable=variable, width=14)
            else:
                control = ttk.Spinbox(grid, from_=low, to=high, textvariable=variable, width=12)
            control.grid(row=row, column=column + 1, sticky="ew", padx=(0, 16), pady=3)

    def _preset_changed(self, *_args):
        state = "normal" if self.preset_var.get() == CUSTOM_PRESET else "disabled"
        self.base_model_entry.configure(state=state)
        self.adapter_repo_entry.configure(state=state)

    def _add_files(self):
        selected = filedialog.askopenfilenames(
            title="Choose WhatsApp text exports",
            filetypes=(("Chat exports", "*.txt *.zip"), ("Text files", "*.txt"), ("ZIP archives", "*.zip"), ("All files", "*.*")),
        )
        self._add_input_paths(Path(name) for name in selected)

    def _add_folder(self):
        selected = filedialog.askdirectory(title="Choose a folder containing .txt exports")
        if selected:
            self._add_input_paths([Path(selected)])

    def _add_input_paths(self, paths):
        known = {str(path.resolve()).casefold() for path in self.input_paths}
        changed = False
        for path in paths:
            if not path.exists():
                continue
            normalized = str(path.resolve()).casefold()
            if normalized in known or (path.is_file() and path.suffix.lower() not in {".txt", ".zip"}):
                continue
            self.input_paths.append(path)
            self.input_list.insert("end", str(path))
            known.add(normalized)
            changed = True
        if changed:
            self._invalidate_dataset()
            self._scan_inputs()

    def _remove_selected(self):
        selected = list(self.input_list.curselection())
        for index in reversed(selected):
            self.input_list.delete(index)
            del self.input_paths[index]
        if selected:
            self._invalidate_dataset()
            if self.input_paths:
                self._scan_inputs()
            else:
                self.scan_generation += 1
                self.speaker_labels.clear()
                self.target_combo.configure(values=())
                self.target_display_var.set("")
                self.scan_label.configure(text="Add files to find speakers.")
                self._clear_estimate()

    def _invalidate_dataset(self):
        self.dataset_generation += 1
        self.dataset_dir = None
        self.dataset_label.configure(text="Dataset: not prepared")
        self.train_button.configure(state="disabled")

    def _browse_output(self):
        selected = filedialog.askdirectory(title="Choose FriendLLM output folder", initialdir=self.output_var.get())
        if selected:
            self.output_var.set(selected)

    def _raw_target(self) -> str:
        return self.speaker_labels.get(self.target_display_var.get(), "")

    def _scan_inputs(self):
        if not self.input_paths:
            return
        self.scan_generation += 1
        generation = self.scan_generation
        self._clear_estimate()
        paths = tuple(self.input_paths)
        self.scan_label.configure(text="Scanning selected exports for speakers…")
        self.status_var.set("Scanning chat files for speakers…")
        self._append_log(f"Scanning {len(paths)} selected input(s)…")
        fmt, date_order = self.format_var.get(), self.date_var.get()

        def work():
            try:
                result = scan_speakers(paths, fmt, date_order)
                self.events.put(("scan_complete", generation, result))
            except Exception as error:
                self.events.put(("scan_error", generation, str(error)))

        threading.Thread(target=work, daemon=True).start()

    def _selected_speaker_changed(self):
        self._invalidate_dataset()
        self._schedule_estimate()

    def _settings_changed(self, *_args):
        if hasattr(self, "scan_label") and self.input_paths:
            self._invalidate_dataset()
            self._schedule_estimate()

    def _parser_settings_changed(self, *_args):
        if hasattr(self, "scan_label") and self.input_paths:
            self._invalidate_dataset()
            self._scan_inputs()

    def _schedule_estimate(self):
        self._clear_estimate()
        if not self._raw_target() or not self.input_paths:
            return
        self.estimate_after = self.root.after(350, self._estimate)

    def _clear_estimate(self):
        # Ignore results already in flight as soon as inputs/settings change.
        self.estimate_generation += 1
        if self.estimate_after:
            try:
                self.root.after_cancel(self.estimate_after)
            except tk.TclError:
                pass
            self.estimate_after = None
        self.example_label.configure(text="Possible examples: —")
        self.split_label.configure(text="Estimated split: train —  |  validation —  |  test —")
        self.example_note.configure(text="Choose a speaker to calculate examples from the current context settings.")

    def _get_preparation_settings(self):
        settings = {
            "seed": int(self.seed_var.get()),
            "gap_minutes": int(self.gap_var.get()),
            "max_segment_messages": int(self.segment_var.get()),
            "max_context_messages": int(self.context_var.get()),
            "max_target_replies": int(self.replies_var.get()),
        }
        if not 0 <= settings["seed"] <= 2147483647:
            raise ValueError("Shuffle seed must be between 0 and 2147483647.")
        if not 1 <= settings["gap_minutes"] <= 1440:
            raise ValueError("Conversation gap must be between 1 and 1440 minutes.")
        if not 10 <= settings["max_segment_messages"] <= 1000:
            raise ValueError("Maximum segment length must be between 10 and 1000 messages.")
        if not 1 <= settings["max_context_messages"] <= 64:
            raise ValueError("Context length must be between 1 and 64 messages.")
        if not 1 <= settings["max_target_replies"] <= 32:
            raise ValueError("Target replies must be between 1 and 32 messages.")
        return settings

    def _estimate(self):
        self.estimate_after = None
        target = self._raw_target()
        if not target:
            return
        try:
            settings = self._get_preparation_settings()
        except ValueError:
            self.example_note.configure(text="Enter whole numbers for the dataset context settings.")
            return
        self.estimate_generation += 1
        generation = self.estimate_generation
        paths = tuple(self.input_paths)
        aliases = [value.strip() for value in self.alias_var.get().split(",") if value.strip()]
        fmt, date_order = self.format_var.get(), self.date_var.get()
        self.example_label.configure(text="Possible examples: scanning…")
        self.example_note.configure(text="Counting usable message examples with the selected settings…")

        def work():
            try:
                result = analyze_dataset(paths, target, aliases, fmt, date_order, **settings)
                self.events.put(("estimate_complete", generation, result))
            except Exception as error:
                self.events.put(("estimate_error", generation, str(error)))

        threading.Thread(target=work, daemon=True).start()

    def _append_log(self, message: str):
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        line_count = int(self.log.index("end-1c").split(".")[0])
        if line_count > 800:
            self.log.delete("1.0", f"{line_count - 700}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_busy(self, value: bool):
        self.busy = value
        self.prepare_button.configure(state="disabled" if value else "normal")
        self.train_button.configure(state="disabled" if value or not self.dataset_dir else "normal")
        self.stop_button.configure(
            state="normal" if value and self.process is not None and self.process.poll() is None else "disabled"
        )
        self._refresh_resume_button()

    def _prepare(self):
        if self.busy:
            return
        target = self._raw_target()
        if not self.input_paths:
            messagebox.showinfo("Choose chat exports", "Add one or more .txt or .zip chat exports first.")
            return
        if not target:
            messagebox.showinfo("Choose a speaker", "Select a speaker from the scanned speaker list first.")
            return
        try:
            settings = self._get_preparation_settings()
            output_parent = Path(self.output_var.get()).expanduser() / "datasets"
        except (OSError, ValueError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        args = (
            tuple(self.input_paths), target,
            [value.strip() for value in self.alias_var.get().split(",") if value.strip()],
            self.format_var.get(), self.date_var.get(), output_parent,
        )
        generation = self.dataset_generation
        self._set_busy(True)
        self.status_var.set("Preparing the local dataset…")
        self.progress_var.set(0)
        self._append_log("Preparing examples and train/validation/test splits…")

        def work():
            try:
                result = prepare_dataset(*args, **settings)
                self.events.put(("prepared", generation, result))
            except Exception as error:
                self.events.put(("error", "Dataset preparation failed", str(error)))

        threading.Thread(target=work, daemon=True).start()

    def _train(self):
        if self.busy or self.dataset_dir is None:
            return
        target = self._raw_target()
        if not target:
            messagebox.showinfo("Choose a speaker", "Select the speaker used when preparing the dataset.")
            return
        if self.dataset_dir.parent.name != "datasets":
            messagebox.showerror("Dataset location", "The prepared dataset path is invalid. Prepare the dataset again.")
            return
        try:
            epochs = int(self.epochs_var.get())
            max_seq_length = int(self.length_var.get())
            batch_size = int(self.batch_var.get())
            grad_accum = int(self.grad_accum_var.get())
            learning_rate = float(self.lr_var.get())
            seed = int(self.seed_var.get())
            if not 1 <= epochs <= 10 or not 256 <= max_seq_length <= 4096:
                raise ValueError
            if not 1 <= batch_size <= 4 or not 1 <= grad_accum <= 128 or not 0 < learning_rate <= 0.01:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid training settings", "Check the Settings page. Values are outside the supported ranges.")
            return
        custom = self.preset_var.get() == CUSTOM_PRESET
        base_model = self.base_model_var.get().strip() if custom else RECOMMENDED_BASE
        adapter_repo = self.adapter_repo_var.get().strip() if custom else RECOMMENDED_ADAPTER
        if not base_model or not adapter_repo:
            messagebox.showerror("Model required", "Enter a Hugging Face base model ID and matching LoRA adapter repository.")
            return
        if not getattr(sys, "frozen", False):
            required_modules = ("torch", "transformers", "peft", "accelerate", "huggingface_hub")
            missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
            if missing:
                messagebox.showerror(
                    "Training dependencies are missing",
                    "Run Setup.bat in the app folder, then reopen FriendLLM.\n\nMissing: " + ", ".join(missing),
                )
                return
        try:
            output_parent = Path(self.output_var.get()).expanduser().resolve()
        except (OSError, ValueError) as error:
            messagebox.showerror("Invalid output folder", str(error))
            return
        config = {
            "dataset_dir": str(self.dataset_dir),
            "output_parent": str(output_parent),
            "target_username": target,
            "model_name": self.model_var.get().strip() or "FriendLLM",
            "base_model_id": base_model,
            "adapter_repo_id": adapter_repo,
            "epochs": epochs,
            "max_seq_length": max_seq_length,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "learning_rate": learning_rate,
            "seed": seed,
            "device_preference": "cpu" if self.device_var.get() == "CPU only" else "auto",
        }
        config_path = output_parent / f"training_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._launch_training(config, config_path)

    def _confirm_data_usage(self, config):
        dialog = DataUsageDialog(self.root, config)
        self.root.wait_window(dialog)
        return dialog.result

    def _launch_training(self, config: dict, config_path: Path):
        if self.busy:
            return
        generation = self.dataset_generation
        agreement = self._confirm_data_usage(config)
        if agreement is None:
            self.status_var.set("Training not started — agreement cancelled")
            return
        if not config.get("resume_from_checkpoint") and (
            generation != self.dataset_generation or self.dataset_dir is None
            or str(self.dataset_dir) != config.get("dataset_dir")
        ):
            messagebox.showinfo("Dataset changed", "The dataset changed while the agreement was open. Prepare it again before training.")
            return
        config = dict(config)
        config["data_usage_agreement"] = agreement
        self._show_page("main")
        action = "Resuming training" if config.get("resume_from_checkpoint") else "Starting training"
        self._append_log(f"{action} for {config.get('target_username', 'selected speaker')} — data-use agreement confirmed.")
        self.performance_finished = False
        self.telemetry_history.clear()
        self._draw_performance()
        self.eta_var.set("Remaining: waiting for measured training speed…")
        for variable in self.performance_vars.values():
            variable.set("Waiting for training…")
        self.training_paused = False
        self._set_busy(True)
        self.status_var.set("Starting local training…")
        self.progress_var.set(0)
        command = [sys.executable, str(Path(__file__).resolve()), "--train-worker", str(config_path)]
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            stop_file = config_path.with_suffix(".stop")
            stop_file.unlink(missing_ok=True)
            config = dict(config)
            config["stop_file"] = str(stop_file)
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            self.stop_file = stop_file
            process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, ValueError) as error:
            self._set_busy(False)
            self.status_var.set("Could not start training")
            messagebox.showerror("Training could not start", str(error))
            return
        self.process = process
        self.stop_button.configure(state="normal")

        def read_output():
            assert process.stdout is not None
            for line in process.stdout:
                self.events.put(("log", line))
            self.events.put(("process_exit", process.wait()))

        threading.Thread(target=read_output, daemon=True).start()

    def _open_output(self):
        folder = Path(self.output_var.get()).expanduser()
        self._open_folder(folder, "output folder")

    def _open_folder(self, folder: Path, description: str):
        folder = folder.expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as error:
            messagebox.showerror("Could not open folder", f"Could not open the {description}:\n{error}")

    def _training_runs_root(self) -> Path:
        return Path(self.output_var.get()).expanduser() / "training_runs"

    def _latest_run_dir(self) -> Path | None:
        root = self._training_runs_root()
        if not root.is_dir():
            return None
        runs = [
            path for path in root.glob("*/*")
            if path.is_dir() and ((path / "best_adapter").is_dir() or (path / "checkpoints").is_dir())
        ]
        return max(runs, key=lambda path: path.stat().st_mtime) if runs else None

    def _latest_checkpoint(self, require_incomplete: bool = False):
        root = self._training_runs_root()
        if not root.is_dir():
            return None
        candidates = []
        for run_dir in root.glob("*/*"):
            checkpoint_root = run_dir / "checkpoints"
            settings_path = run_dir / "TRAINING_SETTINGS.json"
            if require_incomplete and ((run_dir / "TRAINING_COMPLETE.json").exists() or not settings_path.is_file()):
                continue
            if not checkpoint_root.is_dir():
                continue
            for checkpoint in checkpoint_root.glob("checkpoint-*"):
                suffix = checkpoint.name.rsplit("-", 1)[-1]
                if checkpoint.is_dir() and suffix.isdigit():
                    candidates.append((checkpoint.stat().st_mtime, int(suffix), checkpoint, run_dir, settings_path))
        if not candidates:
            return None
        _mtime, _step, checkpoint, run_dir, settings_path = max(
            candidates, key=lambda item: (item[0], item[1], str(item[3]))
        )
        return checkpoint, run_dir, settings_path

    def _refresh_resume_button(self):
        if not hasattr(self, "resume_button"):
            return
        available = not self.busy and self._latest_checkpoint(require_incomplete=True) is not None
        self.resume_button.configure(state="normal" if available else "disabled")

    def _output_changed(self, *_args):
        self._refresh_resume_button()

    def _stop_training(self):
        if self.process is None or self.process.poll() is not None or self.stop_file is None:
            return
        try:
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.write_text("Stop after the current optimizer step and save a checkpoint.\n", encoding="utf-8")
        except OSError as error:
            messagebox.showerror("Could not request stop", str(error))
            return
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stop requested — saving at the next training step…")
        self._append_log("Stop requested. Training will finish its current step and save a resumable checkpoint.")

    def _resume_latest(self):
        if self.busy:
            return
        candidate = self._latest_checkpoint(require_incomplete=True)
        if candidate is None:
            self._refresh_resume_button()
            messagebox.showinfo("No saved run to resume", "There is no incomplete run with a saved checkpoint in this output folder.")
            return
        checkpoint, run_dir, settings_path = candidate
        try:
            config = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not read saved training settings", str(error))
            return
        dataset_value = config.get("dataset_dir")
        if not dataset_value or not Path(dataset_value).is_dir():
            messagebox.showerror("Dataset not found", "The dataset for this saved run is missing. The checkpoint cannot be resumed.")
            return
        config["run_dir"] = str(run_dir.resolve())
        config["resume_from_checkpoint"] = str(checkpoint.resolve())
        output_parent = Path(config.get("output_parent") or self.output_var.get()).expanduser()
        config_path = output_parent / f"resume_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._launch_training(config, config_path)

    def _open_latest_run(self):
        run_dir = self._latest_run_dir()
        if run_dir is None:
            messagebox.showinfo("No saved runs", "No training run files were found in this output folder.")
            return
        self._open_folder(run_dir, "latest training run")

    def _open_latest_adapter(self):
        run_dir = self._latest_run_dir()
        adapter_dir = run_dir / "best_adapter" if run_dir else None
        if adapter_dir is None or not adapter_dir.is_dir():
            messagebox.showinfo("No adapter files", "The latest run has no finished adapter yet. Open its checkpoint files to inspect saved progress.")
            return
        self._open_folder(adapter_dir, "adapter files")

    def _open_latest_checkpoint(self):
        candidate = self._latest_checkpoint()
        if candidate is None:
            messagebox.showinfo("No checkpoints", "No training checkpoint files were found in this output folder.")
            return
        checkpoint, _run_dir, _settings = candidate
        self._open_folder(checkpoint, "checkpoint files")

    def _open_base_model_cache(self):
        if os.environ.get("HF_HUB_CACHE"):
            cache = Path(os.environ["HF_HUB_CACHE"])
        elif os.environ.get("HF_HOME"):
            cache = Path(os.environ["HF_HOME"]) / "hub"
        else:
            cache = Path.home() / ".cache" / "huggingface" / "hub"
        self._open_folder(cache, "base model cache")

    def _handle_progress_status(self, payload: dict):
        if payload.get("stage") == "telemetry":
            self._update_performance(payload)
            return
        message = payload.get("message", payload.get("stage", "Training…"))
        percent = payload.get("percent")
        if percent is not None:
            self.progress_var.set(max(0, min(100, float(percent))))
        stage = payload.get("stage")
        if stage == "training":
            epoch = payload.get("epoch")
            epochs = payload.get("epochs")
            step = payload.get("step")
            total = payload.get("total_steps")
            if epoch is not None and step is not None and total is not None:
                message = f"Training — epoch {epoch}/{epochs}, total progress {step}/{total} steps"
        elif stage == "paused":
            self.performance_finished = True
            self.eta_var.set("Paused — checkpoint saved")
            self.training_paused = True
            self._append_log(message)
        elif stage == "complete":
            self.performance_finished = True
            self.eta_var.set("Complete — adapter ZIP ready")
            self.progress_var.set(100)
        self.status_var.set(message)

    def _drain_events(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "scan_complete":
                _kind, generation, result = event
                if generation != self.scan_generation:
                    continue
                prior = self._raw_target()
                speakers = result["speakers"]
                labels = sorted(speakers.items(), key=lambda item: (-item[1], item[0].casefold()))
                self.speaker_labels = {f"{name}  ({count:,} messages)": name for name, count in labels}
                displays = list(self.speaker_labels)
                self.target_combo.configure(values=displays)
                matching = next((display for display, raw in self.speaker_labels.items() if raw == prior), "")
                self.target_display_var.set(matching or (displays[0] if len(displays) == 1 else ""))
                if self._raw_target() != prior:
                    self._invalidate_dataset()
                self.scan_label.configure(
                    text=f"Scanned {result['source_files']} text file(s): {result['messages']:,} unique messages, {len(displays)} speakers found."
                )
                self.status_var.set("Speaker scan complete" if displays else "No speakers found")
                self._append_log(f"Scan complete: {len(displays)} speakers and {result['messages']:,} messages found.")
                self._schedule_estimate()
            elif kind == "scan_error":
                _kind, generation, error = event
                if generation == self.scan_generation:
                    self.scan_label.configure(text=f"Could not scan files: {error}")
                    self.status_var.set("Speaker scan failed")
                    self._append_log(f"Speaker scan failed: {error}")
            elif kind == "estimate_complete":
                _kind, generation, report = event
                if generation != self.estimate_generation:
                    continue
                counts = report["sets"]
                self.example_label.configure(text=f"Possible examples: {report['example_count']:,}")
                self.split_label.configure(
                    text=f"Estimated split: train {counts['train']:,}  |  validation {counts['validation']:,}  |  test {counts['test']:,}"
                )
                if report["independent_clusters"] < 10:
                    self.example_note.configure(
                        text=f"Found {report['independent_clusters']} independent conversation groups; at least 10 are needed to prepare a split."
                    )
                else:
                    self.example_note.configure(
                        text=f"Estimate uses {report['settings']['max_context_messages']} context messages and up to "
                        f"{report['settings']['max_target_replies']} target replies per example."
                    )
            elif kind == "estimate_error":
                _kind, generation, error = event
                if generation == self.estimate_generation:
                    self.example_label.configure(text="Possible examples: unavailable")
                    self.split_label.configure(text="Estimated split: unavailable")
                    self.example_note.configure(text=error)
            elif kind == "log":
                line = event[1].rstrip()
                if line.startswith(STATUS_PREFIX):
                    try:
                        self._handle_progress_status(json.loads(line[len(STATUS_PREFIX):]))
                    except (ValueError, TypeError):
                        self.status_var.set(line[len(STATUS_PREFIX):])
                elif line:
                    self._append_log(line)
                    if line.startswith(("Run folder:", "GPU:", "Adapter ZIP ready:")):
                        self.status_var.set(line)
            elif kind == "prepared":
                _kind, generation, result = event
                if generation != self.dataset_generation:
                    self._set_busy(False)
                    self.status_var.set("Chat settings changed — prepare the dataset again")
                    continue
                self.dataset_dir = result.dataset_dir
                counts = result.report["sets"]
                self.dataset_label.configure(
                    text=f"Dataset ready · {counts['train']:,} train · {counts['validation']:,} validation · {counts['test']:,} test"
                )
                self.train_button.configure(state="normal")
                self.status_var.set("Dataset ready")
                self.progress_var.set(0)
                self._append_log(f"Dataset ready: {result.dataset_dir}")
                self._append_log(f"Examples — train {counts['train']}, validation {counts['validation']}, test {counts['test']}")
                self._set_busy(False)
            elif kind == "error":
                self._set_busy(False)
                self.status_var.set("Action failed")
                self._append_log(f"{event[1]}: {event[2]}")
                messagebox.showerror(event[1], event[2])
            elif kind == "process_exit":
                return_code = event[1]
                self._set_busy(False)
                self.process = None
                if self.stop_file:
                    try:
                        self.stop_file.unlink(missing_ok=True)
                    except OSError:
                        pass
                if return_code == 0 and self.training_paused:
                    self.status_var.set("Training paused — checkpoint saved; ready to resume")
                    self._append_log("Checkpoint saved. Use Resume latest checkpoint to continue this run.")
                    self._refresh_resume_button()
                elif return_code == 0:
                    self.status_var.set("Training complete — adapter ZIP created")
                    self.progress_var.set(100)
                    self._append_log("Training complete. The trained adapter ZIP is in the training run folder.")
                    self._refresh_resume_button()
                    messagebox.showinfo("Training complete", "Your adapter ZIP is ready in the selected output folder.")
                else:
                    self.performance_finished = True
                    self.eta_var.set("Training stopped — see activity log")
                    self.status_var.set(f"Training stopped with exit code {return_code}")
                    self._append_log(f"Training exited with code {return_code}. See the log above for details.")
                    self._refresh_resume_button()
        self.root.after(100, self._drain_events)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-worker")
    parser.add_argument("--self-check", action="store_true")
    args, _ = parser.parse_known_args()
    if args.train_worker:
        run_training_worker(args.train_worker)
        return
    if args.self_check:
        if sys.stdout is None:
            sys.stdout = os.fdopen(1, "w", encoding="utf-8", buffering=1, closefd=False)
        if sys.stderr is None:
            sys.stderr = os.fdopen(2, "w", encoding="utf-8", buffering=1, closefd=False)
        root = tk.Tk()
        root.withdraw()
        app = FriendLLMApp(root)
        root.update()
        assert set(app.pages) == {"main", "settings"}
        assert app.current_page == "main"
        app._toggle_settings()
        root.update()
        assert app.current_page == "settings"
        app._toggle_settings()
        assert app.current_page == "main"
        root.destroy()
        import torch
        import transformers
        import peft
        import psutil
        from train import _select_runtime
        print(json.dumps({"self_check": "passed", "runtime": _select_runtime(torch),
                          "torch": torch.__version__, "transformers": transformers.__version__,
                          "peft": peft.__version__, "psutil": psutil.__version__}), flush=True)
        return
    root = tk.Tk()
    FriendLLMApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
