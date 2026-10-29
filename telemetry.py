"""Measured training estimates and resource samples; no ML imports at UI startup."""

from collections import deque
from datetime import datetime, timezone
import json
import shutil
import subprocess
import threading
import time


def duration(seconds):
    if seconds is None:
        return "Calculating…"
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


class TrainingMonitor:
    def __init__(self, publish, interval=2.0, clock=time.monotonic):
        self.publish, self.interval, self.clock = publish, interval, clock
        self.started = clock()
        self.session = datetime.now(timezone.utc).isoformat()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.torch = self.device = self.run_dir = None
        self.step_times = deque(maxlen=60)
        self.step_count = self.step_seconds = 0
        self.overheads = []
        self.last_step_at = self.last_step = self.total_steps = None
        self.boundary_at = None
        self.latest = {}
        self.statistics = {}
        self.gpu_query = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=4)

    def configure(self, torch_module, device, run_dir):
        with self.lock:
            self.torch, self.device, self.run_dir = torch_module, device, run_dir
            if device.type == "cuda" and not getattr(torch_module.version, "hip", None):
                executable = shutil.which("nvidia-smi")
                uuid = getattr(torch_module.cuda.get_device_properties(device), "uuid", None)
                if executable and uuid:
                    identifier = str(uuid)
                    if not identifier.startswith("GPU-"):
                        identifier = "GPU-" + identifier
                    self.gpu_query = [executable, "-i", identifier, "--query-gpu=utilization.gpu",
                                      "--format=csv,noheader,nounits"]

    def begin(self, step, total):
        with self.lock:
            self.last_step, self.total_steps, self.last_step_at = step, total, self.clock()

    def progress(self, step, total):
        with self.lock:
            now = self.clock()
            count = step - self.last_step if self.last_step is not None else 0
            if count > 0:
                elapsed = now - self.last_step_at
                self.step_times.append(elapsed / count)
                self.step_seconds += elapsed
                self.step_count += count
            self.last_step, self.total_steps, self.last_step_at = step, total, now

    def epoch_end(self):
        with self.lock:
            self.boundary_at = self.clock()

    def saved(self):
        with self.lock:
            if self.boundary_at is not None:
                self.overheads.append(self.clock() - self.boundary_at)
                self.boundary_at = None
                self.last_step_at = self.clock()

    def timing(self):
        with self.lock:
            average = self.step_seconds / self.step_count if self.step_count else None
            eta = None
            if len(self.step_times) >= 3:
                recent = sum(self.step_times) / len(self.step_times)
                # Blend recent speed with the session mean to reduce noisy estimates.
                eta = max(0, self.total_steps - self.last_step) * (0.6 * recent + 0.4 * average)
                # Budget future evaluation/checkpoints from observed epoch overhead.
                if self.overheads and self.last_step:
                    steps_per_epoch = self.last_step / len(self.overheads)
                    remaining_epochs = max(0, (self.total_steps - self.last_step) / steps_per_epoch)
                    eta += remaining_epochs * sum(self.overheads) / len(self.overheads)
            return {"elapsed_seconds": self.clock() - self.started,
                    "eta_seconds": eta, "average_step_seconds": average,
                    "measured_steps": self.step_count,
                    "epoch_overhead_seconds": sum(self.overheads) / len(self.overheads) if self.overheads else None}

    def summary(self):
        with self.lock:
            return {**self.timing(), "resource_statistics": dict(self.statistics)}

    def _sample_loop(self):
        try:
            import psutil
            process = psutil.Process()
            psutil.cpu_percent()
        except (ImportError, OSError):
            return
        while not self.stop.wait(self.interval):
            try:
                ram = psutil.virtual_memory()
                sample = {"cpu_percent": psutil.cpu_percent(), "ram_percent": ram.percent,
                          "process_ram_gib": process.memory_info().rss / 1024**3}
                with self.lock:
                    torch_module, device, gpu_query = self.torch, self.device, self.gpu_query
                if torch_module is not None and device is not None:
                    try:
                        if device.type in {"cuda", "xpu"}:
                            backend = getattr(torch_module, device.type)
                            sample["gpu_ram_gib"] = backend.memory_allocated(device) / 1024**3
                        elif device.type == "mps":
                            sample["gpu_ram_gib"] = torch_module.mps.current_allocated_memory() / 1024**3
                    except (RuntimeError, AttributeError, NotImplementedError):
                        pass
                if gpu_query:
                    try:
                        result = subprocess.run(gpu_query, capture_output=True, text=True, timeout=1.5,
                                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        if result.returncode == 0:
                            sample["gpu_percent"] = float(result.stdout.strip())
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        pass
                with self.lock:
                    for name, value in sample.items():
                        prior = self.statistics.get(name, {"samples": 0, "mean": 0, "peak": 0})
                        count = prior["samples"] + 1
                        self.statistics[name] = {"samples": count,
                            "mean": prior["mean"] + (value - prior["mean"]) / count,
                            "peak": max(prior["peak"], value)}
                    payload = {**sample, **self.timing(), "statistics": dict(self.statistics),
                               "session_started_utc": self.session,
                               "device": str(device) if device is not None else "Detecting…"}
                    self.latest = payload
                    run_dir = self.run_dir
                if run_dir is not None:
                    with (run_dir / "performance.jsonl").open("a", encoding="utf-8") as output:
                        output.write(json.dumps(payload) + "\n")
                self.publish("telemetry", "Performance sample", **payload)
            except Exception:
                # Resource monitoring must never interrupt training or checkpoint saving.
                continue
