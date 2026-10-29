"""Local LoRA training with automatic accelerator selection and CPU fallback."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from telemetry import TrainingMonitor


RPG_REPO = "Chun121/qwen3-4B-rpg-roleplay"
EXPECTED_BASE = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
PORTABLE_BASE = "Qwen/Qwen3-4B"
STATUS_PREFIX = "FRIENDLLM_STATUS "


def _status(stage: str, message: str, **values):
    print(STATUS_PREFIX + json.dumps({"stage": stage, "message": message, **values}, ensure_ascii=False), flush=True)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-.") or "FriendLLM"


def _chat_text(tokenizer, messages, *, add_generation_prompt: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _tokenize_record(tokenizer, record: dict, system_prompt: str, max_length: int):
    chat = [{"role": "system", "content": system_prompt}, *record["prompt"]]
    completion = record["completion"][0]["content"]
    prompt_ids = tokenizer(
        _chat_text(tokenizer, chat, add_generation_prompt=True), add_special_tokens=False
    ).input_ids
    response_ids = tokenizer(completion, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        response_ids.append(tokenizer.eos_token_id)
    if len(response_ids) > max_length // 2:
        return None
    while len(prompt_ids) + len(response_ids) > max_length and len(chat) > 2:
        chat.pop(1)
        while len(chat) > 2 and chat[1]["role"] == "assistant":
            chat.pop(1)
        prompt_ids = tokenizer(
            _chat_text(tokenizer, chat, add_generation_prompt=True), add_special_tokens=False
        ).input_ids
    if len(prompt_ids) + len(response_ids) > max_length:
        return None
    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


class _EncodedDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class _CompletionCollator:
    def __init__(self, pad_id: int, torch_module):
        self.pad_id = pad_id
        self.torch = torch_module

    def __call__(self, examples):
        max_length = max(len(example["input_ids"]) for example in examples)
        max_length = ((max_length + 7) // 8) * 8

        def pad(values, fill):
            return list(values) + [fill] * (max_length - len(values))

        ids = [pad(example["input_ids"], self.pad_id) for example in examples]
        masks = [pad(example["attention_mask"], 0) for example in examples]
        labels = [pad(example["labels"], -100) for example in examples]
        return {
            "input_ids": self.torch.tensor(ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(masks, dtype=self.torch.long),
            "labels": self.torch.tensor(labels, dtype=self.torch.long),
        }


def _read_split(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_runtime(torch_module, preference: str = "auto") -> dict:
    from accelerate import PartialState

    if preference not in {"auto", "cpu"}:
        raise ValueError("Training device must be auto or cpu.")
    device = PartialState(cpu=preference == "cpu").device
    backend = device.type
    nvidia = backend == "cuda" and not getattr(torch_module.version, "hip", None)
    quantized = (
        nvidia and torch_module.cuda.get_device_capability(device)[0] >= 6
        and importlib.util.find_spec("bitsandbytes") is not None
    )
    if quantized:
        try:
            import bitsandbytes as bnb
            # Importing the package alone does not establish kernel/driver support.
            bnb.functional.quantize_4bit(
                torch_module.zeros((64, 64), device=device), quant_type="nf4"
            )
        except Exception as error:
            quantized = False
            print(f"4-bit acceleration unavailable; using standard LoRA: {error}", flush=True)
    accelerator = getattr(torch_module, backend, None)
    bf16 = backend in {"cuda", "xpu"} and bool(
        getattr(accelerator, "is_bf16_supported", lambda: False)()
    )
    fp16 = backend == "cuda" and not bf16
    return {
        "device": str(device),
        "backend": "rocm" if backend == "cuda" and not nvidia else backend,
        "quantized": quantized,
        "precision": "bf16" if bf16 else "fp16" if fp16 else "fp32",
        "optimizer": "paged_adamw_8bit" if quantized else "adamw_torch",
    }


def train(config: dict, monitor=None):
    # Heavy ML imports happen only after the user starts training.
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoConfig,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    runtime = _select_runtime(torch, config.get("device_preference", "auto"))
    device = torch.device(runtime["device"])
    quantized = runtime["quantized"]
    bf16 = runtime["precision"] == "bf16"
    fp16 = runtime["precision"] == "fp16"
    compute_dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    _status("hardware", f"Training device: {device} ({runtime['backend']}); {runtime['precision']}, "
            + ("4-bit LoRA" if quantized else "standard LoRA"))
    if device.type == "cpu":
        print("CPU selected. Training a 4B model uses substantial RAM and can be very slow. "
              "Install the PyTorch backend for your GPU to enable supported acceleration.", flush=True)

    dataset_dir = Path(config["dataset_dir"]).resolve()
    output_parent = Path(config["output_parent"]).resolve()
    target = str(config["target_username"]).strip()
    model_name = str(config.get("model_name", "FriendLLM")).strip() or "FriendLLM"
    max_length = int(config.get("max_seq_length", 1024))
    epochs = int(config.get("epochs", 2))
    batch_size = int(config.get("batch_size", 1))
    grad_accum = int(config.get("grad_accum", 16))
    learning_rate = float(config.get("learning_rate", 5e-5))
    seed = int(config.get("seed", 3407))
    requested_base = str(config.get("base_model_id", EXPECTED_BASE)).strip()
    base_model = PORTABLE_BASE if not quantized and requested_base == EXPECTED_BASE else requested_base
    adapter_repo = str(config.get("adapter_repo_id", RPG_REPO)).strip()
    runtime["base_model_id"] = base_model
    saved_runtime = config.get("runtime")
    if config.get("resume_from_checkpoint"):
        if saved_runtime and any(saved_runtime.get(key) != runtime[key]
                                 for key in ("quantized", "base_model_id", "optimizer", "precision")):
            raise ValueError("This checkpoint used a different model, precision, or optimizer. "
                             "Resume with the original device and training dependencies.")
        if not saved_runtime and requested_base == EXPECTED_BASE and not quantized:
            raise ValueError("This older checkpoint requires the original NVIDIA 4-bit training environment.")
    config["runtime"] = runtime

    if not target or not (dataset_dir / "preparation_report.json").is_file():
        raise ValueError("The dataset folder is missing its preparation report.")
    if not base_model or not adapter_repo:
        raise ValueError("A Hugging Face base model and compatible LoRA adapter repository are required.")
    if max_length < 256 or max_length > 4096:
        raise ValueError("Sequence length must be between 256 and 4096 tokens.")
    if not 1 <= epochs <= 10 or not 1 <= batch_size <= 4 or not 1 <= grad_accum <= 128:
        raise ValueError("Training settings are outside the supported range.")

    report = json.loads((dataset_dir / "preparation_report.json").read_text(encoding="utf-8"))
    if report.get("target_username") != target:
        raise ValueError("The dataset target does not match the selected speaker.")
    source_splits = {name: _read_split(dataset_dir / f"{name}.jsonl") for name in ("train", "validation", "test")}
    if len(source_splits["train"]) < 20 or len(source_splits["validation"]) < 5 or len(source_splits["test"]) < 5:
        raise ValueError("The prepared split is too small. Import more conversation data before training.")

    model_slug = _slug(model_name)
    target_slug = _slug(target)
    resume_from_checkpoint = config.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint).resolve()
        run_dir = Path(config.get("run_dir") or checkpoint_path.parent.parent).resolve()
        expected_checkpoints = (run_dir / "checkpoints").resolve()
        if checkpoint_path.parent != expected_checkpoints or not checkpoint_path.is_dir():
            raise ValueError("The selected resume checkpoint is not inside this training run.")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Training run folder not found: {run_dir}")
        if (run_dir / "TRAINING_COMPLETE.json").is_file():
            raise ValueError("This run is already complete and cannot be resumed.")
    else:
        run_root = output_parent / "training_runs" / f"{model_slug}_{target_slug}"
        run_root.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        run_dir = run_root / run_id
        serial = 1
        while run_dir.exists():
            run_dir = run_root / f"{run_id}_{serial}"
            serial += 1
        run_dir.mkdir(parents=True)
    config["run_dir"] = str(run_dir)
    if monitor is not None:
        monitor.configure(torch, device, run_dir)
    settings_copy = {key: value for key, value in config.items() if key not in {"stop_file", "resume_from_checkpoint"}}
    (run_dir / "TRAINING_SETTINGS.json").write_text(
        json.dumps(settings_copy, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    stop_file = Path(config.get("stop_file") or (run_dir / "STOP_AND_SAVE")).resolve()
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}", flush=True)
    if resume_from_checkpoint:
        print(f"Resuming checkpoint: {resume_from_checkpoint}", flush=True)
    print(f"Training device: {device} ({runtime['backend']}); precision: {runtime['precision']}; "
          f"optimizer: {runtime['optimizer']}", flush=True)
    if base_model != requested_base:
        print(f"Using {base_model}, the standard Qwen3 4B weights, for this device instead of the "
              "prequantized NVIDIA preset. The same starter adapter is retained.", flush=True)
    _status("downloading_adapter", f"Downloading starter adapter: {adapter_repo}")
    adapter_snapshot = Path(snapshot_download(
        repo_id=adapter_repo,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors", "lora_model/*"],
    ))
    adapter_path = next((
        path for path in (adapter_snapshot, adapter_snapshot / "lora_model")
        if (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file()
    ), None)
    if adapter_path is None:
        raise FileNotFoundError(
            f"Adapter {adapter_repo} must contain adapter_config.json and adapter_model.safetensors "
            "at its root or in a lora_model folder."
        )
    config_path = adapter_path / "adapter_config.json"
    upstream_config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter_base = str(upstream_config.get("base_model_name_or_path", "")).rstrip("/")
    qwen_variants = {EXPECTED_BASE.casefold(), PORTABLE_BASE.casefold()}
    if (adapter_base.casefold() != base_model.rstrip("/").casefold()
            and not {adapter_base.casefold(), base_model.rstrip("/").casefold()} <= qwen_variants):
        raise ValueError(
            f"The selected adapter is built for {adapter_base or 'an unspecified base model'}, "
            f"but the selected starter model is {base_model}. Choose a matching model and adapter."
        )

    model_options = {
        "torch_dtype": compute_dtype,
        "device_map": {"": str(device)},
        "low_cpu_mem_usage": True,
    }
    if quantized:
        model_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_config = AutoConfig.from_pretrained(base_model)
        if getattr(model_config, "quantization_config", None):
            raise ValueError("This custom base model is already quantized. Select its standard, "
                             "unquantized weights for CPU or non-NVIDIA training.")
    _status("downloading_base", f"Downloading or loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_options)
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if not 0 < trainable < total / 10:
        raise RuntimeError(f"Unexpected trainable parameter count: {trainable:,} of {total:,}")
    print(f"Starter adapter loaded: {trainable:,} trainable parameters", flush=True)

    _status("loading_tokenizer", f"Loading tokenizer for {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_prompt = (
        f"You are a fictional chat character inspired by {target}'s texting style, "
        "not the real person. Chat naturally, casually and briefly. "
        "Do not invent private facts about actual people."
    )

    tokenized = {}
    for name, rows in source_splits.items():
        _status("tokenizing", f"Preparing {name} examples…")
        converted = [_tokenize_record(tokenizer, row, system_prompt, max_length) for row in rows]
        kept = [row for row in converted if row is not None]
        minimum = 20 if name == "train" else 5
        if len(kept) < minimum:
            raise ValueError(f"Only {len(kept)} {name} examples fit the sequence limit; need {minimum}.")
        tokenized[name] = _EncodedDataset(kept)
        print(f"{name}: {len(kept)} examples tokenized; {len(converted) - len(kept)} skipped", flush=True)

    set_seed(seed)
    use_signature = inspect.signature(TrainingArguments.__init__).parameters
    args_values = {
        "output_dir": str(run_dir / "checkpoints"),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "warmup_ratio": .05,
        "lr_scheduler_type": "cosine",
        "bf16": bf16,
        "fp16": fp16,
        "use_cpu": device.type == "cpu",
        "optim": runtime["optimizer"],
        "weight_decay": .01,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": device.type in {"cuda", "xpu"},
    }
    if "eval_strategy" in use_signature:
        args_values["eval_strategy"] = "epoch"
    else:
        args_values["evaluation_strategy"] = "epoch"
    args_values = {key: value for key, value in args_values.items() if key in use_signature}
    training_args = TrainingArguments(**args_values)

    class LogCallback(TrainerCallback):
        def __init__(self):
            self.stop_requested = False
            self.saved_checkpoint: Path | None = None

        def on_train_begin(self, args, state, control, **kwargs):
            total = max(1, int(state.max_steps or 1))
            step = min(total, int(state.global_step))
            if monitor is not None:
                monitor.begin(step, total)
            epoch = min(epochs, max(1, int(float(state.epoch or 0) + 0.999999)))
            _status("training", f"Training epoch progress: {epoch}/{epochs}; total progress: {step}/{total} steps",
                    epoch=epoch, epochs=epochs, step=step, total_steps=total, percent=100.0 * step / total)

        def on_step_end(self, args, state, control, **kwargs):
            total = max(1, int(state.max_steps or 1))
            step = min(total, int(state.global_step))
            if monitor is not None:
                monitor.progress(step, total)
            epoch = min(epochs, max(1, int(float(state.epoch or 0) + 0.999999)))
            percent = 100.0 * step / total
            _status(
                "training",
                f"Training epoch progress: {epoch}/{epochs}; total progress: {step}/{total} steps",
                epoch=epoch, epochs=epochs, step=step, total_steps=total, percent=percent,
            )
            if stop_file.is_file() and not self.stop_requested:
                self.stop_requested = True
                control.should_save = True
                control.should_training_stop = True
                _status("saving_checkpoint", "Stop requested — saving a resumable checkpoint at this step…")

        def on_epoch_end(self, args, state, control, **kwargs):
            if monitor is not None:
                monitor.epoch_end()

        def on_save(self, args, state, control, **kwargs):
            if monitor is not None:
                monitor.saved()
            if self.stop_requested:
                self.saved_checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                _status("checkpoint_saved", f"Resumable checkpoint saved: {self.saved_checkpoint}",
                        checkpoint=str(self.saved_checkpoint))

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                print(f"step {state.global_step}: {json.dumps(logs, ensure_ascii=False)}", flush=True)

    progress_callback = LogCallback()
    trainer_values = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": _CompletionCollator(tokenizer.pad_token_id, torch),
        "callbacks": [progress_callback],
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_values["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_values["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_values)
    _status("training", "Training started. Checkpoints are saved locally.")
    result = trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    print(f"Training finished: {result.metrics}", flush=True)

    if progress_callback.stop_requested:
        checkpoint = progress_callback.saved_checkpoint
        if checkpoint is None or not checkpoint.is_dir():
            candidates = sorted(
                (run_dir / "checkpoints").glob("checkpoint-*"),
                key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1,
            )
            checkpoint = candidates[-1] if candidates else None
        if checkpoint is None:
            raise RuntimeError("Training stopped, but no resumable checkpoint was found.")
        paused = {
            "run": run_dir.name,
            "checkpoint": str(checkpoint),
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
        }
        (run_dir / "TRAINING_PAUSED.json").write_text(
            json.dumps(paused, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        stop_file.unlink(missing_ok=True)
        _status("paused", "Training paused and checkpoint saved. Use Resume latest checkpoint to continue.",
                checkpoint=str(checkpoint), percent=100.0 * trainer.state.global_step / max(1, trainer.state.max_steps))
        print(f"Training paused at step {trainer.state.global_step}; checkpoint: {checkpoint}", flush=True)
        return {"run_dir": run_dir, "checkpoint": checkpoint, "paused": True}

    _status("saving_adapter", "Saving the trained adapter and tokenizer…")
    final_adapter = run_dir / "best_adapter"
    trainer.model.save_pretrained(str(final_adapter), safe_serialization=True)
    tokenizer.save_pretrained(str(final_adapter))
    adapter_config_file = final_adapter / "adapter_config.json"
    adapter_weights_file = final_adapter / "adapter_model.safetensors"
    if not adapter_config_file.is_file() or not adapter_weights_file.is_file():
        raise RuntimeError("Adapter files were not saved correctly.")

    summary = {
        "run": run_dir.name,
        "target_username": target,
        "model_name": model_name,
        "base_model": base_model,
        "starter_adapter_repo": adapter_repo,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "training_metrics": result.metrics,
        "dataset_dir": str(dataset_dir),
        "dataset_counts": {name: len(rows) for name, rows in source_splits.items()},
        "precision": runtime["precision"],
        "runtime": runtime,
        "performance": monitor.summary() if monitor is not None else {},
        "max_seq_length": max_length,
        "epochs": epochs,
    }
    summary_path = run_dir / "TRAINING_COMPLETE.json"
    adapter_hash = _sha256_file(adapter_weights_file)
    archive_path = run_dir / f"{model_slug}_Trained_Adapter.zip"
    _status("packaging_adapter", "Packaging and checking the trained adapter ZIP…")
    manifest = {
        "run": run_dir.name,
        "target_username": target,
        "adapter_sha256": adapter_hash,
        "includes_gguf": False,
    }
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for file in sorted(final_adapter.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(final_adapter).as_posix())
        archive.writestr("TRAINING_COMPLETE.json", json.dumps(summary, indent=2, ensure_ascii=False))
        archive.writestr("TRAINING_RUN.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("The adapter ZIP failed its integrity check.")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "TRAINING_PAUSED.json").unlink(missing_ok=True)
    size_mib = archive_path.stat().st_size / (1024 * 1024)
    print(f"Adapter ZIP ready: {archive_path} ({size_mib:.1f} MiB)", flush=True)
    print(f"Adapter SHA256: {adapter_hash}", flush=True)
    print("The ZIP contains the adapter only. A standalone GGUF export is not included.", flush=True)
    _status("complete", "Training complete — adapter ZIP created", percent=100)
    return {"run_dir": run_dir, "archive_path": archive_path}


def main(config_path: str):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    with TrainingMonitor(_status) as monitor:
        train(config, monitor=monitor)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: train.py CONFIG.json")
    main(sys.argv[1])
