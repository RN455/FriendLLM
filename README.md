# FriendLLM for Windows

FriendLLM is a Windows desktop app for preparing WhatsApp chat exports and locally fine-tuning a Qwen 3 4B LoRA adapter. It scans selected `.txt` exports (including `.txt` files inside ZIP archives), lets you choose a speaker, estimates the available examples, and prepares train, validation, and test splits.

The app uses a native Tkinter interface with one training page: upload chats, choose a speaker, select an output folder, prepare the dataset, and start training. Performance and output activity sit beside each other below the training controls, with shortcuts to saved files underneath. The Settings button opens a separate page for chat parsing, aliases, dataset context, sequence length, training settings, and model selection. Both pages scroll. The recommended starter is the Qwen3 4B roleplay preset configured in `app.py`. During training, the app reports download and epoch progress, can stop after saving a resumable checkpoint, and can resume the latest saved checkpoint.

The Performance panel displays measured time remaining, elapsed time, average optimizer-step time, CPU/RAM/GPU usage, session averages and peaks, and a graph of recent usage. Estimates start after three measured optimizer steps and update with recent and average training speed. Evaluation/checkpoint overhead is included once observed; downloads, model loading, and final packaging can add time. GPU load is shown when the backend can report it (the Windows NVIDIA build uses `nvidia-smi`); unavailable values are labeled rather than estimated. Samples are saved to `performance.jsonl` in each training run, with a summary in `TRAINING_COMPLETE.json`. Resuming starts a new measurement session while preserving earlier sample records.

## Requirements

- Windows 10 or 11, 64-bit
- Python 3.14, 64-bit, with the Python launcher (`py`)
- Enough system RAM for the selected model; a supported GPU is optional and needs enough VRAM and a compatible driver/PyTorch backend
- Internet access for setup and the first download of the base model and starter adapter
- Several GB of free disk space for the Python environment, model cache, checkpoints, and build output

Chat parsing and dataset preparation can run without setting up the training dependencies. Training automatically selects a backend exposed by PyTorch/Accelerate, including NVIDIA CUDA, AMD ROCm, Intel XPU, and Apple MPS. If no supported accelerator is available, it runs on CPU. Settings also offers **CPU only**. GPU acceleration depends on the operating system, driver, and installed PyTorch build; an arbitrary GPU is not guaranteed to be supported. This training stack does not use DirectML.

On supported NVIDIA GPUs with working bitsandbytes kernels, training uses the original 4-bit preset. Other devices use standard LoRA with a normal PyTorch optimizer. For the recommended preset, this loads `Qwen/Qwen3-4B`, the corresponding standard Qwen3 4B weights, with the same starter adapter. This path needs more memory; CPU training of a 4B model can be very slow. There is no fixed 10 GB GPU requirement, but the model and selected context/batch sizes must fit in available memory. A memory error requires lowering settings, choosing a smaller compatible custom model/adapter, or using CPU with enough RAM.

## Run from source

1. Install Python 3.14 (64-bit) and the driver for your GPU, if using one.
2. Double-click `Setup.bat` to create `.venv` and install the requirements. On a new environment it detects NVIDIA/Intel Arc hardware and selects CUDA/XPU wheels, or installs CPU wheels otherwise. It preserves an already installed PyTorch build. This downloads several GB.
3. Double-click `Run FriendLLM.bat`.
4. Add WhatsApp `.txt` exports or ZIP archives, scan them, choose the speaker to train, and review the example estimate.
5. Set the output folder, review any options in Settings, and prepare the dataset.
6. Click Start training, review the data-use conditions, and confirm permission and consent before continuing. Resuming a checkpoint requires a fresh confirmation too.

The default output folder is `%USERPROFILE%\FriendLLM`. The app saves prepared datasets, training settings, checkpoints, adapters, and adapter ZIPs there. The Hugging Face model cache is normally under `%USERPROFILE%\.cache\huggingface\hub`.

For an explicit Windows setup, run `Setup.bat cuda`, `Setup.bat xpu`, or `Setup.bat cpu` from a terminal. For AMD ROCm or another supported distribution, install the matching PyTorch build into `.venv` before running setup, or set `FRIENDLLM_TORCH_INDEX` to its wheel index and run `Setup.bat auto`. ROCm support depends on the exact GPU and operating system; setup does not install GPU drivers. A compiled app contains the PyTorch build used to create it, so using another vendor's backend can require running from source or building with that backend.

The selected device, precision, and optimizer are shown in the activity log and saved with each training run. Resume reuses the saved settings and rejects incompatible model/precision/optimizer changes rather than silently changing the training method.

## Build the Windows app

With setup complete, run `Build EXE.bat`. PyInstaller creates an onedir application at `dist\FriendLLM\FriendLLM.exe`. Keep the entire `dist\FriendLLM` folder together when moving or launching the built app; the executable uses files in its `_internal` directory. The build is large because it includes PyTorch and the selected backend's runtime libraries. Base model weights are downloaded separately when training starts.

## Privacy and generated files

Chat exports are parsed and used for training on the local computer. The app downloads the selected public base model and starter adapter from Hugging Face; it does not upload chat exports. Treat prepared datasets, checkpoints, and adapters as sensitive personal data.

Before each training or resume action, a data-use agreement asks you to confirm that everyone whose messages are included has consented, that you will protect personal data and obtain permission before sharing, and that you accept the displayed conditions. All three confirmations are required. Cancelling starts no training process. The confirmation time, agreement version, and displayed conditions are recorded in the saved training settings. This records your acknowledgement; the app cannot independently verify permission or consent.

This repository contains application source and build instructions only. It does not include chat exports, prepared datasets, Hugging Face model weights, checkpoints, trained adapters, the virtual environment, or compiled build output. Those files are excluded by `.gitignore`.

## Notes

- The recommended base model and starter adapter are set in `app.py` as `unsloth/Qwen3-4B-unsloth-bnb-4bit` and `Chun121/qwen3-4B-rpg-roleplay`.
- Custom base models require a compatible LoRA adapter. Its `adapter_config.json` and `adapter_model.safetensors` must be at the repository root or inside a `lora_model` folder.
- Training and dataset options are available on the Settings page. Review them before starting a run.
- Use chat exports only when you have appropriate permission. Models trained on personal conversations are simulations and may reproduce sensitive details.
