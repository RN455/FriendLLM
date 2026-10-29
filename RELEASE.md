# FriendLLM Desktop — initial release

## App download

The ready-to-run Windows build is packaged as `FriendLLM-Windows.zip`. Extract **all** files and run `FriendLLM\FriendLLM.exe`. Keep its `_internal` folder together with the EXE. Python setup is not required for the compiled app. Base model weights download on the first training run; chat data and trained models are not included in this download.

This build includes the NVIDIA CUDA runtime and supports CPU fallback. Other GPU vendors require a build/source environment containing the matching PyTorch backend. A GPU is optional, but CPU training of the default 4B model needs substantial RAM and is slow.

If the app ZIP exceeds GitHub's per-asset limit, the `releases\GitHub-Release` folder contains numbered parts, `release-manifest.json`, and two extraction helpers. Upload **all** files in that folder as release assets. Users download them into one empty folder and double-click `Extract-FriendLLM.bat`. It checks every part, joins and verifies the ZIP, and extracts the app using Windows tools. No separate archiver is required. Allow about 10 GB of free disk space while joining and extracting the CUDA package.

## Included features

- WhatsApp TXT/ZIP import, speaker picker, and usable-example estimates.
- Advanced dataset context, model, and training settings.
- Automatic supported-device selection and optional CPU-only training.
- Training stage/epoch progress, stop-and-save checkpoint, and resume.
- Measured remaining-time estimates, CPU/RAM/GPU statistics, usage graphs, and saved performance samples. Estimates begin after three optimizer steps. Loading/downloads/final packaging add time; validation/checkpoint overhead is included after it has been measured.
- File Explorer shortcuts to model, adapter, checkpoint, and training files.

## Prepare a GitHub repository

The project root contains source and build scripts. `releases`, `build`, `dist`, environments, private chat exports, datasets, and model artifacts are ignored. `releases\FriendLLM-Source.zip` is a clean source archive with an explicit allowlist of project files; extract it before using GitHub's file-upload interface.

1. Create an empty GitHub repository.
2. Commit the project source using GitHub Desktop or Git from this project folder. Do not commit compiled application files or private training data.
3. Add that repository as the remote and push the source.
4. Create a release, for example `v0.1.0`, using the prepared `releases\GitHub-Release-Notes.md` as its description. Attach every file from `releases\GitHub-Release`; that folder also includes the clean source ZIP and checksums.

No remote repository or public download URL is configured by these preparation scripts. Project licensing has not been selected; third-party dependencies and models retain their own licenses.

## Rebuild and package

Run `Setup.bat`, then `Build EXE.bat`. After the build, run:

```powershell
.\.venv\Scripts\python.exe package_release.py
```

The packager creates a normal app ZIP, verifies every ZIP member's CRC, builds the clean source archive, computes SHA-256 checksums, and prepares release assets below GitHub's 2 GiB per-file limit.

Before packaging another release into the same output folder, move the previous `GitHub-Release` folder aside so assets from different builds are not mixed.

For a packaged dependency/UI/device check without downloading a model, run `FriendLLM.exe --self-check` from a terminal or capture its output in a subprocess. The check runs without displaying the main window.
