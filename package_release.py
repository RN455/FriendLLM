"""Create the app ZIP, source ZIP, checksums, and GitHub-sized release assets."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile


SOURCE_FILES = (
    ".gitignore", "app.py", "core.py", "train.py", "telemetry.py", "requirements.txt",
    "FriendLLM.spec", "Setup.bat", "Run FriendLLM.bat", "Build EXE.bat", "README.md",
    "RELEASE.md", "package_release.py",
)

EXTRACT_SCRIPT = r'''$ErrorActionPreference = 'Stop'
function Get-Sha256([string]$Path) {
    $file = [IO.File]::OpenRead($Path)
    $hash = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hash.ComputeHash($file))).Replace('-', '').ToLowerInvariant() }
    finally { $hash.Dispose(); $file.Dispose() }
}
try {
    $manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'release-manifest.json') -Raw | ConvertFrom-Json
    $target = Join-Path $PSScriptRoot 'FriendLLM'
    if (Test-Path -LiteralPath $target) { throw 'A FriendLLM folder already exists here. Move these downloads to a new empty folder first.' }
    foreach ($part in $manifest.parts) {
        if ([IO.Path]::GetFileName($part.name) -ne $part.name) { throw 'Invalid part filename.' }
        $path = Join-Path $PSScriptRoot $part.name
        if (!(Test-Path -LiteralPath $path)) { throw "Download the missing file: $($part.name)" }
        Write-Host "Checking $($part.name)..."
        if ((Get-Item -LiteralPath $path).Length -ne $part.bytes -or
            (Get-Sha256 $path) -ne $part.sha256) {
            throw "Checksum mismatch: $($part.name). Download it again."
        }
    }
    $archive = Join-Path $PSScriptRoot 'FriendLLM-Windows.zip'
    if (Test-Path -LiteralPath $archive) { throw 'Move the existing FriendLLM-Windows.zip out of this folder before joining parts.' }
    $stream = [IO.File]::Open($archive, [IO.FileMode]::CreateNew)
    try {
        foreach ($part in $manifest.parts) {
            $inputStream = [IO.File]::OpenRead((Join-Path $PSScriptRoot $part.name))
            try { $inputStream.CopyTo($stream) } finally { $inputStream.Dispose() }
        }
    } finally { $stream.Dispose() }
    if ((Get-Sha256 $archive) -ne $manifest.sha256) {
        throw 'The joined ZIP failed its checksum check.'
    }
    Write-Host 'Extracting FriendLLM. This can take several minutes...'
    [Reflection.Assembly]::LoadWithPartialName('System.IO.Compression.FileSystem') | Out-Null
    [IO.Compression.ZipFile]::ExtractToDirectory($archive, $PSScriptRoot)
    Remove-Item -LiteralPath $archive
    Write-Host "Ready: $target\FriendLLM.exe"
    exit 0
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
'''


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def add_notices(app_dir, source_root):
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "psutil", "pyinstaller"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    info = {"built_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
            "dependencies": versions, "source_sha256": {name: sha256(source_root / name) for name in SOURCE_FILES}}
    (app_dir / "BUILD-INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (app_dir / "START-HERE.txt").write_text(
        "FriendLLM Desktop for Windows\n\n"
        "Extract the whole ZIP, then double-click FriendLLM.exe. Keep _internal with the EXE.\n"
        "Python installation is not needed. Public base model files download when training starts.\n"
        "This build accelerates on supported NVIDIA GPUs and falls back to CPU. Other GPU backends need a matching build.\n"
        "CPU training of the default 4B model is slow and needs substantial RAM.\n\n"
        "Add WhatsApp TXT/ZIP files, choose a speaker, prepare a dataset, and start training.\n"
        "Settings opens model/context/training options. The main page shows Performance and Output activity side by side.\n"
        "Before starting or resuming training, accept the data-use conditions and confirm permission and consent for all included messages.\n"
        "Stop & save checkpoint pauses training so it can be resumed later.\n"
        "Chat exports, datasets and checkpoints stay local. No personal chats or model weights are included in this app ZIP.\n",
        encoding="utf-8")
    licenses = app_dir / "THIRD-PARTY-LICENSES"
    licenses.mkdir(exist_ok=True)
    for distribution in metadata.distributions():
        package_name = re.sub(r"[^A-Za-z0-9_.-]", "_", distribution.metadata.get("Name", "package"))
        for relative in distribution.files or []:
            path = Path(str(relative))
            if ".." in path.parts or not re.search(r"(?:^|[._-])(license|licence|copying|copyright|notice|eula)(?:$|[._-])", path.name, re.I):
                continue
            if path.suffix.lower() in {".py", ".pyc", ".pyd", ".dll", ".so"}:
                continue
            source = Path(distribution.locate_file(relative))
            if source.is_file() and source.stat().st_size <= 2 * 1024**2:
                target = licenses / package_name / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        shutil.copyfile(python_license, licenses / "Python-LICENSE.txt")


def package(app_dir, output, part_bytes=1_900_000_000, source_root=None, asset_limit=2**31):
    source_root = Path(source_root or Path(__file__).parent)
    app_dir, output = Path(app_dir), Path(output)
    if not (app_dir / "FriendLLM.exe").is_file():
        raise FileNotFoundError("Build FriendLLM.exe before packaging.")
    add_notices(app_dir, source_root)
    output.mkdir(parents=True, exist_ok=True)
    github_dir = output / "GitHub-Release"
    github_dir.mkdir(exist_ok=True)
    if any(github_dir.iterdir()):
        raise FileExistsError("Move the previous GitHub-Release folder aside before packaging a new release.")
    archive_path = output / "FriendLLM-Windows.zip"
    temporary_archive = output / "FriendLLM-Windows.zip.partial"
    print("Creating the app ZIP…", flush=True)
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for index, path in enumerate(sorted(app_dir.rglob("*"))):
            if path.is_file():
                archive.write(path, (Path("FriendLLM") / path.relative_to(app_dir)).as_posix())
            if index and index % 1500 == 0:
                print(f"Archived {index:,} entries…", flush=True)
    print("Checking ZIP integrity…", flush=True)
    with zipfile.ZipFile(temporary_archive) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failure: {bad}")
    temporary_archive.replace(archive_path)
    source_zip = output / "FriendLLM-Source.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            archive.write(source_root / name, f"FriendLLM_Source/{name}")
    digest = sha256(archive_path)
    manifest = {"filename": archive_path.name, "bytes": archive_path.stat().st_size,
                "sha256": digest, "parts": []}
    if manifest["bytes"] >= asset_limit:
        print("Creating GitHub-sized parts…", flush=True)
        with archive_path.open("rb") as source:
            index = 1
            while source.tell() < manifest["bytes"]:
                path = github_dir / f"{archive_path.name}.part{index:03d}"
                left = part_bytes
                with path.open("wb") as output_part:
                    while left:
                        data = source.read(min(left, 8 * 1024**2))
                        if not data:
                            break
                        output_part.write(data)
                        left -= len(data)
                manifest["parts"].append({"name": path.name, "bytes": path.stat().st_size,
                                          "sha256": sha256(path)})
                index += 1
        (github_dir / "Extract-FriendLLM.ps1").write_text(EXTRACT_SCRIPT, encoding="utf-8")
        (github_dir / "Extract-FriendLLM.bat").write_text(
            '@echo off\ncd /d "%~dp0"\npowershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Extract-FriendLLM.ps1"\n'
            'if errorlevel 1 (\n  pause\n  exit /b 1\n)\npause\n', encoding="ascii")
    else:
        shutil.copyfile(archive_path, github_dir / archive_path.name)
    (github_dir / "release-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copyfile(source_zip, github_dir / source_zip.name)
    source_digest = sha256(source_zip)
    (output / "SHA256SUMS.txt").write_text(
        f"{digest}  {archive_path.name}\n{source_digest}  {source_zip.name}\n", encoding="ascii")
    (github_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(github_dir.iterdir()) if path.is_file()),
        encoding="ascii")
    (output / "GitHub-Release-Notes.md").write_text(
        "# FriendLLM Desktop v0.1.0\n\n"
        "A Windows desktop app for preparing WhatsApp exports and training a local chat-style LoRA adapter.\n\n"
        "## Download and run\n\n"
        + ("Download every numbered ZIP part, release-manifest.json, Extract-FriendLLM.bat and Extract-FriendLLM.ps1 "
           "into one empty folder. Double-click Extract-FriendLLM.bat to verify and extract the app.\n\n"
           if manifest["parts"] else "Download FriendLLM-Windows.zip and extract all files.\n\n")
        + "Run FriendLLM\\FriendLLM.exe. Keep its _internal directory with the executable. Python setup is not required. "
        "The first training run downloads public model weights; personal chats and trained models are not bundled.\n\n"
        "## Features\n\n"
        "- TXT/ZIP import, automatic speaker detection, picker, and usable-example estimates.\n"
        "- One training page with uploads, speaker/output selection, performance, and output activity.\n"
        "- Separate Settings page with context, model, and training options and the recommended Qwen3 4B starter.\n"
        "- Required data-use and consent agreement before starting or resuming training.\n"
        "- NVIDIA acceleration and CPU fallback in this build; other backends require a matching PyTorch build.\n"
        "- Epoch progress, measured ETA, usage graphs, session averages and peaks, and saved performance samples.\n"
        "- Stop/save checkpoint, resume training, and shortcuts to saved files.\n\n"
        "CPU training of the default 4B model requires substantial RAM and can be slow. ETA needs a few measured steps; "
        "loading, downloads and final packaging can add time. Use your own authorized chat exports.\n\n"
        "SHA256SUMS.txt contains release asset hashes. FriendLLM-Source.zip contains the clean source project.\n",
        encoding="utf-8")
    print(json.dumps({"app_zip": str(archive_path), "bytes": manifest["bytes"],
                      "sha256": digest, "github_assets": str(github_dir), "source_zip": str(source_zip)}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, default=Path("dist/FriendLLM"))
    parser.add_argument("--output", type=Path, default=Path("releases"))
    args = parser.parse_args()
    package(args.app_dir, args.output)
