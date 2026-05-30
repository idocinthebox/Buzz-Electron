# Buzz Electron

**Buzz with audio file renaming built in and a beautiful Electron front end.**

Transcribe and translate audio offline on your personal computer, powered by
OpenAI's [Whisper](https://github.com/openai/whisper) — now with a modern
Electron desktop UI and a built-in bulk audio renamer that names files from
their spoken content.

![MIT License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20x64-blue)

> **This is a fork of [Buzz](https://github.com/chidiwilliams/buzz)** by
> Chidi Williams. It keeps Buzz's transcription engine and adds an Electron
> front end, a bulk renamer, and dependency security mitigations. All original
> Buzz functionality and credit belong to the upstream project.

---

## What's new in this fork

- **🖥️ Electron desktop UI** — a modern, themed interface with a tabbed layout
  (**Transcribe** | **Rename**), custom title bar, and a live startup splash
  with a real progress bar fed by the backend.
- **✏️ Bulk audio renamer** — transcribe a folder of audio files and rename
  each one from its spoken content, with preview, collision handling, and a
  one-click undo of the last batch.
- **🎙️ Transcribe tab** — import audio/video files, transcribe with Whisper.cpp /
  Faster Whisper / OpenAI Whisper, view and edit segments, and export to
  TXT / SRT / VTT. Tasks persist across restarts (SQLite).
- **🛡️ CVE mitigations** — 56 dependency advisories across 16 packages cleared
  via safe, non-breaking upgrades. See **[SECURITY.md](SECURITY.md)** for the
  full before/after audit and the honestly-documented deferred items.
- **🔧 Robustness fixes** — GPU/driver transcription crashes are now surfaced
  with a clear "Disable GPU" hint instead of silently producing empty output,
  and the Disable-GPU setting now actually reaches the backend.

## Features

- Transcribe and translate audio & video files (Whisper.cpp, Faster Whisper,
  OpenAI Whisper, Hugging Face models)
- Bulk-rename audio files from their transcribed content
- Editable transcript viewer; export to **TXT, SRT, VTT**
- Task queue with persistence across app restarts
- ~100 languages with auto-detect
- Local model download manager
- Runs fully offline on your own machine

## Install / Build (Windows)

This fork ships as a self-contained Windows build (Electron UI + a bundled
Python backend with Whisper, produced via PyInstaller).

**One-click build:**

```bat
build_renamer.bat
```

This rebuilds the Python backend and packages the Electron app into
`D:\Renamer Electron` (edit `OUTPUT_DIR` at the top of the script to change the
destination). The output `Buzz Renamer-1.0.0-win.zip` is fully portable — unzip
and run `Buzz Renamer.exe`; no installation required.

**Run from source (development):**

```bash
uv sync                       # set up the Python environment
cd renamer-ui && npm install  # Electron dependencies
npm start                     # launches the UI against the venv backend
```

> **Tip:** if transcription produces no output on your machine, the GPU path may
> be crashing on your drivers. Open **Settings → Disable GPU** and restart — the
> app will transcribe on CPU.

## Security

Dependency CVEs are tracked and mitigated. The full report — what was fixed,
what's deferred and why — lives in **[SECURITY.md](SECURITY.md)**. Re-run the
audit anytime with `pip-audit` (Python) and `npm audit` (Electron).

## Credits

- **[Buzz](https://github.com/chidiwilliams/buzz)** by Chidi Williams — the
  upstream transcription application this fork is built on.
- **[Whisper](https://github.com/openai/whisper)** by OpenAI — the underlying
  speech-recognition models.
- **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** by Georgi Gerganov
  — the fast C++ Whisper backend.

## License

[MIT](LICENSE). This fork retains the original Buzz copyright
(© 2022 Chidi Williams) and adds the fork author's copyright for the Electron
UI, renamer, and security work (© 2026 idocinthebox), as required by the MIT
license.
