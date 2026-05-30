"""
buzz/transcriber/renamer_server.py

asyncio WebSocket server that exposes BulkRenamer to the Electron UI.

Usage (from project root, inside the venv):
    python -m buzz.transcriber.renamer_server

On startup the server prints a single line to stdout:
    PORT:<n>
so the Electron main process knows which port to connect to.

Only one WebSocket client is accepted at a time (the Electron renderer).

Protocol
--------
Client -> Server (JSON):
  {"cmd": "start_preview",    "directory": "...", "config": {...}}
  {"cmd": "cancel"}
  {"cmd": "apply_renames",    "folder": "...", "plans": [...]}
  {"cmd": "undo",             "folder": "..."}
  {"cmd": "list_models"}
  {"cmd": "download_model",   "model_type": "...", "model_size": "...",
                               "hugging_face_model_id": ""}
  {"cmd": "cancel_download"}

Server -> Client (JSON):
  {"event": "ready"}
  {"event": "log",              "message": "...", "level": "info|warn|error"}
  {"event": "progress",        "done": N, "total": N, "plan": {...}}
  {"event": "preview_done",    "plans": [...]}
  {"event": "apply_done",      "summary": {...}}
  {"event": "undo_done",       "result": {...}}
  {"event": "models",          "models": [...]}
  {"event": "download_progress","downloaded": N, "total": N, "percent": N}
  {"event": "download_done",   "model_path": "..."}
  {"event": "error",           "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import socket
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _status(pct: int, msg: str) -> None:
    """Emit a startup-progress line the Electron splash parses: STATUS:<pct>:<msg>.

    Guarded so multiprocessing 'spawn' child processes never emit it (they
    re-import heavy modules but must not pollute the parent's stdout protocol).
    """
    if multiprocessing.parent_process() is None:
        print(f"STATUS:{pct}:{msg}", flush=True)


_status(8, "Starting backend")

# ---------------------------------------------------------------------------
# MUST be imported first — sets up CUDA/nvidia DLL paths before torch loads.
# Without this, ctranslate2 (faster-whisper) crashes in subprocesses with
# STATUS_ACCESS_VIOLATION (0xC0000005) on Windows.
# ---------------------------------------------------------------------------
import buzz.cuda_setup  # noqa: F401  # auto-runs setup_cuda_libraries()

from buzz.assets import APP_BASE_DIR  # noqa: E402
from platformdirs import user_cache_dir  # noqa: E402

# Route HuggingFace downloads to the same Buzz cache the GUI uses
_model_root = os.environ.get("BUZZ_MODEL_ROOT")
if _model_root:
    os.environ.setdefault("HF_HOME", os.path.dirname(_model_root))
else:
    os.environ.setdefault("HF_HOME", user_cache_dir("Buzz"))

# Add the buzz package dir to the Windows DLL search path so that
# ctranslate2, faster-whisper, whisper.cpp etc. can find their native libs.
if sys.platform == "win32":
    os.add_dll_directory(APP_BASE_DIR)
    for _sub in ("dll_backup", os.path.join("onnxruntime", "capi")):
        _d = os.path.join(APP_BASE_DIR, _sub)
        if os.path.isdir(_d):
            os.add_dll_directory(_d)

# Add APP_BASE_DIR to PATH so ffmpeg and other bundled binaries are found
# (mirrors what buzz.py does at startup).
os.environ["PATH"] = os.pathsep.join(
    [APP_BASE_DIR, os.path.join(APP_BASE_DIR, "_internal")]
    + [os.environ.get("PATH", "")]
)

_status(22, "Initializing environment")

# ---------------------------------------------------------------------------
# A headless QCoreApplication is required before importing any Qt class.
# WhisperFileTranscriber inherits QObject but its transcribe() method is
# entirely synchronous / multiprocessing-based — no Qt event loop needed.
#
# CRITICAL: Only create the QCoreApplication in the **main** server process.
# On Windows with multiprocessing 'spawn', child processes re-import this
# module. Creating a QCoreApplication (especially with QT_QPA_PLATFORM=
# offscreen) inside the child causes the transcription subprocess to crash
# with STATUS_ACCESS_VIOLATION (0xC0000005).
# ---------------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_status(34, "Loading Qt runtime")

from PyQt6.QtCore import QCoreApplication  # noqa: E402

if multiprocessing.parent_process() is None:
    # Main server process — create the Qt app that BulkRenamer (QObject) needs
    _qt_app: QCoreApplication = (
        QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    )  # type: ignore[assignment]

try:
    import websockets  # noqa: E402
    import websockets.exceptions  # noqa: E402
except ImportError:
    print(
        "ERROR: 'websockets' is not installed.\n"
        "Run:  pip install websockets",
        file=sys.stderr,
    )
    sys.exit(1)

_status(52, "Loading AI models (Whisper)...")

from buzz.model_loader import (  # noqa: E402
    ModelDownloader,
    ModelType,
    TranscriptionModel,
    WhisperModelSize,
)
from buzz.transcriber.bulk_renamer import (  # noqa: E402
    BulkRenamer,
    RenamePlan,
    RenamerConfig,
    apply_plan,
    undo_from_log,
)
from buzz.transcriber.transcriber import (  # noqa: E402
    LANGUAGES,
    Task,
    TranscriptionOptions,
    FileTranscriptionOptions,
    FileTranscriptionTask,
    OutputFormat,
    Segment,
)
from buzz.transcriber.file_transcriber import write_output  # noqa: E402
from buzz.transcriber.whisper_file_transcriber import WhisperFileTranscriber  # noqa: E402

# DB layer (transcription task persistence). Imported in the main process only;
# QSqlDatabase/QSqlQuery are NOT thread-safe, so every DB call in this server
# stays on the asyncio main thread — only the blocking transcribe() runs in an
# executor thread (see _cmd_start_transcription).
from buzz.db.db import setup_app_db  # noqa: E402
from buzz.db.dao.transcription_dao import TranscriptionDAO  # noqa: E402
from buzz.db.dao.transcription_segment_dao import TranscriptionSegmentDAO  # noqa: E402
from buzz.db.service.transcription_service import TranscriptionService  # noqa: E402

_status(85, "Finalizing")

log = logging.getLogger(__name__)

# Set up the SQLite database + service once, in the main process only.
# (multiprocessing 'spawn' children re-import this module but must not touch Qt.)
_db = None
_transcription_service: Optional[TranscriptionService] = None
if multiprocessing.parent_process() is None:
    try:
        _db = setup_app_db()
        _transcription_dao = TranscriptionDAO(_db)
        _transcription_segment_dao = TranscriptionSegmentDAO(_db)
        _transcription_service = TranscriptionService(
            _transcription_dao, _transcription_segment_dao
        )
    except Exception:  # pragma: no cover - defensive
        logging.exception("Failed to initialize transcription database")
        _transcription_service = None
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _plan_to_dict(plan: RenamePlan) -> Dict[str, Any]:
    return {
        "original_path": str(plan.original_path),
        "transcript": plan.transcript,
        "proposed_name": plan.proposed_name,
        "proposed_path": str(plan.proposed_path) if plan.proposed_path else None,
        "status": plan.status,
        "error": plan.error,
        "duration_sec": round(plan.duration_sec, 2),
        "will_change": plan.will_change,
    }


def _plan_from_dict(d: Dict[str, Any]) -> RenamePlan:
    p = RenamePlan(original_path=Path(d["original_path"]))
    p.transcript = d.get("transcript", "")
    p.proposed_name = d.get("proposed_name", "")
    raw_pp = d.get("proposed_path")
    p.proposed_path = Path(raw_pp) if raw_pp else None
    p.status = d.get("status", "pending")
    p.error = d.get("error", "")
    p.duration_sec = float(d.get("duration_sec", 0.0))
    return p


def _build_config(raw: Dict[str, Any]) -> RenamerConfig:
    """Build a RenamerConfig from the JSON config block sent by the UI."""
    model_type_str = raw.get("model_type", ModelType.WHISPER_CPP.value)
    try:
        model_type = ModelType(model_type_str)
    except ValueError:
        model_type = ModelType.WHISPER_CPP

    size_str = raw.get("model_size", WhisperModelSize.BASE.value)
    try:
        size = WhisperModelSize(size_str)
    except ValueError:
        size = WhisperModelSize.BASE

    hf_id = raw.get("hugging_face_model_id", "")
    model = TranscriptionModel(
        model_type=model_type,
        whisper_model_size=size,
        hugging_face_model_id=hf_id,
    )

    language = raw.get("language") or None
    opts = TranscriptionOptions(
        language=language,
        task=Task.TRANSCRIBE,
        model=model,
        word_level_timings=False,
        extract_speech=False,
        initial_prompt=raw.get("initial_prompt", ""),
    )

    # model_path: explicit path provided by the UI (required for whisper.cpp,
    # optional for other backends — they resolve it via get_local_model_path()).
    model_path = raw.get("model_path", "")
    if not model_path:
        model_path = model.get_local_model_path() or ""

    return RenamerConfig(
        transcription_options=opts,
        model_path=model_path,
        trim_seconds=float(raw.get("trim_seconds", 5.0)),
        first_words=int(raw.get("first_words", 6)),
        max_filename_len=int(raw.get("max_filename_len", 50)),
        keep_numeric_prefix=bool(raw.get("keep_numeric_prefix", False)),
        collision_strategy=raw.get("collision_strategy", "suffix"),
    )


def _build_transcription_options(raw: Dict[str, Any]) -> TranscriptionOptions:
    """Build TranscriptionOptions from the UI config block (transcription tab).

    Mirrors _build_config but produces full-transcription options (honours
    word_level_timings, the Translate task, and the chosen language).
    """
    model_type_str = raw.get("model_type", ModelType.WHISPER_CPP.value)
    try:
        model_type = ModelType(model_type_str)
    except ValueError:
        model_type = ModelType.WHISPER_CPP

    size_str = raw.get("model_size", WhisperModelSize.BASE.value)
    try:
        size = WhisperModelSize(size_str)
    except ValueError:
        size = WhisperModelSize.BASE

    model = TranscriptionModel(
        model_type=model_type,
        whisper_model_size=size,
        hugging_face_model_id=raw.get("hugging_face_model_id", ""),
    )

    task_str = raw.get("task", Task.TRANSCRIBE.value)
    try:
        task = Task(task_str)
    except ValueError:
        task = Task.TRANSCRIBE

    return TranscriptionOptions(
        language=raw.get("language") or None,
        task=task,
        model=model,
        word_level_timings=bool(raw.get("word_level_timings", False)),
        extract_speech=False,
        initial_prompt=raw.get("initial_prompt", ""),
    )


def _transcription_row_to_dict(row) -> Dict[str, Any]:
    """Serialize a Transcription DB entity (or QSqlRecord-derived dict) for the UI."""
    return {
        "id": row.id,
        "file": row.file,
        "name": row.name or (os.path.basename(row.file) if row.file else ""),
        "status": row.status,
        "task": row.task,
        "model_type": row.model_type,
        "model_size": row.whisper_model_size,
        "language": row.language,
        "progress": float(row.progress or 0.0),
        "error": row.error_message,
        "time_queued": row.time_queued,
        "time_started": row.time_started,
        "time_ended": row.time_ended,
    }


# ---------------------------------------------------------------------------
# Session: one per connected Electron client
# ---------------------------------------------------------------------------

class _Session:
    """Manages a single connected WebSocket client."""

    def __init__(self, ws) -> None:
        self._ws = ws
        self._loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._cancel_event = threading.Event()

    # ---- thread-safe helpers ------------------------------------------------

    async def _send(self, event: Dict[str, Any]) -> None:
        try:
            await self._ws.send(json.dumps(event))
        except websockets.exceptions.ConnectionClosed:
            pass

    def _post(self, event: Dict[str, Any]) -> None:
        """Safe to call from worker threads."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def _drain_until(self, stop_event: str) -> None:
        """Forward queued events to the WebSocket until `stop_event` arrives."""
        while True:
            ev = await self._queue.get()
            await self._send(ev)
            if ev.get("event") == stop_event:
                break

    # ---- command handlers ---------------------------------------------------

    async def _cmd_list_models(self, _msg: Dict) -> None:
        models = []
        for mt in ModelType:
            if not mt.is_available():
                continue
            sizes = []
            if mt in (ModelType.WHISPER, ModelType.WHISPER_CPP,
                      ModelType.FASTER_WHISPER):
                for sz in WhisperModelSize:
                    try:
                        local = TranscriptionModel(
                            model_type=mt, whisper_model_size=sz
                        ).get_local_model_path()
                    except (KeyError, Exception):
                        # Some WhisperModelSize values (e.g. 'lumii') are
                        # Buzz-specific and not in openai-whisper's _MODELS.
                        local = None
                    sizes.append({
                        "size": sz.value,
                        "label": str(sz),
                        "downloaded": local is not None,
                    })
            models.append({
                "type": mt.value,
                "sizes": sizes,
                "needs_path": mt == ModelType.WHISPER_CPP,
            })
        await self._send({"event": "models", "models": models})

    async def _cmd_list_gpus(self, _msg: Dict) -> None:
        """Enumerate the GPUs whisper.cpp sees, for the UI device picker.

        IMPORTANT: we parse whisper-cli's own ``ggml_vulkan: N = <name>`` init
        lines rather than the Python ``vulkan`` package, because the two
        enumerate devices in *different order* — only whisper.cpp's order
        matches the ``--device N`` flag. We run it with ``--no-gpu`` so the
        Vulkan devices are still enumerated at init but compute stays on CPU,
        avoiding the integrated-GPU crash during this probe.
        """
        gpus = await self._loop.run_in_executor(None, self._enumerate_gpus)
        await self._send({"event": "gpus", "gpus": gpus})

    @staticmethod
    def _enumerate_gpus() -> List[Dict[str, Any]]:
        """Return GPUs in whisper.cpp's --device order.

        Strategy: run whisper-cli with --no-gpu against a tiny synthesized
        silent WAV. That makes ggml_vulkan enumerate the Vulkan devices at init
        (printing the authoritative ``ggml_vulkan: N = <name>`` order that
        matches --device N) while computing on CPU, so the probe can't trigger
        the integrated-GPU crash. Requires a local whisper.cpp model; if none
        is available the picker stays on "Auto" (empty list).
        """
        import re as _re
        import struct
        import subprocess
        import tempfile

        cli = "whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"
        cli_path = os.path.join(APP_BASE_DIR, "whisper_cpp", cli)
        if not os.path.exists(cli_path):
            cli_path = os.path.join(APP_BASE_DIR, "buzz", "whisper_cpp", cli)
        if not os.path.exists(cli_path):
            return []

        # Need any local whisper.cpp model to make the CLI init the backend.
        try:
            model_path = TranscriptionModel(
                model_type=ModelType.WHISPER_CPP,
                whisper_model_size=WhisperModelSize.TINY,
            ).get_local_model_path()
        except Exception:
            model_path = None
        if not model_path or not os.path.exists(model_path):
            return []

        # Synthesize a tiny silent 16 kHz mono WAV (0.1 s) to feed the probe.
        tmp = os.path.join(tempfile.gettempdir(), "buzz_gpuprobe.wav")
        try:
            n = 1600  # 0.1 s @ 16 kHz
            data = b"\x00\x00" * n
            with open(tmp, "wb") as f:
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + len(data)))
                f.write(b"WAVEfmt ")
                f.write(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
                f.write(b"data")
                f.write(struct.pack("<I", len(data)))
                f.write(data)
        except Exception as exc:  # pragma: no cover - defensive
            log.info("GPU probe WAV write failed: %s", exc)
            return []

        kwargs: Dict[str, Any] = dict(capture_output=True, text=True, timeout=60)
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = si
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            proc = subprocess.run(
                [cli_path, "--no-gpu", "--model", model_path, "-f", tmp],
                **kwargs,
            )
            out = (proc.stderr or "") + (proc.stdout or "")
        except Exception as exc:  # pragma: no cover - defensive
            log.info("GPU enumeration failed: %s", exc)
            return []
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

        gpus: List[Dict[str, Any]] = []
        for m in _re.finditer(
            r"ggml_vulkan:\s*(\d+)\s*=\s*(.+?)\s*\|", out
        ):
            idx = int(m.group(1))
            # name is like "AMD Radeon(TM) 890M Graphics (AMD proprietary driver)"
            name = _re.sub(r"\s*\([^)]*\)\s*$", "", m.group(2)).strip()
            uma = bool(_re.search(rf"ggml_vulkan:\s*{idx}\s*=.*?uma:\s*1", out))
            gpus.append({
                "index": idx,
                "name": name,
                # uma:1 ⇒ unified-memory (integrated) GPU.
                "type": "integrated" if uma else "discrete",
            })
        return gpus

    async def _cmd_list_languages(self, _msg: Dict) -> None:
        """Return all Whisper-supported languages sorted alphabetically by name."""
        langs = sorted(
            [{"code": code, "name": name} for code, name in LANGUAGES.items()],
            key=lambda x: x["name"].lower(),
        )
        await self._send({"event": "languages", "languages": langs})

    async def _cmd_list_files(self, msg: Dict) -> None:
        """Return the list of audio files in a folder (no transcription)."""
        directory = Path(msg.get("directory", ""))
        if not directory.is_dir():
            await self._send({"event": "error",
                               "message": f"Not a directory: {directory}"})
            return
        # Scan directly — do NOT instantiate BulkRenamer (a QObject) here.
        # Creating a QObject outside the Qt event loop can cause instability.
        AUDIO_EXTENSIONS = (
            ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus",
        )
        found: List[Path] = []
        for ext in AUDIO_EXTENSIONS:
            found.extend(directory.glob(f"*{ext}"))
            found.extend(directory.glob(f"*{ext.upper()}"))
        files = sorted(set(found))
        await self._send({
            "event": "files_listed",
            "files": [str(f) for f in files],
        })

    async def _cmd_start_preview(self, msg: Dict) -> None:
        directory = Path(msg["directory"])
        if not directory.is_dir():
            await self._send({"event": "error",
                               "message": f"Not a directory: {directory}"})
            return

        try:
            cfg = _build_config(msg.get("config", {}))
        except Exception as exc:
            await self._send({"event": "error",
                               "message": f"Config error: {exc}"})
            return

        if not cfg.model_path:
            await self._send({
                "event": "error",
                "message": (
                    "No model found locally. "
                    "Please download a model first or provide a model file path."
                ),
            })
            return

        self._cancel_event.clear()
        renamer = BulkRenamer(cfg)
        files = renamer.find_audio_files(directory)
        total = len(files)

        if total == 0:
            await self._send({"event": "log",
                               "message": "No audio files found in that folder.",
                               "level": "warn"})
            await self._send({"event": "preview_done", "plans": []})
            return

        await self._send({"event": "log",
                           "message": f"Found {total} audio file(s).",
                           "level": "info"})

        plans: List[RenamePlan] = []

        def _worker() -> None:
            with tempfile.TemporaryDirectory(prefix="buzz_rename_") as td:
                tmp_dir = Path(td)
                for i, path in enumerate(files, start=1):
                    if self._cancel_event.is_set():
                        self._post({"event": "log",
                                    "message": "Cancelled by user.",
                                    "level": "warn"})
                        # Fill remaining as skipped
                        for remaining in files[i - 1:]:
                            plans.append(RenamePlan(
                                original_path=remaining,
                                status="skipped",
                                error="cancelled",
                            ))
                        break

                    plan = renamer._process_one(path, tmp_dir)
                    plans.append(plan)
                    self._post({
                        "event": "progress",
                        "done": i,
                        "total": total,
                        "plan": _plan_to_dict(plan),
                    })
                    level = "info" if plan.status == "ready" else "error"
                    if plan.status == "ready":
                        msg_text = (
                            f"  {path.name} → "
                            f"{plan.proposed_name}{path.suffix}"
                        )
                    else:
                        msg_text = f"  {path.name}: {plan.error}"
                    self._post({"event": "log",
                                "message": msg_text,
                                "level": level})

            # Resolve cross-batch collisions
            renamer._resolve_collisions(plans)
            self._post({
                "event": "preview_done",
                "plans": [_plan_to_dict(p) for p in plans],
            })

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        await self._drain_until("preview_done")

    async def _cmd_cancel(self, _msg: Dict) -> None:
        self._cancel_event.set()
        await self._send({"event": "log",
                           "message": "Cancellation requested.",
                           "level": "warn"})

    async def _cmd_apply_renames(self, msg: Dict) -> None:
        raw_plans = msg.get("plans", [])
        plans = [_plan_from_dict(d) for d in raw_plans]
        if not plans:
            await self._send({"event": "error",
                               "message": "No plans provided."})
            return

        folder_str = msg.get("folder")
        folder = (
            Path(folder_str)
            if folder_str
            else plans[0].original_path.parent
        )
        log_path = (
            folder / f".undo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        try:
            summary = apply_plan(plans, log_path)
        except Exception as exc:
            await self._send({"event": "error",
                               "message": f"Apply failed: {exc}"})
            return

        await self._send({
            "event": "apply_done",
            "summary": {
                "applied_count": summary["applied_count"],
                "skipped_count": summary["skipped_count"],
                "error_count": summary["error_count"],
            },
        })

    async def _cmd_undo(self, msg: Dict) -> None:
        folder = Path(msg.get("folder", "."))
        logs = sorted(folder.glob(".undo_*.json"), reverse=True)
        if not logs:
            await self._send({"event": "error",
                               "message": "No undo log found in this folder."})
            return
        try:
            result = undo_from_log(logs[0])
        except Exception as exc:
            await self._send({"event": "error",
                               "message": f"Undo failed: {exc}"})
            return

        await self._send({
            "event": "undo_done",
            "result": {
                "reverted_count": result["reverted_count"],
                "failed_count": result["failed_count"],
                "log_name": logs[0].name,
            },
        })

    # ---- download handlers --------------------------------------------------

    async def _cmd_download_model(self, msg: Dict) -> None:
        """Download a model using Buzz's ModelLoader, streaming progress."""
        model_type_str = msg.get("model_type", ModelType.WHISPER_CPP.value)
        model_size_str = msg.get("model_size", WhisperModelSize.BASE.value)
        hf_id = msg.get("hugging_face_model_id", "")

        try:
            model_type = ModelType(model_type_str)
        except ValueError:
            await self._send({"event": "error",
                               "message": f"Unknown model type: {model_type_str}"})
            return

        try:
            size = WhisperModelSize(model_size_str)
        except ValueError:
            size = WhisperModelSize.BASE

        model = TranscriptionModel(
            model_type=model_type,
            whisper_model_size=size,
            hugging_face_model_id=hf_id,
        )

        loop = asyncio.get_event_loop()
        done_event = asyncio.Event()
        result: Dict = {}

        def _on_progress(progress):
            # progress is a tuple (downloaded_bytes, total_bytes)
            downloaded, total = progress
            pct = round(downloaded / total * 100, 1) if total else 0
            loop.call_soon_threadsafe(
                self._queue.put_nowait,
                {"event": "download_progress",
                 "downloaded": downloaded,
                 "total": total,
                 "percent": pct},
            )

        def _on_finished(path: str):
            result["path"] = path
            loop.call_soon_threadsafe(
                self._queue.put_nowait,
                {"event": "download_done", "model_path": path},
            )
            loop.call_soon_threadsafe(done_event.set)

        def _on_error(err: str):
            result["error"] = err
            loop.call_soon_threadsafe(
                self._queue.put_nowait,
                {"event": "error", "message": f"Download failed: {err}"},
            )
            loop.call_soon_threadsafe(done_event.set)

        loader = ModelDownloader(model=model)
        loader.signals.progress.connect(_on_progress)
        loader.signals.finished.connect(_on_finished)
        loader.signals.error.connect(_on_error)

        self._active_loader = loader

        await self._send({"event": "log",
                           "message": f"Starting download: {model_type_str} / {model_size_str}",
                           "level": "info"})

        thread = threading.Thread(target=loader.run, daemon=True)
        thread.start()

        # Drain progress/done events until download completes.
        # HuggingFace downloads (Faster Whisper, Whisper.cpp) run in a subprocess
        # and emit only one (0,100) progress signal — so we send periodic heartbeats
        # to keep the UI alive and informed.
        import time as _time
        last_heartbeat = _time.monotonic()
        heartbeat_interval = 5  # seconds
        elapsed_ticks = 0

        while not done_event.is_set():
            try:
                ev = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                await self._send(ev)
            except asyncio.TimeoutError:
                now = _time.monotonic()
                if now - last_heartbeat >= heartbeat_interval:
                    elapsed_ticks += heartbeat_interval
                    mins = elapsed_ticks // 60
                    secs = elapsed_ticks % 60
                    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
                    await self._send({
                        "event": "download_progress",
                        "downloaded": 0,
                        "total": 0,   # 0/0 signals indeterminate to the UI
                        "percent": -1,
                        "elapsed": elapsed_str,
                    })
                    last_heartbeat = now
                continue

        self._active_loader = None

    async def _cmd_cancel_download(self, _msg: Dict) -> None:
        loader = getattr(self, "_active_loader", None)
        if loader is not None:
            loader.stopped = True
            await self._send({"event": "log",
                               "message": "Download cancelled.",
                               "level": "warn"})
        else:
            await self._send({"event": "log",
                               "message": "No active download to cancel.",
                               "level": "warn"})

    # ---- transcription handlers --------------------------------------------
    # All DB access below runs on the asyncio main thread (QSqlDatabase is not
    # thread-safe). Only WhisperFileTranscriber.transcribe() — pure
    # multiprocessing, no Qt — is offloaded to an executor thread.

    async def _cmd_import_files(self, msg: Dict) -> None:
        """Create QUEUED transcription rows for the given file paths."""
        if _transcription_service is None:
            await self._send({"event": "error",
                               "message": "Transcription database unavailable."})
            return

        paths = [p for p in msg.get("files", []) if p]
        if not paths:
            await self._send({"event": "error", "message": "No files provided."})
            return

        options = _build_transcription_options(msg.get("config", {}))
        export_formats = set()
        for fmt in msg.get("export_formats", []):
            try:
                export_formats.add(OutputFormat(fmt))
            except ValueError:
                pass

        created = []
        for path in paths:
            if not os.path.isfile(path):
                await self._send({"event": "log",
                                   "message": f"Skipped (not a file): {path}",
                                   "level": "warn"})
                continue
            task = FileTranscriptionTask(
                transcription_options=options,
                file_transcription_options=FileTranscriptionOptions(
                    file_paths=[path], output_formats=export_formats
                ),
                model_path="",
                file_path=path,
                source=FileTranscriptionTask.Source.FILE_IMPORT,
                status=FileTranscriptionTask.Status.QUEUED,
            )
            _transcription_service.create_transcription(task)
            row = _transcription_dao.find_by_id(str(task.uid))
            if row is not None:
                created.append(_transcription_row_to_dict(row))

        await self._send({"event": "files_imported", "tasks": created})

    async def _cmd_get_tasks(self, _msg: Dict) -> None:
        await self._send({"event": "tasks", "tasks": self._list_tasks()})

    def _list_tasks(self) -> List[Dict[str, Any]]:
        """Read every transcription row (most-recent first). Main thread only."""
        if _transcription_service is None:
            return []
        from PyQt6.QtSql import QSqlQuery
        query = QSqlQuery(_db)
        query.prepare("SELECT * FROM transcription ORDER BY time_queued DESC")
        rows: List[Dict[str, Any]] = []
        if query.exec():
            while query.next():
                rec = query.record()
                row = _transcription_dao.to_entity(rec)
                rows.append(_transcription_row_to_dict(row))
        return rows

    async def _cmd_start_transcription(self, msg: Dict) -> None:
        """Transcribe a queued task; stream progress; persist segments."""
        if _transcription_service is None:
            await self._send({"event": "error",
                               "message": "Transcription database unavailable."})
            return

        task_id = msg.get("id")
        row = _transcription_dao.find_by_id(str(task_id)) if task_id else None
        if row is None:
            await self._send({"event": "error",
                               "message": f"Task not found: {task_id}"})
            return

        uid = row.id_as_uuid
        options = _build_transcription_options({
            "model_type": row.model_type,
            "model_size": row.whisper_model_size,
            "language": row.language,
            "task": row.task,
            "hugging_face_model_id": row.hugging_face_model_id or "",
            "word_level_timings": str(row.word_level_timings).lower() == "true",
        })
        model_path = options.model.get_local_model_path() or ""
        if not model_path and options.model.model_type == ModelType.WHISPER_CPP:
            await self._send({"event": "error",
                               "message": "No local model found. Download a model first."})
            return

        task = FileTranscriptionTask(
            transcription_options=options,
            file_transcription_options=FileTranscriptionOptions(
                file_paths=[row.file]
            ),
            model_path=model_path,
            file_path=row.file,
            uid=uid,
        )

        self._cancel_event.clear()
        _transcription_service.update_transcription_as_started(uid)
        await self._send({"event": "task_started", "id": str(uid)})

        transcriber = WhisperFileTranscriber(task=task)
        self._active_transcriber = transcriber

        # Run the blocking transcribe() off the event loop; it manages its own
        # multiprocessing child + read thread internally.
        try:
            segments: List[Segment] = await self._loop.run_in_executor(
                None, transcriber.transcribe
            )
        except Exception as exc:  # transcription failed or canceled
            self._active_transcriber = None
            err = str(exc)
            if "cancel" in err.lower():
                _transcription_service.update_transcription_as_canceled(uid)
                await self._send({"event": "task_canceled", "id": str(uid)})
            else:
                _transcription_service.update_transcription_as_failed(uid, err)
                await self._send({"event": "task_error",
                                   "id": str(uid), "message": err})
            return

        self._active_transcriber = None

        # The whisper-cli child can crash (e.g. a GPU/Vulkan fault, Windows exit
        # 0xC0000409) yet still exit the *Python* child cleanly — transcribe()
        # then returns [] without raising. Detect that here so the user gets a
        # real error (with a fix hint) instead of a silently empty transcript.
        child_error = getattr(transcriber, "error_message", None)
        if child_error:
            _transcription_service.update_transcription_as_failed(uid, child_error)
            hint = ""
            if not os.getenv("BUZZ_FORCE_CPU", "false").lower() == "true":
                hint = (" — if this is a GPU/driver crash, enable "
                        "“Disable GPU” in Settings and retry.")
            await self._send({"event": "task_error", "id": str(uid),
                               "message": f"{child_error}{hint}"})
            return

        # Persist segments + mark completed (main thread).
        _transcription_service.update_transcription_as_completed(uid, segments)

        await self._send({
            "event": "task_completed",
            "id": str(uid),
            "segment_count": len(segments),
        })

    async def _cmd_cancel_transcription(self, _msg: Dict) -> None:
        self._cancel_event.set()
        transcriber = getattr(self, "_active_transcriber", None)
        if transcriber is not None:
            transcriber.stop()
            await self._send({"event": "log",
                               "message": "Transcription cancellation requested.",
                               "level": "warn"})

    async def _cmd_get_segments(self, msg: Dict) -> None:
        if _transcription_service is None:
            await self._send({"event": "error",
                               "message": "Transcription database unavailable."})
            return
        task_id = msg.get("id")
        segments = _transcription_service.get_transcription_segments(task_id)
        await self._send({
            "event": "segments",
            "id": str(task_id),
            "segments": [
                {"id": s.id, "start": s.start_time, "end": s.end_time,
                 "text": s.text, "translation": s.translation}
                for s in segments
            ],
        })

    async def _cmd_update_segment(self, msg: Dict) -> None:
        """Edit a segment's text by rewriting that transcription's segments."""
        if _transcription_service is None:
            return
        task_id = msg.get("id")
        seg_id = msg.get("segment_id")
        new_text = msg.get("text", "")
        existing = _transcription_service.get_transcription_segments(task_id)
        rebuilt = [
            Segment(
                start=s.start_time,
                end=s.end_time,
                text=new_text if s.id == seg_id else s.text,
            )
            for s in existing
        ]
        _transcription_service.replace_transcription_segments(task_id, rebuilt)
        await self._send({"event": "segment_updated",
                          "id": str(task_id), "segment_id": seg_id})

    async def _cmd_export_transcript(self, msg: Dict) -> None:
        """Write a completed transcript to TXT/SRT/VTT and return the path."""
        if _transcription_service is None:
            return
        task_id = msg.get("id")
        fmt_str = msg.get("format", "srt")
        out_path = msg.get("output_path")  # provided by the renderer's save dialog
        try:
            output_format = OutputFormat(fmt_str)
        except ValueError:
            await self._send({"event": "error",
                               "message": f"Unknown export format: {fmt_str}"})
            return

        db_segments = _transcription_service.get_transcription_segments(task_id)
        if not db_segments:
            await self._send({"event": "error",
                               "message": "No segments to export."})
            return
        segments = [
            Segment(start=s.start_time, end=s.end_time, text=s.text,
                    translation=s.translation or "")
            for s in db_segments
        ]

        if not out_path:
            row = _transcription_dao.find_by_id(str(task_id))
            out_path = row.get_output_file_path(output_format) if row else None
        if not out_path:
            await self._send({"event": "error",
                               "message": "No output path for export."})
            return

        try:
            write_output(out_path, segments, output_format)
        except Exception as exc:
            await self._send({"event": "error",
                               "message": f"Export failed: {exc}"})
            return

        await self._send({"event": "export_done",
                          "id": str(task_id), "path": out_path})

    async def _cmd_delete_task(self, msg: Dict) -> None:
        if _transcription_service is None:
            return
        task_id = msg.get("id")
        from PyQt6.QtSql import QSqlQuery
        # Segments are removed via ON DELETE CASCADE (foreign_keys = ON).
        query = QSqlQuery(_db)
        query.prepare("DELETE FROM transcription WHERE id = :id")
        query.bindValue(":id", str(task_id))
        query.exec()
        await self._send({"event": "task_deleted", "id": str(task_id)})

    # ---- main loop ----------------------------------------------------------

    async def run(self) -> None:
        self._active_loader = None
        self._active_transcriber = None
        await self._send({"event": "ready"})
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            cmd = msg.get("cmd")
            log.debug("Received command: %s", cmd)

            if cmd == "list_models":
                await self._cmd_list_models(msg)
            elif cmd == "list_files":
                await self._cmd_list_files(msg)
            elif cmd == "start_preview":
                await self._cmd_start_preview(msg)
            elif cmd == "cancel":
                await self._cmd_cancel(msg)
            elif cmd == "apply_renames":
                await self._cmd_apply_renames(msg)
            elif cmd == "undo":
                await self._cmd_undo(msg)
            elif cmd == "download_model":
                await self._cmd_download_model(msg)
            elif cmd == "cancel_download":
                await self._cmd_cancel_download(msg)
            elif cmd == "list_languages":
                await self._cmd_list_languages(msg)
            elif cmd == "list_gpus":
                await self._cmd_list_gpus(msg)
            # ---- transcription commands ----
            elif cmd == "import_files":
                await self._cmd_import_files(msg)
            elif cmd == "get_tasks":
                await self._cmd_get_tasks(msg)
            elif cmd == "start_transcription":
                await self._cmd_start_transcription(msg)
            elif cmd == "cancel_transcription":
                await self._cmd_cancel_transcription(msg)
            elif cmd == "get_segments":
                await self._cmd_get_segments(msg)
            elif cmd == "update_segment":
                await self._cmd_update_segment(msg)
            elif cmd == "export_transcript":
                await self._cmd_export_transcript(msg)
            elif cmd == "delete_task":
                await self._cmd_delete_task(msg)
            else:
                log.warning("Unknown command: %s", cmd)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    # Pick a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    async def _handler(ws):
        session = _Session(ws)
        try:
            await session.run()
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            log.exception("Unhandled error in session")

    _status(95, "Starting server")

    async with websockets.serve(_handler, "127.0.0.1", port):
        # Tell the parent process (Electron main.js) which port we chose
        print(f"PORT:{port}", flush=True)
        log.info("Renamer server ready on ws://127.0.0.1:%d", port)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    # Required on Windows: prevents multiprocessing subprocesses from
    # re-running this startup code when they re-import the main module.
    multiprocessing.freeze_support()
    asyncio.run(_main())
