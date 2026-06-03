import subprocess
import time
from csv import DictWriter
from datetime import datetime
from pathlib import Path
from settings import apply_cli_overrides, build_settings, load_dotenv_from_project_root

load_dotenv_from_project_root()
apply_cli_overrides()

from nao_common import (
    NAO_IP,
    NAO_PORT,
    nao_say,
    tts_clip,
)
from nao_service import run_nao_record_and_transcribe
from rag_service import answer_query, init_rag_engine
from speech_to_text import (
    ContinuousVADCapture,
    ContinuousVADConfig,
    WHISPER_AVAILABLE,
    VOSK_AVAILABLE,
    configure_input_device_from_env,
    load_whisper_model,
    load_vosk_model,
    resolve_input_device_from_env,
    transcribe_pcm16_with_whisper,
    transcribe_push_to_talk,
)


def _append_stt_debug_row(settings, *, mode: str, stt_text: str, rag_answer: str = "", rag_error: str = "") -> None:
    if not settings.stt_debug_log_enabled:
        return

    log_path = Path(settings.stt_debug_log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with log_path.open("a", encoding="utf-8", newline="") as fp:
            writer = DictWriter(
                fp,
                fieldnames=[
                    "timestamp",
                    "mode",
                    "stt_text",
                    "rag_answer",
                    "rag_error",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "mode": mode,
                    "stt_text": stt_text,
                    "rag_answer": rag_answer,
                    "rag_error": rag_error,
                }
            )
    except Exception as e:
        print(f"[WARN] STT debug log write failed: {e}")


def _append_vad_metrics_row(settings, *, state: str, event: str, metrics: dict) -> None:
    if not settings.stt_debug_log_enabled:
        return

    log_path = Path(settings.stt_debug_log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with log_path.open("a", encoding="utf-8", newline="") as fp:
            writer = DictWriter(
                fp,
                fieldnames=[
                    "timestamp",
                    "mode",
                    "stt_text",
                    "rag_answer",
                    "rag_error",
                    "state",
                    "event",
                    "segment_ms",
                    "speech_ms",
                    "native_sr",
                    "target_sr",
                    "noise_floor_db",
                    "end_reason",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "mode": "vc",
                    "stt_text": "",
                    "rag_answer": "",
                    "rag_error": "",
                    "state": state,
                    "event": event,
                    "segment_ms": metrics.get("segment_ms", ""),
                    "speech_ms": metrics.get("speech_ms", ""),
                    "native_sr": metrics.get("native_sr", ""),
                    "target_sr": metrics.get("target_sr", ""),
                    "noise_floor_db": metrics.get("noise_floor_db", ""),
                    "end_reason": metrics.get("end_reason", ""),
                }
            )
    except Exception as e:
        print(f"[WARN] VAD metrics log write failed: {e}")


def _run_continuous_voice_mode(settings, rag_module, rag_engine, whisper_model) -> None:
    if whisper_model is None:
        print("[WARN] Continuous mode χρειάζεται Whisper STT αλλά δεν είναι διαθέσιμο.")
        return

    try:
        device_index = resolve_input_device_from_env(strict=True)
    except Exception as e:
        print(f"[WARN] Invalid input device configuration: {e}")
        print("[WARN] Ρύθμισε σωστά το SOUNDDEVICE_DEVICE για να αποφευχθεί fallback σε λάθος μικρόφωνο.")
        return

    vad_config = ContinuousVADConfig(
        frame_ms=settings.continuous_vad_frame_ms,
        preroll_ms=settings.continuous_vad_preroll_ms,
        min_speech_ms=settings.continuous_vad_min_speech_ms,
        silence_ms=settings.continuous_vad_silence_ms,
        max_segment_seconds=settings.continuous_vad_max_segment_seconds,
        start_trigger_frames=settings.continuous_vad_start_trigger_frames,
        start_threshold_db=float(settings.continuous_vad_start_threshold_db),
        stop_threshold_db=float(settings.continuous_vad_stop_threshold_db),
        calibration_seconds=float(settings.continuous_vad_calibration_seconds),
        target_samplerate=16000,
    )

    state = "LISTENING"
    print("Continuous conversation mode ενεργό.")
    print("Καταστάσεις: LISTENING -> PROCESSING -> SPEAKING")
    print("Πάτα Ctrl+C για έξοδο από continuous mode.")

    try:
        with ContinuousVADCapture(vad_config, device_index=device_index) as vad_capture:
            while True:
                state = "LISTENING"
                segment = vad_capture.next_segment()
                metrics = segment.get("metrics", {}) if isinstance(segment, dict) else {}
                _append_vad_metrics_row(settings, state=state, event="segment_detected", metrics=metrics)

                if segment.get("discarded"):
                    _append_vad_metrics_row(settings, state=state, event="segment_discarded", metrics=metrics)
                    continue

                state = "PROCESSING"
                user_text = transcribe_pcm16_with_whisper(
                    whisper_model,
                    segment.get("audio_pcm16", b""),
                    samplerate=int(segment.get("samplerate", 16000)),
                    language=settings.stt_language,
                )

                if not user_text:
                    print("[WARN] Whisper επέστρεψε κενό αποτέλεσμα.")
                    _append_vad_metrics_row(settings, state=state, event="whisper_empty", metrics=metrics)
                    continue

                print("Καταγραφή:", user_text)
                rag_result = answer_query(rag_engine, user_text)

                if settings.rag_debug_chunks and rag_module is not None:
                    for i, item in enumerate(rag_result.get("chunks", []), start=1):
                        print(rag_module.format_chunk(item, i))

                reply = (rag_result.get("answer") or "").strip()
                rag_error = rag_result.get("error") or ""
                if not reply:
                    if rag_error:
                        print(f"[WARN] {rag_error}")
                    reply = settings.rag_fallback_reply

                _append_stt_debug_row(
                    settings,
                    mode="vc",
                    stt_text=user_text,
                    rag_answer=reply,
                    rag_error=rag_error,
                )

                print("NAO:", reply)

                state = "SPEAKING"
                vad_capture.set_capture_enabled(False)
                try:
                    nao_say(tts_clip(reply))
                except subprocess.TimeoutExpired:
                    print("[WARN] NAO timeout")
                except Exception as e:
                    print(f"[WARN] NAO error: {e}")
                finally:
                    tail_sec = max(0.0, settings.continuous_speaking_tail_ms / 1000.0)
                    if tail_sec > 0:
                        time.sleep(tail_sec)
                    vad_capture.set_capture_enabled(True)
    except KeyboardInterrupt:
        print("\n[INFO] Continuous mode stopped by user.")
    except Exception as e:
        print(f"[WARN] Continuous mode failed: {e}")


def main():
    settings = build_settings()
    rag_module, rag_engine = init_rag_engine(
        settings.rag_model,
        settings.rag_top_k,
        preload_retrieval_on_boot=settings.rag_preload_retrieval_on_boot,
    )

    print(f"NAO: {NAO_IP}:{NAO_PORT}")
    print(f"RAG model={settings.rag_model}")
    print(f"RAG preload retrieval on boot={settings.rag_preload_retrieval_on_boot}")
    print(f"RAG debug chunks={settings.rag_debug_chunks}")
    print(f"STT whisper model={settings.stt_whisper_model}")
    print(f"STT language={settings.stt_language}")
    print(f"STT debug log enabled={settings.stt_debug_log_enabled}")
    if settings.stt_debug_log_enabled:
        print(f"STT debug log path={settings.stt_debug_log_path}")

    whisper_available = WHISPER_AVAILABLE
    whisper_model = None
    if whisper_available:
        whisper_model = load_whisper_model(settings.stt_whisper_model)
        if whisper_model is None:
            whisper_available = False

    vosk_available = VOSK_AVAILABLE
    vosk_model = None
    if vosk_available:
        vosk_model = load_vosk_model(settings.vosk_model_path)
        if vosk_model is None:
            vosk_available = False

    if whisper_available:
        configure_input_device_from_env()

    print(
        "Επιλογές εισόδου: "
        "(k) πληκτρολόγιο / (v) φωνή / (vs) φωνή μόνο STT (debug) / "
        "(vc) συνεχής φωνητικός διάλογος (VAD) / (n) NAO (test) / (q) έξοδος"
    )

    while True:
        try:
            mode = input("Επιλογή > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if not mode or mode == "q":
            break

        if mode == "k":
            try:
                user_text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_text:
                continue
        elif mode in {"v", "vs"}:
            if not whisper_available or whisper_model is None:
                print("[WARN] Φωνητική είσοδος μη διαθέσιμη — επίλεξε πληκτρολόγιο.")
                continue
            try:
                user_text = transcribe_push_to_talk(whisper_model, max_seconds=settings.push_max_seconds)
            except Exception as e:
                print(f"[WARN] STT error: {e}")
                continue
            if not user_text:
                print("[WARN] Δεν ανιχνεύτηκε κείμενο στην εγγραφή.")
                continue
            print("Καταγραφή:", user_text)
            if mode == "vs":
                _append_stt_debug_row(settings, mode=mode, stt_text=user_text)
                print("[DEBUG] STT-only mode: παράλειψη RAG/AI απάντησης.")
                continue
        elif mode == "vc":
            _run_continuous_voice_mode(settings, rag_module, rag_engine, whisper_model)
            continue
        elif mode == "n":
            if not whisper_available or whisper_model is None:
                print("[WARN] NAO mode χρειάζεται Whisper STT αλλά δεν είναι διαθέσιμο.")
                continue
            user_text = run_nao_record_and_transcribe(whisper_model, NAO_IP, NAO_PORT)
            if not user_text:
                continue
        else:
            print("Άγνωστη επιλογή — πάτα k, v, vs, vc, n ή q")
            continue

        rag_result = answer_query(rag_engine, user_text)

        if settings.rag_debug_chunks and rag_module is not None:
            for i, item in enumerate(rag_result.get("chunks", []), start=1):
                print(rag_module.format_chunk(item, i))

        reply = (rag_result.get("answer") or "").strip()
        rag_error = rag_result.get("error") or ""
        if not reply:
            if rag_error:
                print(f"[WARN] {rag_error}")
            reply = settings.rag_fallback_reply

        if mode in {"v", "vs"}:
            _append_stt_debug_row(
                settings,
                mode=mode,
                stt_text=user_text,
                rag_answer=reply,
                rag_error=rag_error,
            )

        print("NAO:", reply)

        try:
            nao_say(tts_clip(reply))
        except subprocess.TimeoutExpired:
            print("[WARN] NAO timeout")
        except Exception as e:
            print(f"[WARN] NAO error: {e}")

    print("Τέλος")


if __name__ == "__main__":
    main()