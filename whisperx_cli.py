import os
import sys
import io
import logging
import warnings
import platform

# Forcer l'utilisation du CPU sur macOS
if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_DATASETS_OFFLINE"] = "1"

# Suppress external logs unless debug is enabled (will be set later)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pyannote.audio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded your loaded checkpoint.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Lightning automatically upgraded your loaded checkpoint.*")

import argparse
import json
import whisperx

def suppress_stdout(func, *args, **kwargs):
    """Execute a function while temporarily suppressing stdout."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout

def maybe_call(func, debug, *args, **kwargs):
    """Call func with output suppressed if debug is False."""
    if debug:
        return func(*args, **kwargs)
    else:
        return suppress_stdout(func, *args, **kwargs)

def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def save_json(output_path, data):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_txt(output_path, segments):
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            speaker = seg.get("speaker", "")
            line = seg.get("text", "").strip()
            if speaker:
                line = f"[{speaker}] {line}"
            f.write(line + "\n")

def save_srt(output_path, segments):
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start_time = seconds_to_srt_time(seg.get("start", 0))
            end_time = seconds_to_srt_time(seg.get("end", 0))
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")
            if speaker:
                text = f"[{speaker}] {text}"
            f.write(f"{idx}\n{start_time} --> {end_time}\n{text}\n\n")

def main():
    # Default parameters
    default_values = {
        "model": "large-v3",
        "diarize": True,
        "batch_size": 8,
        "output_format": "txt",
        "language": "fr",
    }
    parser = argparse.ArgumentParser(
        description="Transcription, alignment and diarization with WhisperX"
    )
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    parser.add_argument("--model", type=str, default=default_values["model"],
                        help="WhisperX model to use (default: large-v3)")
    parser.add_argument("--diarize", dest="diarize", action="store_true",
                        default=default_values["diarize"],
                        help="Enable diarization (default: enabled)")
    parser.add_argument("--no-diarize", dest="diarize", action="store_false",
                        help="Disable diarization")
    parser.add_argument("--batch_size", type=int, default=default_values["batch_size"],
                        help="Batch size (default: 8)")
    parser.add_argument("--compute_type", type=str, default="float16",
                        help="Compute type (default: float16)")
    parser.add_argument("--language", type=str, default=default_values["language"],
                        help="Language code (default: fr)")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face token for diarization")
    parser.add_argument("--initial_prompt", type=str, default="",
                        help="Initial prompt passed in asr_options (default: empty)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: same as audio file with corresponding extension)")
    parser.add_argument("--output_format", type=str, choices=["json", "txt", "srt"],
                        default=default_values["output_format"],
                        help="Output format (default: txt)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode (display all logs)")
    parser.add_argument("--nb_speaker", type=int, default=None, 
                        help="Exact number of speakers (sets both min_speakers and max_speakers)")
    args = parser.parse_args()

    # Set default output if not provided
    if args.output is None:
        base, _ = os.path.splitext(os.path.basename(args.audio_file))
        ext = {"json": ".json", "txt": ".txt", "srt": ".srt"}[args.output_format.lower()]
        args.output = os.path.join(os.path.dirname(args.audio_file), base + ext)

    # Display used parameters
    print(">> Parameters used:")
    print(f"   - audio_file     : {args.audio_file}")

    # Afficher tous les paramètres
    all_params = vars(args)
    for key, val in all_params.items():
        if key == 'audio_file':
            continue  # Déjà affiché
        
        # Vérifier si le paramètre est dans default_values
        if key in default_values:
            if val == default_values[key]:
                print(f"   - {key:<15}: {val} (default)")
            else:
                print(f"   - {key:<15}: {val} (overridden)")
        else:
            # Pour les paramètres qui n'ont pas de valeur par défaut dans default_values
            print(f"   - {key:<15}: {val}")

    # Ajouter l'information sur le device
    print(f"   - device         : {'cpu' if platform.system() == 'Darwin' else 'cuda'} (auto-detected)")
    print("")

    # Définir les étapes principales et leur progression
    steps = [
        (10, "Chargement du modèle"),
        (20, "Préparation de l'audio"),
        (40, "Transcription"),
        (60, "Alignement des timestamps"),
        (80, "Diarisation") if args.diarize else None,
        (95, "Sauvegarde des résultats"),
        (100, "Transcription terminée")
    ]
    steps = [s for s in steps if s is not None]
    
    print(">> Démarrage de la transcription")
    try:
        print(f"[{steps[0][0]}%] - {steps[0][1]}...")
        # Toujours utiliser CPU sur macOS
        device = "cpu" if platform.system() == "Darwin" else "cuda"
        asr_options = {"initial_prompt": args.initial_prompt}
        
        # Afficher les paramètres de chargement pour le débogage
        if args.debug:
            print(f"   -> Using device: {device}, compute_type: {args.compute_type}")
        
        # Pass language directly
        model = maybe_call(whisperx.load_model, args.debug, args.model, device,
                           compute_type=args.compute_type, language=args.language, asr_options=asr_options)
    except Exception as e:
        print("   !! Error loading model:", e)
        print("   !! Sur macOS, essayez avec --compute_type int8")
        return

    try:
        print(f"[{steps[1][0]}%] - {steps[1][1]}...")
        audio = maybe_call(whisperx.load_audio, args.debug, args.audio_file)
    except Exception as e:
        print("   !! Error loading audio:", e)
        return

    try:
        print(f"[{steps[2][0]}%] - {steps[2][1]}...")
        result = maybe_call(model.transcribe, args.debug, audio, batch_size=args.batch_size)
    except Exception as e:
        print("   !! Error during transcription:", e)
        return

    try:
        print(f"[{steps[3][0]}%] - {steps[3][1]}...")
        lang = args.language if args.language else result.get("language", "fr")
        model_a, metadata = maybe_call(whisperx.load_align_model, args.debug, language_code=lang, device=device)
        result_aligned = maybe_call(whisperx.align, args.debug, result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    except Exception as e:
        print("   !! Error during alignment:", e)
        return

    if args.diarize:
        try:
            diarize_step = next(s for s in steps if s[1] == "Diarisation")
            print(f"[{diarize_step[0]}%] - {diarize_step[1]}...")
            diarize_model = maybe_call(whisperx.DiarizationPipeline, args.debug, use_auth_token=args.hf_token, device=device)
            if args.nb_speaker is not None:
                diarize_segments = maybe_call(
                    diarize_model, args.debug, audio, min_speakers=args.nb_speaker, max_speakers=args.nb_speaker
                )
            else:
                diarize_segments = maybe_call(
                    diarize_model, args.debug, audio, min_speakers=args.min_speakers, max_speakers=args.max_speakers
                )
            result_aligned = whisperx.assign_word_speakers(diarize_segments, result_aligned)
        except Exception as e:
            if "token" in str(e).lower():
                token_input = input("   -> Diarization failed due to token issues. Please enter your Hugging Face token: ").strip()
                if not token_input:
                    print("   !! Error: No token provided. Diarization cannot be performed.")
                    return
                try:
                    diarize_model = maybe_call(whisperx.DiarizationPipeline, args.debug, use_auth_token=token_input, device=device)
                    diarize_segments = maybe_call(diarize_model, args.debug, audio)
                    result_aligned = whisperx.assign_word_speakers(diarize_segments, result_aligned)
                except Exception as e2:
                    print("   !! Error during diarization after token input:", e2)
                    return
            else:
                print("   !! Error during diarization:", e)
                return

    try:
        save_step = next(s for s in steps if s[1] == "Sauvegarde des résultats")
        print(f"[{save_step[0]}%] - {save_step[1]}...")
        fmt = args.output_format.lower()
        if fmt == "json":
            save_json(args.output, result_aligned)
        elif fmt == "txt":
            save_txt(args.output, result_aligned.get("segments", []))
        elif fmt == "srt":
            save_srt(args.output, result_aligned.get("segments", []))
    except Exception as e:
        print("   !! Error during saving:", e)
        return

    final_step = next(s for s in steps if s[1] == "Transcription terminée")
    print(f"[{final_step[0]}%] - {final_step[1]}.")

if __name__ == "__main__":
    main()