import os
import sys
import io
import logging
import warnings

# Suppress external logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pyannote.audio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

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
    # New argument for initial prompt (empty by default)
    parser.add_argument("--initial_prompt", type=str, default="",
                        help="Initial prompt to pass via asr_options (default: empty)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: same as audio file with corresponding extension)")
    parser.add_argument("--output_format", type=str, choices=["json", "txt", "srt"],
                        default=default_values["output_format"],
                        help="Output format (default: txt)")
    args = parser.parse_args()

    # Set default output file if not provided
    if args.output is None:
        base, _ = os.path.splitext(os.path.basename(args.audio_file))
        ext = {"json": ".json", "txt": ".txt", "srt": ".srt"}[args.output_format.lower()]
        args.output = os.path.join(os.path.dirname(args.audio_file), base + ext)

    # Display parameters used
    print(">> Parameters used:")
    print(f"   - audio_file     : {args.audio_file}")
    for key in default_values:
        val = getattr(args, key)
        if val == default_values[key]:
            print(f"   - {key:<15}: {val} (default)")
        else:
            print(f"   - {key:<15}: {val} (overridden)")
    print(f"   - output_format  : {args.output_format} (default)" if args.output_format == default_values["output_format"]
          else f"   - output_format  : {args.output_format} (overridden)")
    print(f"   - output         : {args.output} (default if not specified)")
    print(f"   - initial_prompt : '{args.initial_prompt}' (default)" if args.initial_prompt == "" else f"   - initial_prompt : '{args.initial_prompt}' (overridden)")
    print("")

    print(">> Starting transcription")
    try:
        print("   -> Chargement du modèle...")
        device = "cuda"
        asr_options = {"initial_prompt": args.initial_prompt}
        # On passe la langue directement via l'argument language de load_model()
        model = suppress_stdout(
            whisperx.load_model,
            args.model,
            device,
            compute_type=args.compute_type,
            language=args.language,      # Passage de la langue ici
            asr_options=asr_options
        )
    except Exception as e:
        print("   !! Erreur lors du chargement du modèle :", e)
    return

    try:
        print("   -> Loading and preparing audio...")
        audio = suppress_stdout(whisperx.load_audio, args.audio_file)
    except Exception as e:
        print("   !! Error loading audio:", e)
        return

    try:
        print("   -> Transcribing...")
        result = suppress_stdout(model.transcribe, audio, batch_size=args.batch_size)
    except Exception as e:
        print("   !! Error during transcription:", e)
        return

    try:
        lang = args.language if args.language else result.get("language", "fr")
        print("   -> Aligning timestamps...")
        model_a, metadata = suppress_stdout(whisperx.load_align_model, language_code=lang, device=device)
        result_aligned = suppress_stdout(whisperx.align, result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    except Exception as e:
        print("   !! Error during alignment:", e)
        return

    if args.diarize:
        try:
            print("   -> Running diarization...")
            diarize_model = suppress_stdout(whisperx.DiarizationPipeline, use_auth_token=args.hf_token, device=device)
            diarize_segments = suppress_stdout(diarize_model, audio)
            result_aligned = whisperx.assign_word_speakers(diarize_segments, result_aligned)
        except Exception as e:
            if "token" in str(e).lower():
                token_input = input("   -> Diarization failed due to token issues. Please enter your Hugging Face token: ").strip()
                if not token_input:
                    print("   !! Error: No token provided. Diarization cannot be performed.")
                    return
                try:
                    diarize_model = suppress_stdout(whisperx.DiarizationPipeline, use_auth_token=token_input, device=device)
                    diarize_segments = suppress_stdout(diarize_model, audio)
                    result_aligned = whisperx.assign_word_speakers(diarize_segments, result_aligned)
                except Exception as e2:
                    print("   !! Error during diarization after token input:", e2)
                    return
            else:
                print("   !! Error during diarization:", e)
                return

    try:
        print("   -> Saving results...")
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

    print(">> Transcription completed successfully.")

if __name__ == "__main__":
    main()