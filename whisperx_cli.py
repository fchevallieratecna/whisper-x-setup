import argparse
import json
import whisperx

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
    parser = argparse.ArgumentParser(
        description="Transcription, alignement et diarization avec WhisperX"
    )
    parser.add_argument("audio_file", type=str, help="Chemin vers le fichier audio")
    parser.add_argument("--model", type=str, default="large-v3", help="Modèle WhisperX (default: large-v3)")
    parser.add_argument("--language", type=str, default="", help="Code langue (ex: fr, en, etc.)")
    parser.add_argument("--hf_token", type=str, default="", help="Token Hugging Face pour diarization")
    parser.add_argument("--diarize", action="store_true", help="Activer la diarization (nécessite --hf_token)")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch (default: 4)")
    parser.add_argument("--compute_type", type=str, default="float16", help="Type de calcul (default: float16)")
    parser.add_argument("--output", type=str, default="transcription.json", help="Fichier de sortie (default: transcription.json)")
    parser.add_argument("--output_format", type=str, choices=["json", "txt", "srt"], default="json", help="Format de sortie (default: json)")
    args = parser.parse_args()

    # Affichage minimal de la progression
    print(">> Démarrage de la transcription")
    try:
        print("   -> Chargement du modèle...")
        device = "cuda"
        model = whisperx.load_model(args.model, device, compute_type=args.compute_type)
    except Exception as e:
        print("   !! Erreur lors du chargement du modèle :", e)
        return

    try:
        print("   -> Chargement et préparation de l'audio...")
        audio = whisperx.load_audio(args.audio_file)
    except Exception as e:
        print("   !! Erreur lors du chargement de l'audio :", e)
        return

    try:
        print("   -> Transcription...")
        result = model.transcribe(audio, batch_size=args.batch_size)
    except Exception as e:
        print("   !! Erreur pendant la transcription :", e)
        return

    try:
        language = args.language if args.language else result["language"]
        print("   -> Alignement des timestamps...")
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    except Exception as e:
        print("   !! Erreur lors de l'alignement :", e)
        return

    if args.diarize:
        if not args.hf_token:
            print("   !! Erreur: Token Hugging Face requis pour la diarization")
            return
        try:
            print("   -> Exécution de la diarization...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=args.hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result_aligned = whisperx.assign_word_speakers(diarize_segments, result_aligned)
        except Exception as e:
            print("   !! Erreur lors de la diarization :", e)
            return

    try:
        print("   -> Sauvegarde des résultats...")
        output_format = args.output_format.lower()
        if output_format == "json":
            save_json(args.output, result_aligned)
        elif output_format == "txt":
            save_txt(args.output, result_aligned.get("segments", []))
        elif output_format == "srt":
            save_srt(args.output, result_aligned.get("segments", []))
    except Exception as e:
        print("   !! Erreur lors de la sauvegarde :", e)
        return

    print(">> Transcription terminée avec succès.")

if __name__ == "__main__":
    main()