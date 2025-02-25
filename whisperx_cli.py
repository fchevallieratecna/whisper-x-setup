import argparse
import json
import whisperx
import os

def seconds_to_srt_time(seconds: float) -> str:
    """Convertit un temps en secondes au format SRT (HH:MM:SS,mmm)."""
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
        description="Script de transcription avec WhisperX, alignement et option diarization."
    )
    parser.add_argument("audio_file", type=str, help="Chemin vers le fichier audio (ex: audio.mp3)")
    parser.add_argument("--model", type=str, default="large-v3", help="Modèle WhisperX à utiliser (default: large-v3)")
    parser.add_argument("--language", type=str, default="", help="Code langue (ex: fr, en, etc.). Si non spécifié, le langage détecté sera utilisé.")
    parser.add_argument("--hf_token", type=str, default="", help="Token Hugging Face (requis pour la diarization)")
    parser.add_argument("--diarize", action="store_true", help="Active la diarization (nécessite --hf_token)")
    parser.add_argument("--batch_size", type=int, default=4, help="Taille du batch pour la transcription (default: 4)")
    parser.add_argument("--compute_type", type=str, default="float16", help="Type de calcul, ex: float16 (GPU) ou int8 (pour réduire la mémoire GPU)")
    parser.add_argument("--output", type=str, default="transcription.json", help="Fichier de sortie (default: transcription.json)")
    parser.add_argument("--output_format", type=str, choices=["json", "txt", "srt"], default="json", help="Format de sortie : json, txt ou srt (default: json)")

    args = parser.parse_args()

    device = "cuda"
    print(f"Chargement du modèle {args.model} sur {device} avec compute_type {args.compute_type} ...")
    model = whisperx.load_model(args.model, device, compute_type=args.compute_type)
    
    print("Chargement et préparation de l'audio ...")
    audio = whisperx.load_audio(args.audio_file)
    
    print("Transcription ...")
    result = model.transcribe(audio, batch_size=args.batch_size)
    print("Transcription terminée.")

    language = args.language if args.language else result["language"]
    
    print("Chargement du modèle d'alignement ...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    
    print("Alignement des timestamps ...")
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    if args.diarize:
        if not args.hf_token:
            print("Erreur : Le token Hugging Face est requis pour la diarization (--hf_token).")
        else:
            print("Exécution de la diarization ...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=args.hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result_aligned = whisperx.assign_word_speakers(diarize_segments, result_aligned)
    
    output_path = args.output
    output_format = args.output_format.lower()
    print(f"Sauvegarde des résultats au format {output_format} dans {output_path} ...")
    if output_format == "json":
        save_json(output_path, result_aligned)
    elif output_format == "txt":
        segments = result_aligned.get("segments", [])
        save_txt(output_path, segments)
    elif output_format == "srt":
        segments = result_aligned.get("segments", [])
        save_srt(output_path, segments)
    else:
        print("Format de sortie non supporté.")
    
    print("Opération terminée.")

if __name__ == "__main__":
    main()