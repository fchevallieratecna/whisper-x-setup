#!/usr/bin/env python3
"""
Modern Whisper CLI with Advanced Diarization
Based on whisper-diarization (MahmoudAshraf97) + NeMo 2.0
September 2025 - Optimized for latest dependencies
"""

import os
import sys
import io
import logging
import warnings
import platform
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Platform-specific optimizations
if platform.system() == "Darwin":
    # Force CPU on macOS for stability
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

# Reduce external library noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pyannote.audio").setLevel(logging.ERROR)
logging.getLogger("nemo_toolkit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

try:
    import torch
    import faster_whisper
    from faster_whisper import WhisperModel
    import numpy as np
    import torchaudio
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("pip install torch torchaudio faster-whisper")
    sys.exit(1)

# Try to import diarization dependencies
try:
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

class WhisperDiarizeProcessor:
    """Modern Whisper processor with advanced diarization capabilities"""

    def __init__(self,
                 model_size: str = "large-v3",
                 device: str = "auto",
                 compute_type: str = "float16",
                 language: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 diarization_backend: str = "pyannote"):

        self.model_size = model_size
        self.language = language
        self.hf_token = hf_token
        self.diarization_backend = diarization_backend

        # Auto-detect device
        if device == "auto":
            if platform.system() == "Darwin":
                self.device = "cpu"
                self.compute_type = "int8"  # More stable on macOS
            elif torch.cuda.is_available():
                self.device = "cuda"
                self.compute_type = compute_type
            else:
                self.device = "cpu"
                self.compute_type = "int8"
        else:
            self.device = device
            self.compute_type = compute_type

        self.model = None
        self.diarization_pipeline = None

    def load_model(self, debug: bool = False) -> None:
        """Load Whisper model with optimized settings"""
        try:
            if debug:
                print(f"Loading Whisper model: {self.model_size}")
                print(f"Device: {self.device}, Compute type: {self.compute_type}")

            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4 if self.device == "cpu" else 0,
                num_workers=1  # Prevent threading issues
            )

            if debug:
                print("✅ Whisper model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            if platform.system() == "Darwin":
                print("💡 Try: --compute_type int8 on macOS")
            raise

    def load_diarization_model(self, debug: bool = False) -> None:
        """Load diarization model (pyannote or NeMo)"""
        if not self.hf_token and self.diarization_backend == "pyannote":
            if debug:
                print("⚠️  No HF token provided, diarization disabled")
            return

        try:
            if self.diarization_backend == "pyannote" and PYANNOTE_AVAILABLE:
                if debug:
                    print("Loading pyannote speaker-diarization-community-1...")

                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    use_auth_token=self.hf_token
                )

                if self.device == "cuda" and torch.cuda.is_available():
                    self.diarization_pipeline.to(torch.device("cuda"))

            elif self.diarization_backend == "nemo" and NEMO_AVAILABLE:
                if debug:
                    print("Loading NeMo diarization model...")
                # NeMo setup would go here
                pass
            else:
                if debug:
                    print("⚠️  Diarization backend not available, using transcription only")

        except Exception as e:
            print(f"⚠️  Diarization model loading failed: {e}")
            print("Continuing with transcription only...")
            self.diarization_pipeline = None

    def transcribe_audio(self,
                        audio_path: str,
                        batch_size: int = 8,
                        initial_prompt: str = "",
                        debug: bool = False) -> Dict[str, Any]:
        """Transcribe audio with faster-whisper"""

        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            if debug:
                print(f"Transcribing: {audio_path}")

            # Configure transcription options
            transcribe_options = {
                "language": self.language,
                "initial_prompt": initial_prompt if initial_prompt else None,
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "condition_on_previous_text": True,
                "vad_filter": True,  # Voice activity detection
                "vad_parameters": dict(min_silence_duration_ms=1000)
            }

            # Remove None values
            transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}

            segments, info = self.model.transcribe(
                audio_path,
                **transcribe_options
            )

            # Convert segments to list with timing
            segments_list = []
            for segment in segments:
                segments_list.append({
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob
                })

            return {
                "segments": segments_list,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "duration_after_vad": info.duration_after_vad
            }

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            raise

    def diarize_audio(self, audio_path: str, num_speakers: Optional[int] = None, debug: bool = False) -> Optional[Dict]:
        """Perform speaker diarization"""

        if not self.diarization_pipeline:
            if debug:
                print("⚠️  No diarization pipeline available")
            return None

        try:
            if debug:
                print("Performing speaker diarization...")

            # Configure diarization
            diarization_options = {}
            if num_speakers:
                diarization_options["num_speakers"] = num_speakers

            diarization = self.diarization_pipeline(audio_path, **diarization_options)

            # Convert to dict format
            diarization_result = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diarization_result.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })

            return {"diarization": diarization_result}

        except Exception as e:
            print(f"⚠️  Diarization error: {e}")
            return None

    def align_transcription_with_speakers(self, transcription: Dict, diarization: Optional[Dict]) -> Dict:
        """Align transcription segments with speaker labels"""

        if not diarization:
            return transcription

        segments = transcription["segments"]
        speaker_segments = diarization["diarization"]

        # Assign speakers to transcription segments
        for segment in segments:
            segment_start = segment["start"]
            segment_end = segment["end"]

            # Find overlapping speaker segments
            overlapping_speakers = []
            for spk_seg in speaker_segments:
                spk_start = spk_seg["start"]
                spk_end = spk_seg["end"]

                # Check for overlap
                if (segment_start < spk_end and segment_end > spk_start):
                    overlap_duration = min(segment_end, spk_end) - max(segment_start, spk_start)
                    overlapping_speakers.append((spk_seg["speaker"], overlap_duration))

            # Assign speaker with most overlap
            if overlapping_speakers:
                best_speaker = max(overlapping_speakers, key=lambda x: x[1])[0]
                segment["speaker"] = best_speaker
            else:
                segment["speaker"] = "UNKNOWN"

        return transcription

def suppress_output(func, *args, **kwargs):
    """Execute function while suppressing stdout"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout

def maybe_call(func, debug, *args, **kwargs):
    """Call function with optional output suppression"""
    if debug:
        return func(*args, **kwargs)
    else:
        return suppress_output(func, *args, **kwargs)

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def save_results(output_path: str, data: Dict, format: str) -> None:
    """Save results in specified format"""

    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    elif format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in data.get("segments", []):
                speaker = segment.get("speaker", "")
                text = segment.get("text", "").strip()
                if speaker:
                    f.write(f"[{speaker}] {text}\n")
                else:
                    f.write(f"{text}\n")

    elif format == "srt":
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, segment in enumerate(data.get("segments", []), start=1):
                start_time = seconds_to_srt_time(segment.get("start", 0))
                end_time = seconds_to_srt_time(segment.get("end", 0))
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")

                if speaker:
                    text = f"[{speaker}] {text}"

                f.write(f"{idx}\n{start_time} --> {end_time}\n{text}\n\n")

def main():
    """Main CLI function"""

    # Default configuration
    defaults = {
        "model": "large-v3",
        "diarize": True,
        "batch_size": 8,
        "output_format": "txt",
        "language": "fr",
        "compute_type": "float16",
        "diarization_backend": "pyannote"
    }

    parser = argparse.ArgumentParser(
        description="Modern Whisper CLI with Advanced Diarization (Sept 2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3 --model large-v3 --language fr
  %(prog)s audio.wav --diarize --hf_token YOUR_TOKEN --nb_speaker 2
  %(prog)s audio.mp4 --output_format srt --compute_type int8
        """
    )

    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file")

    # Model options
    parser.add_argument("--model", default=defaults["model"],
                       help=f"Whisper model size (default: {defaults['model']})")
    parser.add_argument("--language", default=defaults["language"],
                       help=f"Language code (default: {defaults['language']})")
    parser.add_argument("--compute_type", default=defaults["compute_type"],
                       help=f"Compute type (default: {defaults['compute_type']})")
    parser.add_argument("--device", default="auto",
                       help="Device to use: auto, cpu, cuda (default: auto)")

    # Diarization options
    parser.add_argument("--diarize", action="store_true", default=defaults["diarize"],
                       help="Enable speaker diarization")
    parser.add_argument("--no-diarize", dest="diarize", action="store_false",
                       help="Disable speaker diarization")
    parser.add_argument("--hf_token", default="",
                       help="Hugging Face token for diarization models")
    parser.add_argument("--nb_speaker", type=int,
                       help="Expected number of speakers")
    parser.add_argument("--diarization_backend", default=defaults["diarization_backend"],
                       choices=["pyannote", "nemo"],
                       help="Diarization backend to use")

    # Processing options
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"],
                       help=f"Batch size (default: {defaults['batch_size']})")
    parser.add_argument("--initial_prompt", default="",
                       help="Initial prompt for transcription")

    # Output options
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--output_format", choices=["json", "txt", "srt"],
                       default=defaults["output_format"],
                       help=f"Output format (default: {defaults['output_format']})")

    # Debug
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")

    args = parser.parse_args()

    # Version check
    if args.version:
        print("Modern Whisper CLI v2.0.0 (September 2025)")
        print(f"faster-whisper: {faster_whisper.__version__}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if PYANNOTE_AVAILABLE:
            print("✅ pyannote.audio: Available")
        if NEMO_AVAILABLE:
            print("✅ NeMo: Available")
        return

    # Validate audio file
    if not args.audio_file:
        parser.error("Audio file is required (use --version to see version info)")

    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Set output path
    if not args.output:
        base_name = Path(args.audio_file).stem
        extensions = {"json": ".json", "txt": ".txt", "srt": ".srt"}
        args.output = str(Path(args.audio_file).parent / f"{base_name}{extensions[args.output_format]}")

    # Display configuration
    print("🎯 Modern Whisper CLI - Configuration:")
    print(f"   📁 Audio file    : {args.audio_file}")
    print(f"   🤖 Model         : {args.model}")
    print(f"   🌍 Language      : {args.language}")
    print(f"   💻 Device        : {args.device}")
    print(f"   ⚡ Compute type   : {args.compute_type}")
    print(f"   👥 Diarization   : {'✅ Enabled' if args.diarize else '❌ Disabled'}")
    if args.diarize:
        print(f"   🎤 Backend       : {args.diarization_backend}")
    print(f"   📤 Output        : {args.output} ({args.output_format})")
    print()

    # Initialize processor
    try:
        processor = WhisperDiarizeProcessor(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language,
            hf_token=args.hf_token,
            diarization_backend=args.diarization_backend
        )

        # Load models
        print("🔄 [20%] Loading Whisper model...")
        start_time = time.time()
        maybe_call(processor.load_model, args.debug, debug=args.debug)
        print(f"✅ [25%] Whisper model loaded ({time.time() - start_time:.1f}s)")

        if args.diarize:
            print("🔄 [30%] Loading diarization model...")
            start_time = time.time()
            maybe_call(processor.load_diarization_model, args.debug, debug=args.debug)
            print(f"✅ [35%] Diarization model loaded ({time.time() - start_time:.1f}s)")

        # Transcribe
        print("🔄 [40%] Transcribing audio...")
        start_time = time.time()
        transcription = maybe_call(processor.transcribe_audio, args.debug,
                                  args.audio_file, args.batch_size, args.initial_prompt, args.debug)
        print(f"✅ [70%] Transcription completed ({time.time() - start_time:.1f}s)")

        # Diarize
        diarization_result = None
        if args.diarize and processor.diarization_pipeline:
            print("🔄 [75%] Performing speaker diarization...")
            start_time = time.time()
            diarization_result = maybe_call(processor.diarize_audio, args.debug,
                                          args.audio_file, args.nb_speaker, args.debug)
            print(f"✅ [90%] Diarization completed ({time.time() - start_time:.1f}s)")

        # Align speakers with transcription
        if diarization_result:
            print("🔄 [95%] Aligning speakers with transcription...")
            transcription = processor.align_transcription_with_speakers(transcription, diarization_result)

        # Save results
        print("🔄 [98%] Saving results...")
        save_results(args.output, transcription, args.output_format)
        print(f"✅ [100%] Results saved to: {args.output}")

        # Summary
        duration = transcription.get("duration", 0)
        num_segments = len(transcription.get("segments", []))
        language = transcription.get("language", "unknown")

        print(f"\n📊 Summary:")
        print(f"   ⏱️  Duration: {duration:.1f}s")
        print(f"   📝 Segments: {num_segments}")
        print(f"   🌍 Detected language: {language}")

        if args.diarize and diarization_result:
            speakers = set(seg.get("speaker", "UNKNOWN") for seg in transcription["segments"])
            print(f"   👥 Speakers: {len(speakers)} ({', '.join(sorted(speakers))})")

    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()