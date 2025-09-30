# 🎤 Modern Whisper Setup (September 2025)

**Transcription et diarization avancées avec les dernières technologies 2025**

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/fchevallieratecna/whisper-x-setup)
[![Python](https://img.shields.io/badge/python-3.9--3.12-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.7.1-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/cuda-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🆕 Nouveautés 2025

- **🚀 Passage à `faster-whisper`** : Performances 70x plus rapides que WhisperX
- **🧠 NeMo 2.0** : Diarization de pointe avec NVIDIA NeMo
- **⚡ PyTorch 2.7.1** : Support CUDA 12.8 optimisé
- **🎯 CLI moderne** : Interface utilisateur repensée
- **📊 Community-1** : Modèle de diarization pyannote le plus récent
- **🐍 Python 3.9-3.12** : Compatibilité étendue

## 📋 Pré-requis

### Système
- **Ubuntu 20.04+** ou **macOS 12+**
- **Python 3.9-3.12**
- **Git** et **FFmpeg**
- **16GB RAM** recommandés (8GB minimum)

### GPU (Optionnel mais recommandé)
- **NVIDIA GPU** avec **8GB+ VRAM**
- **CUDA 11.0+** (12.8 optimal)
- **Driver NVIDIA** récent

## 🚀 Installation Rapide

### Installation Complète (Recommandée)

```bash
# Cloner le projet
git clone https://github.com/fchevallieratecna/whisper-x-setup.git
cd whisper-x-setup

# Lancer l'installation moderne
chmod +x setup_modern.sh
./setup_modern.sh --verbose --hf-token=YOUR_HF_TOKEN
```

### Options d'installation

```bash
# Installation avec options avancées
./setup_modern.sh \
  --verbose \
  --hf-token=hf_xxxxxxxxxxxx \
  --ngrok-token=xxxxxxxxxxxx \
  --use-conda

# Installation API uniquement
./setup_modern.sh --only-api --ngrok-token=xxxxxxxxxxxx

# Installation sans NeMo (plus rapide)
./setup_modern.sh --no-nemo --hf-token=hf_xxxxxxxxxxxx
```

## 📱 Utilisation du CLI

### Commandes de base

```bash
# Transcription simple
whisper_modern_cli audio.mp3 --model large-v3 --language fr

# Avec diarization (identification des locuteurs)
whisper_modern_cli meeting.wav \
  --diarize \
  --hf_token YOUR_TOKEN \
  --nb_speaker 3 \
  --output_format srt

# Sur macOS (CPU optimisé)
whisper_modern_cli audio.mp3 \
  --compute_type int8 \
  --device cpu \
  --model large-v3
```

### Options avancées

```bash
# Diarization avec NeMo (plus précis)
whisper_modern_cli interview.mp3 \
  --diarize \
  --diarization_backend nemo \
  --model large-v3 \
  --language fr

# Traitement par batch optimisé
whisper_modern_cli long_audio.wav \
  --batch_size 16 \
  --compute_type float16 \
  --initial_prompt "Cette réunion concerne..."

# Debug et développement
whisper_modern_cli test.mp3 \
  --debug \
  --model base \
  --output debug_output.json \
  --output_format json
```

## 🎯 Formats de sortie

### TXT (Défaut)
```
[SPEAKER_00] Bonjour et bienvenue dans cette réunion.
[SPEAKER_01] Merci, je suis ravi d'être ici.
[SPEAKER_00] Commençons par le premier point de l'ordre du jour.
```

### SRT (Sous-titres)
```
1
00:00:00,000 --> 00:00:03,500
[SPEAKER_00] Bonjour et bienvenue dans cette réunion.

2
00:00:03,500 --> 00:00:06,200
[SPEAKER_01] Merci, je suis ravi d'être ici.
```

### JSON (Détaillé)
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "Bonjour et bienvenue dans cette réunion.",
      "speaker": "SPEAKER_00",
      "confidence": 0.98
    }
  ],
  "language": "fr",
  "duration": 1800.5
}
```

## 🔧 Configuration Avancée

### Variables d'environnement

```bash
# Token Hugging Face
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Configuration GPU
export CUDA_VISIBLE_DEVICES="0"

# Optimisation mémoire
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

### Fichier de configuration (optionnel)

Créez `config.yaml` :

```yaml
model:
  size: "large-v3"
  language: "fr"
  compute_type: "float16"

diarization:
  enabled: true
  backend: "pyannote"  # ou "nemo"
  min_speakers: 1
  max_speakers: 10

output:
  format: "srt"
  include_timestamps: true
  include_confidence: true

processing:
  batch_size: 8
  use_vad: true
  beam_size: 5
```

## 📊 Comparaison des Performances

| Méthode | Vitesse | Précision | Diarization | VRAM |
|---------|---------|-----------|-------------|------|
| OpenAI Whisper | 1x | ⭐⭐⭐⭐ | ❌ | 6GB |
| WhisperX | 70x | ⭐⭐⭐⭐ | ⭐⭐⭐ | 8GB |
| **Modern Setup** | **70x** | **⭐⭐⭐⭐⭐** | **⭐⭐⭐⭐⭐** | **6GB** |

## 🔍 Dépannage

### Erreurs communes

```bash
# Erreur de mémoire GPU
whisper_modern_cli audio.mp3 --compute_type int8 --batch_size 4

# Problème de token Hugging Face
whisper_modern_cli audio.mp3 --no-diarize

# Erreur macOS
whisper_modern_cli audio.mp3 --device cpu --compute_type int8
```

### Vérification de l'installation

```bash
# Version et informations
whisper_modern_cli --version

# Test rapide
whisper_modern_cli test_audio.wav --model base --debug
```

## 🌐 API REST

L'installation inclut également une API REST moderne :

```bash
# Démarrer l'API
pm2 start whisper-api-modern

# Test de l'API
curl -X POST "http://localhost:3000/api/transcribe" \
  -F "file=@audio.mp3" \
  -F "model=large-v3" \
  -F "language=fr" \
  -F "diarize=true"
```

### Endpoints disponibles

- `POST /api/transcribe` - Transcription avec diarization
- `GET /api/models` - Liste des modèles disponibles
- `GET /api/status` - Statut du service
- `GET /api/health` - Health check

## 🔄 Migration depuis WhisperX

### Commandes équivalentes

```bash
# Ancienne commande WhisperX
whisperx audio.mp3 --model large-v2 --diarize --language fr

# Nouvelle commande (Modern Setup)
whisper_modern_cli audio.mp3 --model large-v3 --diarize --language fr
```

### Avantages de la migration

- **Performance** : 2-3x plus rapide
- **Précision** : Meilleure diarization avec Community-1
- **Stabilité** : Moins de bugs et dépendances
- **Support** : Projet activement maintenu

## 📚 Documentation Technique

### Architecture

```
Modern Whisper Setup
├── faster-whisper (ASR core)
├── pyannote.audio (Diarization)
├── NeMo 2.0 (Advanced diarization)
├── PyTorch 2.7.1 (Backend)
└── Custom CLI (Interface)
```

### Modèles supportés

| Modèle | Taille | VRAM | Langue | Vitesse |
|--------|--------|------|--------|---------|
| tiny | 39MB | 1GB | Multilingue | Très rapide |
| base | 74MB | 1GB | Multilingue | Rapide |
| small | 244MB | 2GB | Multilingue | Moyen |
| medium | 769MB | 5GB | Multilingue | Lent |
| large-v3 | 1550MB | 10GB | Multilingue | Très lent |

## 🤝 Contribution

### Développement local

```bash
# Installation en mode développement
git clone https://github.com/fchevallieratecna/whisper-x-setup.git
cd whisper-x-setup

# Installation des dépendances de développement
pip install -r requirements_modern.txt
pip install -e .

# Tests
python -m pytest tests/
```

### Signaler un bug

1. Vérifiez les [issues existantes](https://github.com/fchevallieratecna/whisper-x-setup/issues)
2. Créez une nouvelle issue avec :
   - Version du système
   - Commande utilisée
   - Logs d'erreur complets
   - Fichier audio de test (si possible)

## 📄 Licence

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- [OpenAI](https://openai.com/) pour Whisper
- [SYSTRAN](https://github.com/SYSTRAN/faster-whisper) pour faster-whisper
- [pyannote](https://github.com/pyannote/pyannote-audio) pour la diarization
- [NVIDIA](https://github.com/NVIDIA/NeMo) pour NeMo
- [MahmoudAshraf97](https://github.com/MahmoudAshraf97/whisper-diarization) pour l'inspiration

## 📞 Support

- 📧 Email : support@whisper-modern.com
- 💬 Discord : [Communauté Whisper FR](https://discord.gg/whisper-fr)
- 📖 Wiki : [Documentation complète](https://github.com/fchevallieratecna/whisper-x-setup/wiki)

---

**Fait avec ❤️ pour la communauté française de l'IA**