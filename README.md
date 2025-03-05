# WhisperX CLI & API - Compatibilité Mac

Ce projet fournit une solution complète pour la transcription audio avec WhisperX, comprenant :
- Un CLI pour la transcription, l'alignement et la diarisation
- Une API REST pour traiter les fichiers audio via une interface web

## Prérequis

- Pour Mac : Python 3.10 recommandé
- Pour Linux/Windows : CUDA 12.4 et libcudnn8
- Node.js (installé automatiquement si nécessaire)
- Un token Hugging Face (pour la diarisation)

## Installation rapide

### Pour Linux/Windows avec GPU NVIDIA
```bash
wget -qO- https://raw.githubusercontent.com/fchevallieratecna/whisper-x-setup/main/setup.sh > setup.sh && chmod +x setup.sh && ./setup.sh
```

### Pour Mac (CPU uniquement)
```bash
# Avec conda
conda create --name whisperx python=3.10
conda activate whisperx

# Installer PyTorch pour CPU
pip install torch torchaudio

# Installer WhisperX
pip install whisperx

# Installer les dépendances supplémentaires
pip install nltk
```

### Pour Mac avec venv (sans conda)
```bash
# Créer un environnement virtuel avec Python 3.10
python3.10 -m venv whisperx_env
source whisperx_env/bin/activate

# Installer PyTorch pour CPU
pip install torch torchaudio

# Installer WhisperX
pip install whisperx

# Installer les dépendances supplémentaires
pip install nltk
```

## Utilisation du CLI

### Sur Linux/Windows (après installation)
```bash
whisperx_cli audio.mp3 --model large-v3 --language fr --diarize --output transcript.srt
```

### Sur Mac
```bash
whisperx audio.mp3 --compute_type int8 --model large-v3 --language fr --diarize --output transcript.srt
```

### Options principales

- `--model` : Modèle WhisperX (défaut: large-v3)
- `--language` : Code de langue (défaut: fr)
- `--diarize` / `--no-diarize` : Activer/désactiver la diarisation
- `--hf_token` : Token Hugging Face pour la diarisation
- `--compute_type` : Type de calcul (utiliser `int8` pour Mac)
- `--output` : Fichier de sortie
- `--output_format` : Format de sortie (json, txt, srt)
- `--nb_speaker` : Nombre exact de locuteurs

## API REST

L'API est automatiquement lancée via PM2 pendant l'installation sur Linux/Windows.

Pour Mac, l'API n'est pas configurée automatiquement.

## Dépannage

- Sur Mac, utilisez toujours `--compute_type int8`
- Pour la diarisation, un token Hugging Face valide est nécessaire
- Sur Mac, les performances seront limitées (CPU uniquement)

### Problèmes connus sur Mac

Si vous rencontrez une erreur liée à OpenSSL (`module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'`), essayez cette solution:

```bash
# Créer un environnement propre avec Python 3.10
python3.10 -m venv whisperx_env
source whisperx_env/bin/activate

# Installer d'abord pyOpenSSL avec une version compatible
pip install pyOpenSSL==22.0.0

# Installer les dépendances dans le bon ordre
pip install urllib3==1.26.6
pip install torch torchaudio
pip install transformers
pip install whisperx
pip install nltk
```

IMPORTANT: Si vous utilisez pyenv, les commandes peuvent toujours pointer vers la mauvaise version. Utilisez le chemin complet vers l'exécutable dans votre environnement virtuel:

```bash
# Utilisez le chemin complet vers l'exécutable whisperx dans votre environnement
./whisperx_env/bin/whisperx audio.mp3 --compute_type int8 --model large-v3 --language fr

# Ou créez un alias temporaire
alias whisperx_fixed="./whisperx_env/bin/whisperx"
whisperx_fixed audio.mp3 --compute_type int8 --model large-v3 --language fr
```

Si cela ne fonctionne toujours pas, essayez une approche alternative avec conda:

```bash
# Créer un environnement conda isolé
conda create -n whisperx_conda python=3.10 -y
conda activate whisperx_conda

# Installer les dépendances
conda install -c conda-forge pyopenssl=22.0.0 -y
pip install whisperx
```

## Licence

MIT
