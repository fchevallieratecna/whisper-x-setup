# WhisperX CLI & API

Ce projet fournit une solution complète pour la transcription audio avec WhisperX, comprenant :
- Un CLI pour la transcription, l'alignement et la diarisation
- Une API REST pour traiter les fichiers audio via une interface web

## Prérequis

- CUDA 12.4
- libcudnn8
- Node.js (installé automatiquement si nécessaire)
- Un token Hugging Face (pour la diarisation)

## Installation rapide

Exécutez la commande suivante pour télécharger et lancer le script d'installation :

```bash
curl -sSL https://raw.githubusercontent.com/fchevallieratecna/whisper-x-setup/main/setup.sh | bash
```

ou avec wget :

```bash
wget -qO- https://raw.githubusercontent.com/fchevallieratecna/whisper-x-setup/main/setup.sh | bash
```

Pour le mode verbose, ajoutez `-v` :

```bash
curl -sSL https://raw.githubusercontent.com/fchevallieratecna/whisper-x-setup/main/setup.sh | bash -s -- -v
```

## Utilisation du CLI

Après l'installation, vous pouvez utiliser la commande `whisperx_cli` :

```bash
whisperx_cli audio.mp3 --model large-v3 --language fr --diarize --output transcript.srt
```

### Options principales

- `--model` : Modèle WhisperX (défaut: large-v3)
- `--language` : Code de langue (défaut: fr)
- `--diarize` / `--no-diarize` : Activer/désactiver la diarisation
- `--hf_token` : Token Hugging Face pour la diarisation
- `--output` : Fichier de sortie
- `--output_format` : Format de sortie (json, txt, srt)
- `--nb_speaker` : Nombre exact de locuteurs
- `--debug` : Mode debug

## API REST

L'API est automatiquement lancée via PM2 pendant l'installation.

### Mise à jour de l'API

Pour mettre à jour l'API :

```bash
whisper_api_update
```

### Accès à l'API

L'API est accessible à l'adresse : http://localhost:3000

## Dépannage

Si vous rencontrez des problèmes :

1. Vérifiez que CUDA 12.4 est correctement installé
2. Assurez-vous que libcudnn8 est installé
3. Pour la diarisation, un token Hugging Face valide est nécessaire

## Licence

MIT
