#!/bin/bash
# Nom du script : setup.sh
# Ce script clone le dépôt, installe l'environnement virtuel et les dépendances, puis lance un test de transcription.
# Rendre exécutable avec : chmod +x setup.sh

# --- Variables d'affichage ---
BOLD="\e[1m"
RESET="\e[0m"
LOADING="⏳"
DONE="✅"

# --- URL du dépôt à cloner ---
REPO_URL="https://github.com/fchevallieratecna/whisper-x-setup.git"
REPO_DIR="whisper-x-setup"

# --- Fonctions de log avec affichage sur une seule ligne ---
run_step() {
  local description="$1"
  shift
  echo -ne "${LOADING} ${BOLD}${description}${RESET} [${LOADING} en cours...]"
  "$@" > /dev/null 2>&1
  echo -e "\r${DONE} ${BOLD}${description}${RESET} [${DONE} terminé]"
}

# --- Vérification de CUDA 12.4 ---
check_cuda() {
  echo -ne "${LOADING} ${BOLD}Vérification de CUDA 12.4${RESET} [${LOADING} en cours...]"
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep -o "release [0-9]*\.[0-9]*" | head -n1 | cut -d' ' -f2)
  elif [ -f "/usr/local/cuda/version.txt" ]; then
    CUDA_VERSION=$(grep -o "[0-9]*\.[0-9]*" /usr/local/cuda/version.txt | head -n1)
  else
    CUDA_VERSION=""
  fi

  if [[ "$CUDA_VERSION" == "12.4" ]]; then
    echo -e "\r${DONE} ${BOLD}Vérification de CUDA 12.4${RESET} [${DONE} présent (version ${CUDA_VERSION})]"
  else
    echo -e "\r❌ ${BOLD}Vérification de CUDA 12.4${RESET} [${BOLD}ERREUR${RESET}]"
    echo "CUDA 12.4 n'est pas installé sur cette machine (trouvé: '$CUDA_VERSION')."
    exit 1
  fi
}

# --- Début du script ---

# 1. Cloner le dépôt (si nécessaire)
if [ ! -d "$REPO_DIR" ]; then
  echo -e "${LOADING} ${BOLD}Clonage du dépôt depuis GitHub${RESET} [${LOADING} en cours...]"
  git clone "$REPO_URL" > /dev/null 2>&1
  echo -e "\r${DONE} ${BOLD}Clonage du dépôt depuis GitHub${RESET} [${DONE} terminé]"
else
  echo -e "${DONE} ${BOLD}Dépôt déjà cloné${RESET}"
fi

# Se placer dans le dossier cloné
cd "$REPO_DIR" || exit

# 2. Vérifier la présence du fichier Python
if [ ! -f "whisperx_cli.py" ]; then
  echo -e "\n❌ ${BOLD}Erreur${RESET}: Le fichier 'whisperx_cli.py' est introuvable dans $(pwd)."
  echo "Veuillez vérifier la structure de votre dépôt."
  exit 1
else
  echo -e "${DONE} ${BOLD}Fichier 'whisperx_cli.py' trouvé${RESET}"
fi

# 3. Vérifier CUDA 12.4
check_cuda

# 4. Création de l'environnement virtuel
run_step "Création de l'environnement virtuel 'whisperx_env'" python3 -m venv whisperx_env

# 5. Activation de l'environnement virtuel (cette étape reste dans le shell courant)
echo -ne "${LOADING} ${BOLD}Activation de l'environnement virtuel${RESET} [${LOADING} en cours...]"
source whisperx_env/bin/activate
echo -e "\r${DONE} ${BOLD}Activation de l'environnement virtuel${RESET} [${DONE} terminé]"

# 6. Mise à jour de pip
run_step "Mise à jour de pip" pip install --upgrade pip

# 7. Installation de PyTorch, torchvision et torchaudio pour CUDA 12.4
run_step "Installation de PyTorch, torchvision et torchaudio pour CUDA 12.4" pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# 8. Installation de WhisperX depuis PyPI
run_step "Installation de WhisperX" pip install whisperx

# 9. Création du wrapper exécutable 'whisperx_cli'
run_step "Création du wrapper exécutable 'whisperx_cli'" bash -c "cat > whisperx_cli << 'EOF'
#!/bin/bash
# Wrapper pour lancer 'whisperx_cli.py' dans l'environnement virtuel

DIR=\"\$( cd \"\$( dirname \"\${BASH_SOURCE[0]}\" )\" && pwd )\"
source \"\$DIR/whisperx_env/bin/activate\"
python \"\$DIR/whisperx_cli.py\" \"\$@\"
EOF"
chmod +x whisperx_cli

# 10. Demander le token Hugging Face (facultatif) pour la diarization
echo -n "Veuillez entrer votre token Hugging Face (pour la diarization) ou appuyez sur Entrée pour l'ignorer : "
read -r HF_TOKEN

# 11. Lancement d'un test final pour télécharger les modèles et traiter 'audio.mp3'
echo -ne "${LOADING} ${BOLD}Test final : transcription sur 'audio.mp3'${RESET} [${LOADING} en cours...]"
if [ -n "$HF_TOKEN" ]; then
  python whisperx_cli.py "audio.mp3" --model large-v3 --language fr --hf_token "$HF_TOKEN" --diarize --output test_output.srt --output_format srt > /dev/null 2>&1
else
  python whisperx_cli.py "audio.mp3" --model large-v3 --language fr --output test_output.srt --output_format srt > /dev/null 2>&1
fi
echo -e "\r${DONE} ${BOLD}Test final : transcription sur 'audio.mp3'${RESET} [${DONE} terminé]"

echo -e "\n${DONE} ${BOLD}Setup complet.${RESET} Vous pouvez lancer vos transcriptions via './whisperx_cli' dans ce dossier."