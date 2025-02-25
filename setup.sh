#!/bin/bash
# Nom du script : setup.sh
# Ce script clone le dépôt (si nécessaire), demande le token HF, configure l'environnement virtuel,
# installe les dépendances, crée le wrapper, lance un test final et, en cas d'interruption (Ctrl+C),
# annule les changements effectués.
#
# Usage : ./setup.sh [-v]
#   -v : mode verbose (affiche les sorties des commandes)
#
# Rendre exécutable avec : chmod +x setup.sh

# --- Variables d'affichage ---
BOLD="\e[1m"
RESET="\e[0m"
LOADING="⏳"
DONE="✅"

# --- Mode verbose ---
VERBOSE=0
if [ "$1" == "-v" ]; then
  VERBOSE=1
fi

# --- Indicateurs pour rollback ---
CLONED=0
ENV_CREATED=0
WRAPPER_INSTALLED=0

# --- Fonction de nettoyage en cas d'interruption ---
cleanup() {
  echo -e "\n${BOLD}Interruption détectée, annulation de l'installation...${RESET}"
  if [ $ENV_CREATED -eq 1 ]; then
    echo "Suppression de l'environnement virtuel..."
    rm -rf whisperx_env
  fi
  if [ $CLONED -eq 1 ]; then
    cd .. && rm -rf "$REPO_DIR"
    echo "Suppression du dépôt cloné..."
  fi
  if [ $WRAPPER_INSTALLED -eq 1 ]; then
    echo "Suppression du wrapper installé globalement..."
    if [ "$(id -u)" -ne 0 ]; then
      sudo rm -f /usr/local/bin/whisperx_cli
    else
      rm -f /usr/local/bin/whisperx_cli
    fi
  fi
  exit 1
}
trap cleanup SIGINT

# --- Demande du token Hugging Face dès le début ---
read -p "Veuillez entrer votre token Hugging Face (pour la diarization) ou appuyez sur Entrée pour l'ignorer : " HF_TOKEN

# --- Fonction de log et d'exécution d'une étape ---
run_step() {
  local description="$1"
  shift
  echo -ne "${LOADING} ${BOLD}${description}${RESET} [${LOADING} en cours...]"
  if [ $VERBOSE -eq 1 ]; then
    "$@"
  else
    "$@" > /dev/null 2>&1
  fi
  echo -e "\r${DONE} ${BOLD}${description}${RESET} [${DONE} terminé]"
}

# --- Vérification de CUDA 12.4 via nvcc et nvidia-smi ---
check_cuda() {
  echo -ne "${LOADING} ${BOLD}Vérification de CUDA 12.4${RESET} [${LOADING} en cours...]"
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep -o "release [0-9]*\.[0-9]*" | head -n1 | cut -d' ' -f2)
  elif [ -f "/usr/local/cuda/version.txt" ]; then
    CUDA_VERSION=$(grep -o "[0-9]*\.[0-9]*" /usr/local/cuda/version.txt | head -n1)
  else
    CUDA_VERSION=""
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo -e "\r❌ ${BOLD}Vérification de CUDA 12.4${RESET} [${BOLD}ERREUR${RESET}]"
    echo "nvidia-smi ne fonctionne pas ou n'est pas installé."
    exit 1
  fi

  if [[ "$CUDA_VERSION" == "12.4" ]]; then
    echo -e "\r${DONE} ${BOLD}Vérification de CUDA 12.4${RESET} [${DONE} présent (version ${CUDA_VERSION})]"
  else
    echo -e "\r❌ ${BOLD}Vérification de CUDA 12.4${RESET} [${BOLD}ERREUR${RESET}]"
    echo "CUDA 12.4 n'est pas installé sur cette machine (trouvé: '$CUDA_VERSION')."
    exit 1
  fi
}

# --- URL et nom du dépôt à cloner ---
REPO_URL="https://github.com/fchevallieratecna/whisper-x-setup.git"
REPO_DIR="whisper-x-setup"

# --- Début du script ---

# 1. Cloner le dépôt (si nécessaire)
if [ ! -d "$REPO_DIR" ]; then
  echo -e "${LOADING} ${BOLD}Clonage du dépôt depuis GitHub${RESET} [${LOADING} en cours...]"
  git clone "$REPO_URL" > /dev/null 2>&1
  CLONED=1
  echo -e "\r${DONE} ${BOLD}Clonage du dépôt depuis GitHub${RESET} [${DONE} terminé]"
else
  echo -e "${DONE} ${BOLD}Dépôt déjà cloné${RESET}"
fi

cd "$REPO_DIR" || exit

# 2. Vérifier la présence du fichier Python
if [ ! -f "whisperx_cli.py" ]; then
  echo -e "\n❌ ${BOLD}Erreur${RESET}: Le fichier 'whisperx_cli.py' est introuvable dans $(pwd)."
  echo "Veuillez vérifier la structure de votre dépôt."
  exit 1
else
  echo -e "${DONE} ${BOLD}Fichier 'whisperx_cli.py' trouvé${RESET}"
fi

# 3. Vérification de CUDA 12.4 et de nvidia-smi
check_cuda

# 4. Création de l'environnement virtuel
run_step "Création de l'environnement virtuel 'whisperx_env'" python3 -m venv whisperx_env
ENV_CREATED=1

# 5. Activation de l'environnement virtuel (reste dans le shell courant)
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

# 10. Déconnexion de l'environnement virtuel pour le test final
deactivate 2>/dev/null

# 11. Lancement du test final via le wrapper fraîchement créé (hors du venv courant)
echo -ne "${LOADING} ${BOLD}Test final : transcription sur 'audio.mp3'${RESET} [${LOADING} en cours...]"
if [ -n "$HF_TOKEN" ]; then
  ./whisperx_cli audio.mp3 --model large-v3 --language fr --hf_token "$HF_TOKEN" --diarize --output test_output.srt --output_format srt > /dev/null 2>&1
else
  ./whisperx_cli audio.mp3 --model large-v3 --language fr --output test_output.srt --output_format srt > /dev/null 2>&1
fi
echo -e "\r${DONE} ${BOLD}Test final : transcription sur 'audio.mp3'${RESET} [${DONE} terminé]"

# 12. Vérification finale du fichier de sous-titres
if [ -s "test_output.srt" ]; then
  echo -e "${DONE} ${BOLD}Fichier de sous-titres 'test_output.srt' créé et non vide.${RESET}"
else
  echo -e "❌ ${BOLD}Erreur${RESET}: Le fichier 'test_output.srt' n'existe pas ou est vide."
  exit 1
fi

# 13. Copier le wrapper dans /usr/local/bin pour le rendre accessible de partout
echo -ne "${LOADING} ${BOLD}Installation globale de 'whisperx_cli'${RESET} [${LOADING} en cours...]"
if [ "$(id -u)" -ne 0 ]; then
  sudo cp whisperx_cli /usr/local/bin/whisperx_cli
else
  cp whisperx_cli /usr/local/bin/whisperx_cli
fi
chmod +x /usr/local/bin/whisperx_cli
WRAPPER_INSTALLED=1
echo -e "\r${DONE} ${BOLD}Installation globale de 'whisperx_cli'${RESET} [${DONE} terminé]"

# --- Documentation d'utilisation ---
echo -e "\n${BOLD}Documentation d'utilisation de 'whisperx_cli':${RESET}"
echo "---------------------------------------------------------"
echo "Syntaxe de base :"
echo "  whisperx_cli [audio_file] [OPTIONS]"
echo ""
echo "Paramètres :"
echo "  audio_file          : Chemin vers le fichier audio (ex : audio.mp3)"
echo "  --model MODEL       : Modèle WhisperX à utiliser (default : large-v3)"
echo "  --language LANG     : Code langue (ex : fr, en, etc.). Si non spécifié, le langage est détecté automatiquement."
echo "  --hf_token TOKEN    : Token Hugging Face (requis pour activer la diarization)"
echo "  --diarize           : Active la diarization (nécessite --hf_token)"
echo "  --batch_size N      : Taille du batch pour la transcription (default : 4)"
echo "  --compute_type TYPE : Type de calcul (ex : float16 pour GPU, int8 pour réduire l'utilisation de la mémoire GPU)"
echo "  --output FILE       : Fichier de sortie (default : transcription.json)"
echo "  --output_format FMT : Format de sortie : json, txt ou srt (default : json)"
echo ""
echo "Exemple d'utilisation :"
echo "  whisperx_cli audio.mp3 --model large-v3 --language fr --hf_token YOUR_TOKEN --diarize --output sous_titres.srt --output_format srt"
echo "---------------------------------------------------------"
echo -e "\n${DONE} ${BOLD}Setup complet.${RESET} Vous pouvez lancer vos transcriptions depuis n'importe où avec la commande 'whisperx_cli'."