#!/bin/bash
# Nom du script : setup.sh
# Ce script réalise les opérations suivantes :
# PARTIE 1 : Installation et configuration du CLI WhisperX
#   - Clone le dépôt whisper-x-setup (si nécessaire)
#   - Demande le token HF
#   - Configure l'environnement virtuel, installe les dépendances et crée le wrapper
#   - Lance un test final de transcription
#   - Copie le wrapper dans /usr/local/bin pour une utilisation globale
#
# PARTIE 2 : Installation et lancement de whisper-api
#   - Clone le dépôt whisper-api (si nécessaire)
#   - Installe PM2 (globalement) si besoin
#   - Exécute "npm install" puis "npm run build"
#   - Lance le projet avec PM2 en utilisant UPLOAD_PATH=/tmp
#   - Crée un script de mise à jour (whisper_api_update) pour actualiser l'API
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

# --- PARTIE 1 : Installation de WhisperX CLI ---

echo -e "\n${BOLD}=== Partie 1 : Configuration de WhisperX CLI ===${RESET}"

# URL et nom du dépôt whisper-x-setup
REPO_URL="https://github.com/fchevallieratecna/whisper-x-setup.git"
REPO_DIR="whisper-x-setup"

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

# 5. Activation de l'environnement virtuel (dans ce shell)
echo -ne "${LOADING} ${BOLD}Activation de l'environnement virtuel${RESET} [${LOADING} en cours...]"
source whisperx_env/bin/activate
echo -e "\r${DONE} ${BOLD}Activation de l'environnement virtuel${RESET} [${DONE} terminé]"

# 6. Mise à jour de pip
run_step "Mise à jour de pip" pip install --upgrade pip

# 7. Installation de PyTorch, torchvision et torchaudio pour CUDA 12.4
# (Cette étape peut prendre plusieurs minutes)
run_step "Installation de PyTorch, torchvision et torchaudio pour CUDA 12.4 (peut prendre plusieurs minutes)" \
         pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# 8. Installation de WhisperX depuis PyPI
run_step "Installation de WhisperX" pip install whisperx

# Récupérer le chemin absolu du répertoire cloné
REPO_PATH="$(pwd)"

# Créer le fichier wrapper avec le chemin en dur
run_step "Création du wrapper exécutable 'whisperx_cli'" bash -c "cat <<EOF > whisperx_cli
#!/bin/bash
# Wrapper pour lancer 'whisperx_cli.py' dans l'environnement virtuel
source \"${REPO_PATH}/whisperx_env/bin/activate\"
python \"${REPO_PATH}/whisperx_cli.py\" \"\$@\"
EOF"
chmod +x whisperx_cli

# 10. Déconnexion de l'environnement virtuel pour le test final
deactivate 2>/dev/null

# 11. Lancement du test final via le wrapper (hors du venv courant)
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

# 13. Copier le wrapper dans /usr/local/bin pour le rendre accessible globalement
echo -ne "${LOADING} ${BOLD}Installation globale de 'whisperx_cli'${RESET} [${LOADING} en cours...]"
if [ "$(id -u)" -ne 0 ]; then
  sudo cp whisperx_cli /usr/local/bin/whisperx_cli
else
  cp whisperx_cli /usr/local/bin/whisperx_cli
fi
chmod +x /usr/local/bin/whisperx_cli
WRAPPER_INSTALLED=1
echo -e "\r${DONE} ${BOLD}Installation globale de 'whisperx_cli'${RESET} [${DONE} terminé]"

echo -e "\n${DONE} ${BOLD}Partie 1 terminée.${RESET} Vous pouvez lancer vos transcriptions via la commande 'whisperx_cli'."

# --- PARTIE 2 : Installation et lancement de whisper-api ---

echo -e "\n${BOLD}=== Partie 2 : Configuration de whisper-api ===${RESET}"

# Définir le répertoire du projet API
API_REPO_URL="https://github.com/fchevallieratecna/whisper-api.git"
API_REPO_DIR="whisper-api"

# 14. Cloner le dépôt whisper-api (si nécessaire)
if [ ! -d "$API_REPO_DIR" ]; then
  echo -e "${LOADING} ${BOLD}Clonage du dépôt whisper-api depuis GitHub${RESET} [${LOADING} en cours...]"
  git clone "$API_REPO_URL" > /dev/null 2>&1
  echo -e "\r${DONE} ${BOLD}Clonage du dépôt whisper-api depuis GitHub${RESET} [${DONE} terminé]"
else
  echo -e "${DONE} ${BOLD}Dépôt whisper-api déjà cloné${RESET}"
fi

cd "$API_REPO_DIR" || exit

# Récupérer le chemin absolu du répertoire API
API_PATH="$(pwd)"

# 15. Installation de pm2 globalement (si nécessaire)
echo -ne "${LOADING} ${BOLD}Installation de pm2 (npm install -g pm2)${RESET} [${LOADING} en cours...]"
if ! command -v pm2 >/dev/null 2>&1; then
  sudo npm install -g pm2 > /dev/null 2>&1
fi
echo -e "\r${DONE} ${BOLD}Installation de pm2${RESET} [${DONE} terminé]"

# 16. Installation des dépendances du projet API
run_step "Installation des dépendances de whisper-api (npm install)" npm install

# 17. Construction du projet (npm run build)
run_step "Construction du projet whisper-api (npm run build)" npm run build

# 18. Lancement de whisper-api avec pm2
echo -ne "${LOADING} ${BOLD}Lancement de whisper-api avec pm2 (npm start)${RESET} [${LOADING} en cours...]"
UPLOAD_PATH=/tmp pm2 start npm --name "whisper-api" -- start > /dev/null 2>&1
echo -e "\r${DONE} ${BOLD}Lancement de whisper-api avec pm2${RESET} [${DONE} terminé]"

# 19. Création du script de mise à jour de whisper-api
echo -ne "${LOADING} ${BOLD}Création du script de mise à jour de whisper-api${RESET} [${LOADING} en cours...]"
cat > whisper_api_update << EOF
#!/bin/bash
# Script de mise à jour de whisper-api
API_PATH="${API_PATH}"
cd "\$API_PATH" || exit 1
echo "Mise à jour de whisper-api..."
git pull
npm install
npm run build
pm2 restart whisper-api
EOF
chmod +x whisper_api_update
# Copier le script dans /usr/local/bin pour un accès global
if [ "$(id -u)" -ne 0 ]; then
  sudo cp whisper_api_update /usr/local/bin/whisper_api_update
else
  cp whisper_api_update /usr/local/bin/whisper_api_update
fi
echo -e "\r${DONE} ${BOLD}Création du script de mise à jour de whisper-api${RESET} [${DONE} terminé]"

echo -e "\n${DONE} ${BOLD}Partie 2 terminée.${RESET} Le projet whisper-api est lancé via pm2."
echo -e "\n${DONE} ${BOLD}Setup complet.${RESET} Vous pouvez lancer vos transcriptions avec 'whisperx_cli' et mettre à jour l'API avec 'whisper_api_update'."