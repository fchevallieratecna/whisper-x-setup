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
#   - Installe PM2 globalement via npm et met à jour le PATH pour s'assurer que PM2 est accessible
#   - Exécute "npm install" puis "npm run build"
#   - Lance le projet avec PM2 en utilisant UPLOAD_PATH=/tmp
#   - Crée un script de mise à jour (whisper_api_update) pour actualiser l'API
#
# Usage : ./setup.sh [-v]
#   -v : mode verbose (affiche les sorties des commandes)
#
# Rendre exécutable avec : chmod +x setup.sh

# --- Variables d'affichage ---
if [[ "$(uname)" == "Darwin" ]]; then
  BOLD="\033[1m"
  RESET="\033[0m"
  LOADING="..."
  DONE="✓"
else
  BOLD="\e[1m"
  RESET="\e[0m"
  LOADING="⏳"
  DONE="✅"
fi

# --- Mode verbose et options ---
VERBOSE=0
ONLY_API=0

# Traitement des arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    --only-api)
      ONLY_API=1
      shift
      ;;
    *)
      echo "Option non reconnue: $1"
      echo "Usage: ./setup.sh [-v|--verbose] [--only-api]"
      exit 1
      ;;
  esac
done

# --- Indicateurs pour rollback ---
CLONED=0
ENV_CREATED=0
WRAPPER_INSTALLED=0

# --- Fonction de vérification de libcudnn ---
check_libcudnn() {
  # Vérifier si on est sur macOS
  if [[ "$(uname)" == "Darwin" ]]; then
    echo -e "${DONE} ${BOLD}Système macOS détecté, vérification libcudnn ignorée${RESET}"
    return 0
  fi
  
  echo -ne "${LOADING} ${BOLD}Vérification de libcudnn_ops_infer.so.8${RESET} [${LOADING} en cours...]"
  if ldconfig -p | grep -q "libcudnn_ops_infer.so.8"; then
    echo -e "\r${DONE} ${BOLD}libcudnn_ops_infer.so.8 trouvé${RESET}"
  else
    echo -e "\r❌ ${BOLD}libcudnn_ops_infer.so.8 non trouvé${RESET}"
    echo "Veuillez installer libcudnn (par exemple, 'apt update && apt install -y libcudnn8 libcudnn8-dev libcudnn8-samples') ou configurer LD_LIBRARY_PATH pour inclure le dossier contenant libcudnn_ops_infer.so.8."
    exit 1
  fi
}

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
  # Vérifier si on est sur macOS
  if [[ "$(uname)" == "Darwin" ]]; then
    echo -e "${DONE} ${BOLD}Système macOS détecté, vérification CUDA ignorée${RESET}"
    return 0
  fi

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
if [ $ONLY_API -eq 0 ]; then
  echo -e "\n${BOLD}=== Partie 1 : Configuration de WhisperX CLI ===${RESET}"

  # --- Demande du token Hugging Face dès le début ---
  read -p "Veuillez entrer votre token Hugging Face (pour la diarization) ou appuyez sur Entrée pour l'ignorer : " HF_TOKEN


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

  # 3.5. Vérification de libcudnn
  check_libcudnn

  # 4. Création de l'environnement virtuel
  run_step "Création de l'environnement virtuel 'whisperx_env'" python3 -m venv whisperx_env
  ENV_CREATED=1

  # 5. Activation de l'environnement virtuel (dans ce shell)
  echo -ne "${LOADING} ${BOLD}Activation de l'environnement virtuel${RESET} [${LOADING} en cours...]"
  source whisperx_env/bin/activate
  echo -e "\r${DONE} ${BOLD}Activation de l'environnement virtuel${RESET} [${DONE} terminé]"

  # 6. Mise à jour de pip
  run_step "Mise à jour de pip" pip install --upgrade pip

  # 7. Installation de PyTorch, torchvision et torchaudio selon le système
  if [[ "$(uname)" == "Darwin" ]]; then
    run_step "Installation de PyTorch et torchaudio pour macOS" \
             pip install torch torchaudio
  else
    run_step "Installation de PyTorch, torchvision et torchaudio pour CUDA 12.4 (peut prendre plusieurs minutes)" \
             pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
  fi

  # 8. Installation de WhisperX depuis PyPI et dépendances supplémentaires pour macOS
  if [[ "$(uname)" == "Darwin" ]]; then
    run_step "Installation de pyOpenSSL compatible pour macOS" pip install pyOpenSSL==22.0.0
    run_step "Installation d'urllib3 compatible" pip install urllib3==1.26.6
    run_step "Installation de PyTorch et torchaudio pour macOS" pip install torch torchaudio
    run_step "Installation de transformers" pip install transformers
    run_step "Installation de WhisperX avec dépendances pour macOS" pip install whisperx
    run_step "Installation de nltk" pip install nltk
    run_step "Réinstallation de CTranslate2 compatible" pip install --force-reinstall ctranslate2==4.4.0
  else
    run_step "Installation de WhisperX" pip install whisperx
  fi

  # Récupérer le chemin absolu du répertoire cloné
  ABS_PATH="$(pwd)"

  # 9. Création du wrapper exécutable 'whisperx_cli'
  run_step "Création du wrapper exécutable 'whisperx_cli'" bash -c "cat <<EOF > whisperx_cli
#!/bin/bash
source \"${ABS_PATH}/whisperx_env/bin/activate\"
python \"${ABS_PATH}/whisperx_cli.py\" \"\\\$@\"
EOF"
  chmod +x whisperx_cli

  # 10. Déconnexion de l'environnement virtuel pour le test final
  deactivate 2>/dev/null

  # 11. Lancement du test final via le wrapper (hors du venv courant)
  echo -ne "${LOADING} ${BOLD}Test final : transcription sur 'audio.mp3'${RESET} [${LOADING} en cours...]"

  # Définir les options de compute_type selon le système
  if [[ "$(uname)" == "Darwin" ]]; then
    COMPUTE_TYPE="--compute_type int8"
  else
    COMPUTE_TYPE=""
  fi

  # Construire la commande complète
  if [ -n "$HF_TOKEN" ]; then
    CMD="./whisperx_cli audio.mp3 --model large-v3 --language fr --hf_token \"$HF_TOKEN\" --diarize --output test_output.srt --output_format srt --nb_speaker 1 $COMPUTE_TYPE"
  else
    CMD="./whisperx_cli audio.mp3 --model large-v3 --language fr --output test_output.srt --output_format srt --nb_speaker 1 $COMPUTE_TYPE"
  fi

  # Afficher la commande
  echo -e "\n${BOLD}Exécution de la commande :${RESET} $CMD"

  # Exécuter la commande
  if [ $VERBOSE -eq 1 ]; then
    eval "$CMD"
  else
    eval "$CMD > /dev/null 2>&1"
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

  # Afficher un message spécifique à la fin selon le système
  if [[ "$(uname)" == "Darwin" ]]; then
    echo -e "\n${DONE} ${BOLD}Setup complet sur macOS.${RESET}"
    echo -e "${BOLD}Note importante:${RESET} Sur macOS, utilisez toujours l'option ${BOLD}--compute_type int8${RESET} avec whisperx_cli."
    echo -e "Exemple: ${BOLD}whisperx_cli audio.mp3 --compute_type int8 --model large-v3 --language fr${RESET}"
  else
    echo -e "\n${DONE} ${BOLD}Setup complet.${RESET} Vous pouvez lancer vos transcriptions avec 'whisperx_cli' et mettre à jour l'API avec 'whisper_api_update'."
  fi
else
  echo -e "\n${BOLD}=== Option --only-api détectée, passage directement à la partie 2 ===${RESET}"
  
  # Si on est dans le répertoire whisper-x-setup, on remonte d'un niveau
  if [ -n "$(pwd | grep -o "whisper-x-setup")" ]; then
    cd ..
  fi
fi

# --- PARTIE 2 : Installation et lancement de whisper-api ---
echo -e "\n${BOLD}=== Partie 2 : Configuration de whisper-api ===${RESET}"

# Vérifier que Node.js est installé
if ! command -v node >/dev/null 2>&1; then
  echo -e "\n❌ ${BOLD}Erreur : Node.js n'est pas installé. Veuillez installer Node.js et npm avant de poursuivre.${RESET}"
  echo -e "\n${BOLD}Installation de Node.js et npm via nvm${RESET}"
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
  echo -e "\n${DONE} ${BOLD}Installation de Node.js et npm via nvm terminée.${RESET}"
  # Installation de node 22 + alias default
  nvm install 22
  nvm alias default 22
  echo -e "\n${DONE} ${BOLD}Installation de Node.js et npm via nvm terminée.${RESET}"
  echo -e "\n${BOLD}Veuillez fermer et rouvrir le terminal, puis relancer le script.${RESET}"
  exit 1
fi
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
NPM_BIN="$(which npm)"
if [ -z "$NPM_BIN" ]; then
  echo -e "\n❌ ${BOLD}npm n'est pas trouvé. Veuillez installer Node.js et npm.${RESET}"
  exit 1
fi

if ! command -v pm2 >/dev/null 2>&1; then
  # Si npm est installé via nvm, il est généralement dans un chemin contenant "nvm"
  if echo "$NPM_BIN" | grep -qi "nvm"; then
    "$NPM_BIN" install -g pm2
  else
    sudo "$NPM_BIN" install -g pm2
  fi
fi

# Récupérer le répertoire global des binaires npm et mettre à jour le PATH
GLOBAL_NPM_BIN="$("$NPM_BIN" config get prefix)/bin"
export PATH="$PATH:$GLOBAL_NPM_BIN"

if ! command -v pm2 >/dev/null 2>&1; then
  echo -e "\n❌ ${BOLD}pm2 n'est toujours pas trouvé dans le PATH. Veuillez vérifier l'installation de pm2.${RESET}"
  exit 1
fi
echo -e "\r${DONE} ${BOLD}Installation de pm2${RESET} [${DONE} terminé]"


# 16. Installation des dépendances du projet API
run_step "Installation des dépendances de whisper-api (npm install)" npm install

# 17. Construction du projet (npm run build)
run_step "Construction du projet whisper-api (npm run build)" npm run build

# 18. Lancement de whisper-api avec pm2 (en définissant UPLOAD_PATH=/tmp)
echo -ne "${LOADING} ${BOLD}Lancement de whisper-api avec pm2 (npm start)${RESET} [${LOADING} en cours...]"
UPLOAD_PATH=/tmp pm2 start npm --name "whisper-api" -- start > /dev/null 2>&1
echo -e "\r${DONE} ${BOLD}Lancement de whisper-api avec pm2${RESET} [${DONE} terminé]"

# 19. Création du script de mise à jour de whisper-api
echo -ne "${LOADING} ${BOLD}Création du script de mise à jour de whisper-api${RESET} [${LOADING} en cours...]"
cat <<EOF > whisper_api_update
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
if [ "$(id -u)" -ne 0 ]; then
  sudo cp whisper_api_update /usr/local/bin/whisper_api_update
else
  cp whisper_api_update /usr/local/bin/whisper_api_update
fi
echo -e "\r${DONE} ${BOLD}Création du script de mise à jour de whisper-api${RESET} [${DONE} terminé]"

echo -e "\n${DONE} ${BOLD}Partie 2 terminée.${RESET} Le projet whisper-api est lancé via pm2."
echo -e "\n${DONE} ${BOLD}Setup complet.${RESET} Vous pouvez lancer vos transcriptions avec 'whisperx_cli' et mettre à jour l'API avec 'whisper_api_update'."