#!/bin/bash
# Modern Whisper Setup Script (September 2025)
# Utilise whisper-diarization + NeMo 2.0 + PyTorch 2.7.1 + CUDA 12.8
# Compatible: Ubuntu 20.04+, macOS 12+, CUDA 12.8, Python 3.9-3.12

set -euo pipefail  # Strict error handling

# --- Configuration et constantes ---
readonly SCRIPT_VERSION="2.0.0"
readonly PYTHON_MIN_VERSION="3.9"
readonly PYTHON_MAX_VERSION="3.12"
readonly PYTORCH_VERSION="2.7.1"
readonly CUDA_VERSION="12.8"

# URLs des dépôts
readonly WHISPER_REPO_URL="https://github.com/fchevallieratecna/whisper-x-setup.git"
readonly API_REPO_URL="https://github.com/fchevallieratecna/whisper-api.git"

# --- Variables d'affichage ---
if [[ "$(uname)" == "Darwin" ]]; then
  readonly BOLD="\033[1m"
  readonly RESET="\033[0m"
  readonly GREEN="\033[32m"
  readonly YELLOW="\033[33m"
  readonly RED="\033[31m"
  readonly BLUE="\033[34m"
  readonly LOADING="..."
  readonly DONE="✓"
else
  readonly BOLD="\e[1m"
  readonly RESET="\e[0m"
  readonly GREEN="\e[32m"
  readonly YELLOW="\e[33m"
  readonly RED="\e[31m"
  readonly BLUE="\e[34m"
  readonly LOADING="⏳"
  readonly DONE="✅"
fi

# --- Variables globales ---
VERBOSE=0
ONLY_API=0
FORCE_INSTALL=0
HF_TOKEN=""
NGROK_TOKEN=""
INSTALL_NEMO=1
USE_CONDA=0

# Indicateurs pour rollback
CLONED=0
ENV_CREATED=0
WRAPPER_INSTALLED=0

# --- Fonctions utilitaires ---
log_info() {
    echo -e "${BLUE}ℹ️  ${BOLD}$1${RESET}"
}

log_success() {
    echo -e "${GREEN}${DONE} ${BOLD}$1${RESET}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  ${BOLD}$1${RESET}"
}

log_error() {
    echo -e "${RED}❌ ${BOLD}$1${RESET}"
}

log_step() {
    echo -e "${BLUE}🔄 ${BOLD}$1${RESET}"
}

show_usage() {
    cat << EOF
${BOLD}Modern Whisper Setup Script v${SCRIPT_VERSION} (September 2025)${RESET}

Usage: $0 [OPTIONS]

Options:
  -v, --verbose          Enable verbose output
  --only-api            Install only the API (skip CLI)
  --force               Force reinstallation even if already installed
  --no-nemo             Skip NeMo installation (pyannote only)
  --use-conda           Use conda instead of venv for environment
  --hf-token=TOKEN      Hugging Face token for diarization
  --ngrok-token=TOKEN   Ngrok token for external access
  -h, --help            Show this help message

Examples:
  $0 --verbose --hf-token=hf_xxxx
  $0 --only-api --ngrok-token=xxxx
  $0 --force --no-nemo
EOF
}

# --- Gestion des arguments ---
parse_arguments() {
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
            --force)
                FORCE_INSTALL=1
                shift
                ;;
            --no-nemo)
                INSTALL_NEMO=0
                shift
                ;;
            --use-conda)
                USE_CONDA=1
                shift
                ;;
            --hf-token=*)
                HF_TOKEN="${1#*=}"
                shift
                ;;
            --ngrok-token=*)
                NGROK_TOKEN="${1#*=}"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Option non reconnue: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# --- Vérifications système ---
check_python_version() {
    log_step "Vérification de la version Python"

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 n'est pas installé"
        exit 1
    fi

    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

    local min_version_num
    local max_version_num
    local current_version_num

    min_version_num=$(echo "$PYTHON_MIN_VERSION" | sed 's/\.//')
    max_version_num=$(echo "$PYTHON_MAX_VERSION" | sed 's/\.//')
    current_version_num=$(echo "$python_version" | sed 's/\.//')

    if [[ $current_version_num -lt $min_version_num ]] || [[ $current_version_num -gt $max_version_num ]]; then
        log_error "Python version $python_version n'est pas supportée (requis: $PYTHON_MIN_VERSION-$PYTHON_MAX_VERSION)"
        exit 1
    fi

    log_success "Python $python_version détecté"
}

check_system_requirements() {
    log_step "Vérification des prérequis système"

    # Vérifier Git
    if ! command -v git &> /dev/null; then
        log_error "Git n'est pas installé"
        exit 1
    fi

    # Vérifier FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        log_warning "FFmpeg n'est pas installé - certaines fonctionnalités audio pourraient ne pas fonctionner"
        if [[ "$(uname)" == "Darwin" ]]; then
            log_info "Installez avec: brew install ffmpeg"
        else
            log_info "Installez avec: sudo apt update && sudo apt install ffmpeg"
        fi
    fi

    log_success "Prérequis système vérifiés"
}

check_cuda_compatibility() {
    if [[ "$(uname)" == "Darwin" ]]; then
        log_info "Système macOS détecté - CUDA non nécessaire (utilisation CPU)"
        return 0
    fi

    log_step "Vérification de la compatibilité CUDA"

    if command -v nvidia-smi &> /dev/null; then
        local cuda_version
        cuda_version=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | cut -d' ' -f3 | head -1)

        if [[ -n "$cuda_version" ]]; then
            log_success "CUDA $cuda_version détecté"

            # Vérifier la compatibilité
            local major_version
            major_version=$(echo "$cuda_version" | cut -d'.' -f1)

            if [[ $major_version -ge 11 ]]; then
                log_success "Version CUDA compatible (≥11.0)"
            else
                log_warning "Version CUDA ancienne ($cuda_version) - performance réduite possible"
            fi
        else
            log_warning "CUDA installé mais version non détectable"
        fi
    else
        log_info "CUDA non détecté - utilisation CPU"
    fi
}

# --- Installation des dépendances ---
create_environment() {
    log_step "Création de l'environnement Python"

    local env_name="whisper_modern_env"

    if [[ $USE_CONDA -eq 1 ]]; then
        if ! command -v conda &> /dev/null; then
            log_error "Conda n'est pas installé mais --use-conda spécifié"
            exit 1
        fi

        if conda env list | grep -q "$env_name" && [[ $FORCE_INSTALL -eq 0 ]]; then
            log_warning "Environnement conda '$env_name' existe déjà (utilisez --force pour recréer)"
            conda activate "$env_name"
        else
            [[ $FORCE_INSTALL -eq 1 ]] && conda env remove -n "$env_name" 2>/dev/null || true
            conda create -n "$env_name" python=3.11 -y
            conda activate "$env_name"
        fi
    else
        if [[ -d "$env_name" ]] && [[ $FORCE_INSTALL -eq 0 ]]; then
            log_warning "Environnement venv '$env_name' existe déjà (utilisez --force pour recréer)"
        else
            [[ $FORCE_INSTALL -eq 1 ]] && rm -rf "$env_name" 2>/dev/null || true
            python3 -m venv "$env_name"
        fi

        source "$env_name/bin/activate"
        ENV_CREATED=1
    fi

    # Mise à jour de pip
    python -m pip install --upgrade pip setuptools wheel

    log_success "Environnement Python créé et activé"
}

install_pytorch() {
    log_step "Installation de PyTorch $PYTORCH_VERSION"

    local install_cmd

    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS - CPU uniquement
        install_cmd="pip install torch==$PYTORCH_VERSION torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    else
        # Linux - Vérifier CUDA
        if command -v nvidia-smi &> /dev/null; then
            # CUDA disponible
            local cuda_version
            cuda_version=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | cut -d' ' -f3 | head -1)
            local cuda_major
            cuda_major=$(echo "$cuda_version" | cut -d'.' -f1)

            if [[ $cuda_major -ge 12 ]]; then
                # CUDA 12.x - utiliser cu121 ou cu124
                install_cmd="pip install torch==$PYTORCH_VERSION torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            elif [[ $cuda_major -eq 11 ]]; then
                # CUDA 11.x
                install_cmd="pip install torch==$PYTORCH_VERSION torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            else
                # CUDA trop ancien - utiliser CPU
                install_cmd="pip install torch==$PYTORCH_VERSION torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            fi
        else
            # Pas de CUDA - CPU uniquement
            install_cmd="pip install torch==$PYTORCH_VERSION torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        fi
    fi

    if [[ $VERBOSE -eq 1 ]]; then
        eval "$install_cmd"
    else
        eval "$install_cmd" > /dev/null 2>&1
    fi

    # Vérifier l'installation
    python -c "import torch; print(f'PyTorch {torch.__version__} installé')"

    log_success "PyTorch installé avec succès"
}

install_whisper_dependencies() {
    log_step "Installation des dépendances Whisper modernes"

    local packages=(
        "faster-whisper>=1.1.0"
        "openai-whisper"
        "transformers"
        "librosa"
        "soundfile"
        "pyannote.audio"
        "omegaconf"
    )

    # Installation spécifique pour macOS
    if [[ "$(uname)" == "Darwin" ]]; then
        packages+=(
            "pyOpenSSL==22.0.0"
            "urllib3==1.26.18"
            "ctranslate2>=4.4.0"
        )
    fi

    for package in "${packages[@]}"; do
        if [[ $VERBOSE -eq 1 ]]; then
            pip install "$package"
        else
            pip install "$package" > /dev/null 2>&1
        fi
    done

    log_success "Dépendances Whisper installées"
}

install_nemo_dependencies() {
    if [[ $INSTALL_NEMO -eq 0 ]]; then
        log_info "Installation NeMo ignorée (--no-nemo spécifié)"
        return 0
    fi

    log_step "Installation de NeMo 2.0 pour la diarization avancée"

    # NeMo nécessite des dépendances spécifiques
    local nemo_packages=(
        "Cython"
        "nemo_toolkit[asr]"
        "hydra-core>=1.1"
        "omegaconf>=2.1"
    )

    for package in "${nemo_packages[@]}"; do
        if [[ $VERBOSE -eq 1 ]]; then
            pip install "$package"
        else
            pip install "$package" > /dev/null 2>&1
        fi
    done

    log_success "NeMo 2.0 installé pour diarization avancée"
}

# --- Création du wrapper CLI ---
create_modern_wrapper() {
    log_step "Création du wrapper CLI moderne"

    local abs_path
    abs_path="$(pwd)"

    cat > whisper_modern_cli << EOF
#!/bin/bash
# Modern Whisper CLI Wrapper (September 2025)

if [[ "\$1" == "--version" ]]; then
    echo "Modern Whisper CLI v${SCRIPT_VERSION}"
    echo "Based on whisper-diarization + NeMo 2.0"
    exit 0
fi

# Activer l'environnement
source "${abs_path}/whisper_modern_env/bin/activate"

# Lancer le CLI moderne
python "${abs_path}/whisper_diarize_cli.py" "\$@"
EOF

    chmod +x whisper_modern_cli

    # Installation globale
    if [[ "$(id -u)" -ne 0 ]]; then
        sudo cp whisper_modern_cli /usr/local/bin/whisper_modern_cli
    else
        cp whisper_modern_cli /usr/local/bin/whisper_modern_cli
    fi

    chmod +x /usr/local/bin/whisper_modern_cli
    WRAPPER_INSTALLED=1

    log_success "Wrapper CLI installé globalement"
}

# --- Test du système ---
test_installation() {
    log_step "Test de l'installation"

    # Vérifier que le CLI fonctionne
    if ./whisper_modern_cli --version > /dev/null 2>&1; then
        log_success "CLI fonctionne correctement"
    else
        log_error "Problème avec le CLI"
        return 1
    fi

    # Test rapide avec un fichier audio (si disponible)
    if [[ -f "audio.mp3" ]]; then
        log_step "Test de transcription sur audio.mp3"

        local test_cmd="./whisper_modern_cli audio.mp3 --model base --output test_output.txt --output_format txt"

        if [[ "$(uname)" == "Darwin" ]]; then
            test_cmd+=" --compute_type int8"
        fi

        if [[ -n "$HF_TOKEN" ]]; then
            test_cmd+=" --hf_token '$HF_TOKEN'"
        else
            test_cmd+=" --no-diarize"
        fi

        if [[ $VERBOSE -eq 1 ]]; then
            eval "$test_cmd"
        else
            eval "$test_cmd" > /dev/null 2>&1
        fi

        if [[ -f "test_output.txt" ]] && [[ -s "test_output.txt" ]]; then
            log_success "Test de transcription réussi"
            rm -f test_output.txt
        else
            log_warning "Test de transcription échoué"
        fi
    fi
}

# --- Installation de l'API ---
setup_api() {
    if [[ $ONLY_API -eq 0 ]]; then
        echo
        log_info "=== Configuration de l'API Whisper ==="
    fi

    # Retourner au répertoire parent si nécessaire
    if [[ "$(basename "$(pwd)")" == "whisper-x-setup" ]]; then
        cd ..
    fi

    # Cloner l'API
    if [[ ! -d "whisper-api" ]]; then
        log_step "Clonage du dépôt whisper-api"
        git clone "$API_REPO_URL" > /dev/null 2>&1
        log_success "Dépôt API cloné"
    else
        log_info "Dépôt API déjà présent"
    fi

    cd whisper-api || exit 1

    # Vérifier Node.js
    if ! command -v node &> /dev/null; then
        log_step "Installation de Node.js via nvm"
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

        nvm install 20
        nvm alias default 20
        nvm use 20

        log_success "Node.js installé"
    fi

    # Installer PM2
    if ! command -v pm2 &> /dev/null; then
        log_step "Installation de PM2"
        npm install -g pm2 > /dev/null 2>&1
        log_success "PM2 installé"
    fi

    # Installer les dépendances et build
    log_step "Installation des dépendances API"
    npm install > /dev/null 2>&1

    log_step "Build de l'API"
    npm run build > /dev/null 2>&1

    # Lancement avec PM2
    log_step "Lancement de l'API avec PM2"
    local api_port=${API_PORT:-3000}
    UPLOAD_PATH=/tmp PORT=$api_port pm2 start npm --name "whisper-api-modern" -- start > /dev/null 2>&1

    log_success "API lancée sur le port $api_port"

    # Configuration Ngrok si token fourni
    if [[ -n "$NGROK_TOKEN" ]]; then
        setup_ngrok "$api_port"
    fi
}

setup_ngrok() {
    local port=$1
    log_step "Configuration de ngrok"

    if ! command -v ngrok &> /dev/null; then
        # Installation de ngrok
        if [[ "$(uname)" == "Darwin" ]]; then
            brew install ngrok > /dev/null 2>&1
        else
            curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
            echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
            sudo apt update && sudo apt install ngrok > /dev/null 2>&1
        fi
    fi

    # Configuration du token
    ngrok config add-authtoken "$NGROK_TOKEN" > /dev/null 2>&1

    # Démarrage du tunnel
    nohup ngrok http "$port" > /dev/null 2>&1 &
    local ngrok_pid=$!

    sleep 3

    # Récupérer l'URL publique
    local public_url
    public_url=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*"' | cut -d'"' -f4 | grep https)

    if [[ -n "$public_url" ]]; then
        log_success "Tunnel ngrok actif: $public_url"
    else
        log_warning "Problème avec le tunnel ngrok"
    fi
}

# --- Nettoyage en cas d'erreur ---
cleanup() {
    log_warning "Interruption détectée - nettoyage en cours"

    if [[ $ENV_CREATED -eq 1 ]]; then
        rm -rf whisper_modern_env
        log_info "Environnement supprimé"
    fi

    if [[ $CLONED -eq 1 ]]; then
        cd .. && rm -rf whisper-x-setup
        log_info "Dépôt supprimé"
    fi

    if [[ $WRAPPER_INSTALLED -eq 1 ]]; then
        sudo rm -f /usr/local/bin/whisper_modern_cli 2>/dev/null || true
        log_info "Wrapper désinstallé"
    fi

    exit 1
}

# --- Affichage du résumé ---
show_summary() {
    echo
    log_success "=== Installation terminée ===="
    echo
    echo -e "${BOLD}🎯 Commandes disponibles :${RESET}"
    echo "   • ${BOLD}whisper_modern_cli${RESET} - CLI moderne avec diarization avancée"
    echo "   • ${BOLD}whisper_modern_cli --version${RESET} - Informations sur la version"
    echo
    echo -e "${BOLD}📝 Exemples d'utilisation :${RESET}"
    echo "   • Transcription simple:"
    echo "     ${BLUE}whisper_modern_cli audio.mp3 --model large-v3 --language fr${RESET}"
    echo
    echo "   • Avec diarization:"
    echo "     ${BLUE}whisper_modern_cli audio.mp3 --diarize --hf_token YOUR_TOKEN --nb_speaker 2${RESET}"
    echo
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "   • Sur macOS:"
        echo "     ${BLUE}whisper_modern_cli audio.mp3 --compute_type int8 --device cpu${RESET}"
        echo
    fi

    echo -e "${BOLD}🚀 Nouveautés 2025 :${RESET}"
    echo "   ✅ PyTorch $PYTORCH_VERSION avec CUDA 12.8"
    echo "   ✅ faster-whisper optimisé"
    echo "   ✅ pyannote Community-1 diarization"
    if [[ $INSTALL_NEMO -eq 1 ]]; then
        echo "   ✅ NeMo 2.0 pour diarization avancée"
    fi
    echo "   ✅ Support Python 3.9-3.12"
    echo "   ✅ Interface CLI modernisée"
    echo
}

# --- Fonction principale ---
main() {
    trap cleanup SIGINT

    echo -e "${BOLD}🎤 Modern Whisper Setup v${SCRIPT_VERSION} (September 2025)${RESET}"
    echo -e "${BOLD}Basé sur whisper-diarization + NeMo 2.0 + PyTorch $PYTORCH_VERSION${RESET}"
    echo

    parse_arguments "$@"

    # Demander les tokens si non fournis
    if [[ -z "$HF_TOKEN" ]] && [[ $ONLY_API -eq 0 ]]; then
        read -p "Token Hugging Face (pour diarization, optionnel): " HF_TOKEN
    fi

    if [[ -z "$NGROK_TOKEN" ]]; then
        read -p "Token ngrok (pour accès externe, optionnel): " NGROK_TOKEN
    fi

    # Vérifications système
    check_python_version
    check_system_requirements
    check_cuda_compatibility

    if [[ $ONLY_API -eq 0 ]]; then
        echo
        log_info "=== Installation du CLI Whisper moderne ==="

        # Cloner si nécessaire
        if [[ ! -d "whisper-x-setup" ]]; then
            log_step "Clonage du dépôt whisper-x-setup"
            git clone "$WHISPER_REPO_URL" > /dev/null 2>&1
            CLONED=1
            log_success "Dépôt cloné"
        fi

        cd whisper-x-setup || exit 1

        # Vérifier la présence du nouveau CLI
        if [[ ! -f "whisper_diarize_cli.py" ]]; then
            log_error "Le fichier whisper_diarize_cli.py est introuvable"
            exit 1
        fi

        # Installation
        create_environment
        install_pytorch
        install_whisper_dependencies
        install_nemo_dependencies
        create_modern_wrapper
        test_installation

        log_success "Installation du CLI terminée"
    fi

    # Installation de l'API
    setup_api

    # Résumé final
    show_summary
}

# --- Point d'entrée ---
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi