#!/bin/bash
# Modern Whisper Setup Script (September 2025)
# Utilise whisper-diarization + NeMo 2.0 + PyTorch 2.7.1 + CUDA 12.8
# Compatible: Ubuntu 20.04+, macOS 12+, CUDA 12.8, Python 3.9-3.12

set -euo pipefail  # Strict error handling

# --- Configuration et constantes ---
readonly SCRIPT_VERSION="2.0.0"
readonly PYTHON_MIN_VERSION="3.9"
readonly PYTHON_MAX_VERSION="3.13"
readonly PYTORCH_VERSION="2.8.0"
readonly CUDA_VERSION="12.9"

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
        install_cmd="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        log_info "🍎 Système macOS détecté - utilisation version CPU"
    else
        # Linux - Vérifier CUDA
        if command -v nvidia-smi &> /dev/null; then
            # CUDA disponible
            local cuda_version
            cuda_version=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | cut -d' ' -f3 | head -1)
            local cuda_major
            cuda_major=$(echo "$cuda_version" | cut -d'.' -f1)

            log_info "🔧 CUDA $cuda_version détecté (version majeure: $cuda_major)"

            if [[ $cuda_major -ge 12 ]]; then
                # CUDA 12.x - utiliser cu129 pour PyTorch 2.8.0 (dernière version officielle)
                install_cmd="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129"
                log_info "🚀 Utilisation de PyTorch 2.8.0 avec CUDA 12.x (cu129)"
            elif [[ $cuda_major -eq 11 ]]; then
                # CUDA 11.x
                install_cmd="pip install torch==$PYTORCH_VERSION torchvision==0.18.0 torchaudio==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/cu118"
                log_info "🚀 Utilisation de PyTorch avec CUDA 11.x (cu118)"
            else
                # CUDA trop ancien - utiliser CPU
                install_cmd="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
                log_warning "⚠️  CUDA trop ancien ($cuda_version) - utilisation version CPU"
            fi
        else
            # Pas de CUDA - CPU uniquement
            install_cmd="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            log_info "💻 Pas de CUDA détecté - utilisation version CPU"
        fi
    fi

    log_info "📦 Commande d'installation: $install_cmd"
    
    # Toujours afficher les logs pour le debugging
    log_step "Exécution de l'installation PyTorch..."
    if ! eval "$install_cmd"; then
        log_error "❌ Échec de l'installation PyTorch"
        log_info "💡 Tentative avec une version alternative..."
        
        # Fallback vers une version plus ancienne et stable
        local fallback_cmd="pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"
        log_info "📦 Commande fallback: $fallback_cmd"
        
        if ! eval "$fallback_cmd"; then
            log_error "❌ Échec de l'installation PyTorch (même avec fallback)"
            exit 1
        fi
    fi

    # Vérifier l'installation
    log_step "Vérification de l'installation PyTorch..."
    if python -c "import torch; print(f'✅ PyTorch {torch.__version__} installé avec succès')" 2>/dev/null; then
        log_success "PyTorch installé avec succès"
    else
        log_error "❌ PyTorch installé mais non importable"
        log_info "🔍 Diagnostic:"
        python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "❌ Import torch échoué"
        exit 1
    fi

    # Configuration cuDNN pour compatibilité GPU
    if command -v nvidia-smi &> /dev/null; then
        log_step "Configuration cuDNN pour compatibilité GPU"
        
        # Vérifier la version cuDNN installée
        local cudnn_version
        if python -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')" 2>/dev/null; then
            cudnn_version=$(python -c "import torch; print(torch.backends.cudnn.version())" 2>/dev/null)
            log_info "🔧 cuDNN version détectée: $cudnn_version"
        fi
        
        # Configurer les variables d'environnement pour éviter les conflits
        log_info "🔧 Configuration des variables d'environnement cuDNN"
        
        # Définir le chemin de l'environnement
        local env_path
        if [[ $USE_CONDA -eq 1 ]]; then
            env_path="$(conda info --base)/envs/whisper_modern_env"
        else
            env_path="$(pwd)/whisper_modern_env"
        fi
        
        # Créer le script d'environnement
        cat > "$env_path/bin/setup_cuda_env.sh" << 'EOF'
#!/bin/bash
# Configuration automatique CUDA/cuDNN pour Whisper
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Chemin vers les bibliothèques cuDNN de PyTorch
CUDNN_LIB_PATH="$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)"
if [ -d "$CUDNN_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB_PATH:$LD_LIBRARY_PATH"
fi

# Chemin vers les bibliothèques NVIDIA dans l'environnement
NVIDIA_LIB_PATH="$(find $(python -c "import site; print(site.getsitepackages()[0])") -name "nvidia" -type d 2>/dev/null | head -1)"
if [ -d "$NVIDIA_LIB_PATH" ]; then
    for lib_dir in "$NVIDIA_LIB_PATH"/*/lib; do
        if [ -d "$lib_dir" ]; then
            export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
        fi
    done
fi

# Nettoyer les doublons dans LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//')
EOF

        chmod +x "$env_path/bin/setup_cuda_env.sh"
        log_success "Script de configuration CUDA créé"
        
        # Ajouter au script d'activation
        echo "source \"\$(dirname \"\$BASH_SOURCE\")/setup_cuda_env.sh\"" >> "$env_path/bin/activate"
        log_success "Configuration CUDA ajoutée à l'activation de l'environnement"
    fi
}

install_whisper_dependencies() {
    log_step "Installation des dépendances Whisper-Diarization"

    # Contraintes importantes selon whisper-diarization
    log_step "Installation des contraintes de base"
    if ! pip install "numpy<2"; then
        log_error "❌ Échec de l'installation de numpy<2"
        exit 1
    fi

    local packages=(
        "faster-whisper>=1.1.0"
        "nltk"
        "librosa"
        "soundfile"
        "omegaconf"
        "pyannote.audio==4.0.0"
    )

    # Installation spécifique pour macOS
    if [[ "$(uname)" == "Darwin" ]]; then
        packages+=(
            "pyOpenSSL==22.0.0"
            "urllib3==1.26.18"
            "ctranslate2>=4.4.0"
        )
        log_info "🍎 Ajout des dépendances spécifiques macOS"
    else
        # Linux - Ajouter cuDNN pour compatibilité pyannote
        packages+=(
            "nvidia-cudnn-cu12"
        )
        log_info "🐧 Ajout de cuDNN pour Linux"
    fi

    log_info "📦 Installation de ${#packages[@]} packages Whisper-Diarization..."

    for package in "${packages[@]}"; do
        log_step "Installation de $package"
        if ! pip install "$package"; then
            log_error "❌ Échec de l'installation de $package"
            log_info "💡 Tentative de continuer avec les autres packages..."
        else
            log_info "✅ $package installé"
        fi
    done

    # Installation des dépôts Git spécifiques
    log_step "Installation des dépendances Git spécifiques"
    local git_packages=(
        "git+https://github.com/MahmoudAshraf97/demucs.git"
        "git+https://github.com/oliverguhr/deepmultilingualpunctuation.git"
        "git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git"
        "git+https://github.com/AI4Bharat/indic-numtowords.git"
    )

    for git_package in "${git_packages[@]}"; do
        log_step "Installation de $git_package"
        if ! pip install "$git_package"; then
            log_warning "⚠️  Échec de l'installation de $git_package (optionnel)"
        else
            log_info "✅ $git_package installé"
        fi
    done

    log_success "Installation des dépendances Whisper-Diarization terminée"
}

install_nemo_dependencies() {
    if [[ $INSTALL_NEMO -eq 0 ]]; then
        log_info "Installation NeMo ignorée (--no-nemo spécifié)"
        return 0
    fi

    log_step "Installation de NeMo >=2.3.0 (compatible whisper-diarization)"

    # NeMo selon les spécifications whisper-diarization
    log_step "Installation de Cython (prérequis)"
    if ! pip install "Cython"; then
        log_error "❌ Échec de l'installation de Cython"
        exit 1
    fi

    log_step "Installation de nemo_toolkit[asr]>=2.4.0"
    if ! pip install "nemo_toolkit[asr]>=2.4.0"; then
        log_error "❌ Échec de l'installation de NeMo"
        log_warning "⚠️  NeMo est requis pour whisper-diarization"
        log_info "💡 Vérifiez la compatibilité PyTorch/CUDA"
        exit 1
    fi

    # Vérifier l'installation
    if python -c "import nemo; print(f'✅ NeMo {nemo.__version__} installé')" 2>/dev/null; then
        log_success "NeMo >=2.3.0 installé avec succès"
    else
        log_error "❌ NeMo installé mais non importable"
        exit 1
    fi
}

# --- Création du wrapper CLI ---
create_modern_wrapper() {
    log_step "Création du wrapper CLI moderne"

    local abs_path
    abs_path="$(pwd)"
    
    # Déterminer le chemin d'activation selon le type d'environnement
    local activation_cmd
    if [[ $USE_CONDA -eq 1 ]]; then
        activation_cmd="conda activate whisper_modern_env"
    else
        activation_cmd="source \"${abs_path}/whisper_modern_env/bin/activate\""
    fi

    cat > whisperx_cli << EOF
#!/bin/bash
# Modern Whisper CLI Wrapper (September 2025)

if [[ "\$1" == "--version" ]]; then
    echo "Modern Whisper CLI v${SCRIPT_VERSION}"
    echo "Based on whisper-diarization + NeMo 2.0"
    exit 0
fi

# Activer l'environnement
${activation_cmd}

# Lancer le CLI moderne
python "${abs_path}/whisperx_cli.py" "\$@"
EOF

    chmod +x whisperx_cli

    # Installation globale
    if [[ "$(id -u)" -ne 0 ]]; then
        sudo cp whisperx_cli /usr/local/bin/whisperx_cli
    else
        cp whisperx_cli /usr/local/bin/whisperx_cli
    fi

    chmod +x /usr/local/bin/whisperx_cli
    WRAPPER_INSTALLED=1

    log_success "Wrapper CLI installé globalement"
}

# --- Test du système ---
test_installation() {
    log_step "Test de l'installation"

    # Vérifier que le CLI fonctionne
    if ./whisperx_cli --version > /dev/null 2>&1; then
        log_success "CLI fonctionne correctement"
    else
        log_error "Problème avec le CLI"
        return 1
    fi

    # Test rapide avec un fichier audio (si disponible)
    if [[ -f "audio.mp3" ]]; then
        log_step "Test de transcription sur audio.mp3"

        local test_cmd="./whisperx_cli audio.mp3 --model large-v3 --output test_output.txt --output_format txt"

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

    # Test de compatibilité cuDNN/GPU si CUDA disponible
    if command -v nvidia-smi &> /dev/null && [[ -n "$HF_TOKEN" ]]; then
        log_step "Test de compatibilité cuDNN/GPU"
        
        # Test simple de diarization GPU
        local gpu_test_cmd="python -c \"
import torch
import warnings
warnings.filterwarnings('ignore')

try:
    # Test CUDA disponible
    if not torch.cuda.is_available():
        print('⚠️  CUDA non disponible')
        exit(0)
    
    # Test cuDNN
    if not torch.backends.cudnn.enabled:
        print('⚠️  cuDNN non activé')
        exit(0)
    
    # Test création tensor GPU
    test_tensor = torch.randn(1, 1).cuda()
    del test_tensor
    
    # Test convolution simple (utilise cuDNN)
    import torch.nn as nn
    conv = nn.Conv1d(1, 1, 3).cuda()
    input_tensor = torch.randn(1, 1, 10).cuda()
    output = conv(input_tensor)
    del conv, input_tensor, output
    
    print('✅ Compatibilité cuDNN/GPU confirmée')
    
except Exception as e:
    print(f'⚠️  Problème cuDNN détecté: {str(e)[:100]}...')
    print('💡 La diarization utilisera le CPU (plus lent mais stable)')
\""
        
        if [[ $VERBOSE -eq 1 ]]; then
            eval "$gpu_test_cmd"
        else
            local gpu_result
            gpu_result=$(eval "$gpu_test_cmd" 2>&1)
            if echo "$gpu_result" | grep -q "✅"; then
                log_success "Compatibilité cuDNN/GPU confirmée"
            else
                log_warning "Problème cuDNN détecté - fallback CPU disponible"
                if [[ $VERBOSE -eq 1 ]]; then
                    echo "$gpu_result"
                fi
            fi
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

    # Démarrage du tunnel avec URL fixe
    nohup ngrok http --url=innocent-new-mole.ngrok-free.app "$port" > /dev/null 2>&1 &
    local ngrok_pid=$!

    sleep 3

    log_success "Tunnel ngrok actif: https://innocent-new-mole.ngrok-free.app"
}

# --- Nettoyage en cas d'erreur ---
cleanup() {
    log_warning "Interruption détectée - nettoyage en cours"

    if [[ $ENV_CREATED -eq 1 ]]; then
        if [[ $USE_CONDA -eq 1 ]]; then
            conda env remove -n whisper_modern_env -y 2>/dev/null || true
            log_info "Environnement conda supprimé"
        else
            rm -rf whisper_modern_env
            log_info "Environnement venv supprimé"
        fi
    fi

    if [[ $CLONED -eq 1 ]]; then
        cd .. && rm -rf whisper-x-setup
        log_info "Dépôt supprimé"
    fi

    if [[ $WRAPPER_INSTALLED -eq 1 ]]; then
        sudo rm -f /usr/local/bin/whisperx_cli 2>/dev/null || true
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
    echo "   • ${BOLD}whisperx_cli${RESET} - CLI moderne avec diarization avancée"
    echo "   • ${BOLD}whisperx_cli --version${RESET} - Informations sur la version"
    echo
    
    echo -e "${BOLD}🌐 API Whisper :${RESET}"
    echo "   • URL locale: ${BLUE}http://localhost:3000${RESET}"
    if [[ -n "$NGROK_TOKEN" ]]; then
        echo "   • URL externe: ${BLUE}https://innocent-new-mole.ngrok-free.app${RESET}"
    fi
    echo
    echo -e "${BOLD}📝 Exemples d'utilisation :${RESET}"
    echo "   • Transcription simple:"
    echo "     ${BLUE}whisperx_cli audio.mp3 --model large-v3 --language fr${RESET}"
    echo
    echo "   • Avec diarization:"
    echo "     ${BLUE}whisperx_cli audio.mp3 --diarize --hf_token YOUR_TOKEN --nb_speaker 2${RESET}"
    echo
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "   • Sur macOS:"
        echo "     ${BLUE}whisperx_cli audio.mp3 --compute_type int8 --device cpu${RESET}"
        echo
    fi

    echo -e "${BOLD}🚀 Nouveautés 2025 :${RESET}"
    echo "   ✅ PyTorch $PYTORCH_VERSION avec CUDA 12.x"
    echo "   ✅ faster-whisper >=1.1.0 optimisé"
    echo "   ✅ pyannote.audio 4.0.0 avec community-1 pipeline"
    if [[ $INSTALL_NEMO -eq 1 ]]; then
        echo "   ✅ NeMo >=2.4.0 pour diarization avancée"
    fi
    echo "   ✅ Demucs + CTC-forced-aligner"
    echo "   ✅ Support Python 3.9-3.12"
    echo "   ✅ Interface CLI modernisée"
    echo "   ✅ Configuration cuDNN automatique pour GPU"
    echo
    
    # Instructions spéciales pour GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${BOLD}🎮 Configuration GPU :${RESET}"
        echo "   • Les variables d'environnement cuDNN sont configurées automatiquement"
        echo "   • L'environnement active automatiquement la configuration CUDA"
        echo "   • En cas de problème cuDNN, le fallback CPU est transparent"
        echo
    fi
}

# --- Fonction principale ---
main() {
    trap cleanup SIGINT

    echo -e "${BOLD}🎤 Modern Whisper Setup v${SCRIPT_VERSION} (September 2025)${RESET}"
    echo -e "${BOLD}Basé sur whisper-diarization + NeMo >=2.4.0 + PyTorch $PYTORCH_VERSION${RESET}"
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

        # Vérifier la présence du CLI moderne
        if [[ ! -f "whisperx_cli.py" ]]; then
            log_error "Le fichier whisperx_cli.py est introuvable"
            log_info "💡 Vérifiez que le dépôt contient bien whisperx_cli.py"
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