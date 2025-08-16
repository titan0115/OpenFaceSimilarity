#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PY_DIR="$SCRIPT_DIR/python"
PY_VERSION="3.13.3"
VENV_DIR="$SCRIPT_DIR/venv"

detect_platform() {
    OS=$(uname -s)
    ARCH=$(uname -m)

    case "$ARCH" in
        "arm64") ARCH="aarch64" ;;
        "x86_64") ARCH="x86_64" ;;
        *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
    esac

    case "$OS" in
        "Darwin") PLATFORM="${ARCH}-apple-darwin" ;;
        "Linux") PLATFORM="${ARCH}-unknown-linux-gnu" ;;
        *) echo "Unsupported OS: $OS"; exit 1 ;;
    esac
}

install_python() {
    detect_platform

    if [ ! -d "$PY_DIR" ]; then
        echo "Python not found. Downloading portable Python $PY_VERSION for $PLATFORM..."
        
        # Исправленный URL с актуальной датой релиза
        BASE_URL="https://github.com/indygreg/python-build-standalone/releases/download/20250409"
        FILENAME="cpython-${PY_VERSION}+20250409-${PLATFORM}-install_only.tar.gz"
        PY_URL="${BASE_URL}/${FILENAME}"
        
        DOWNLOAD_FILE="$SCRIPT_DIR/python.tar.gz"

        command -v wget >/dev/null || { echo "Please install wget"; exit 1; }
        command -v tar >/dev/null || { echo "Please install tar"; exit 1; }

        echo "Downloading Python from: $PY_URL"
        if ! wget -q --show-progress -O "$DOWNLOAD_FILE" "$PY_URL"; then
            echo "❌ Download failed! Check the URL or network connection."
            exit 1
        fi

        echo "Extracting Python..."
        mkdir -p "$PY_DIR"
        if ! tar -xzf "$DOWNLOAD_FILE" -C "$PY_DIR" --strip-components=1; then
            echo "❌ Extraction failed!"
            rm -rf "$PY_DIR" "$DOWNLOAD_FILE"
            exit 1
        fi
        rm "$DOWNLOAD_FILE"

        PY_EXEC="$PY_DIR/bin/python3"
        if [ -f "$PY_EXEC" ]; then
            chmod +x "$PY_EXEC"
            echo "✅ Python installed: $PY_EXEC"
        else
            echo "❌ Python installation failed!"
            exit 1
        fi
    fi
}

setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment..."
        PY_EXEC="$PY_DIR/bin/python3"
        
        if [ ! -f "$PY_EXEC" ]; then
            echo "❌ Python executable missing: $PY_EXEC"
            exit 1
        fi

        if ! "$PY_EXEC" -m venv "$VENV_DIR"; then
            echo "❌ Failed to create venv"
            exit 1
        fi

        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip || exit 1

        if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
            pip install -r "$SCRIPT_DIR/requirements.txt" || exit 1
        else
            echo "❌ requirements.txt not found"
            exit 1
        fi

        deactivate
        echo "✅ Virtual environment ready"
    fi
}

fix_paths() {
    [ ! -d "$VENV_DIR" ] && return

    if [ "$OS" = "Darwin" ]; then
        sed -i "" "s|home = .*|home = $PY_DIR|" "$VENV_DIR/pyvenv.cfg"
        sed -i "" "s|VIRTUAL_ENV=.*|VIRTUAL_ENV=\"$VENV_DIR\"|" "$VENV_DIR/bin/activate"
    else
        sed -i "s|home = .*|home = $PY_DIR|" "$VENV_DIR/pyvenv.cfg"
        sed -i "s|VIRTUAL_ENV=.*|VIRTUAL_ENV=\"$VENV_DIR\"|" "$VENV_DIR/bin/activate"
    fi
}

cleanup() {
    echo "Cleaning temporary files..."
    rm -rf tmp/* 2>/dev/null
}

main() {
    install_python
    setup_venv
    fix_paths
    cleanup
    
    source "$VENV_DIR/bin/activate"
    python train.py 
}

main