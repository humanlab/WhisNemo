#!/bin/bash
set -e

echo "============================================="
echo "WhisNemo Installation Script"
echo "============================================="

# --- Phase 0: Check Python version ---
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $python_version"

if [[ "$python_version" != "3.10" && "$python_version" != "3.11" ]]; then
    echo "ERROR: WhisNemo requires Python 3.10 or 3.11, found $python_version"
    exit 1
fi

# --- Phase 1: Build tools ---
echo ""
echo "[Phase 1/5] Installing build tools..."
pip install --no-cache-dir \
    "setuptools>=68.0" \
    wheel \
    Cython==3.0.11 \
    pybind11 \
    packaging

# --- Phase 2: PyTorch (must match your CUDA version) ---
echo ""
echo "[Phase 2/5] Installing PyTorch..."
TORCH_INDEX="https://download.pytorch.org/whl/cu118"
echo "Installing torch from: $TORCH_INDEX"
pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url "$TORCH_INDEX"

# --- Phase 3: FFmpeg + av (system-level dep for faster-whisper) ---
echo ""
echo "[Phase 3/5] Installing FFmpeg and PyAV..."
conda install -c conda-forge 'ffmpeg>=4.2,<5.0' pkg-config -y
pip install --no-cache-dir av==11.0.0
pip install --no-cache-dir ffmpeg-python==0.2.0

# --- Phase 4: Python dependencies in correct order ---
echo ""
echo "[Phase 4/5] Installing Python dependencies..."

# HuggingFace stack pinned first (order matters)
pip install --no-cache-dir --no-deps \
    huggingface-hub==0.23.2 \
    tokenizers==0.15.2 \
    safetensors==0.6.2 \
    transformers==4.39.3

# youtokentome needs Cython at build time (must come before NeMo)
pip install --no-cache-dir --no-build-isolation youtokentome==1.0.6

# NeMo and heavy deps
pip install --no-cache-dir --no-build-isolation \
    numpy==1.23.5 \
    omegaconf==2.2.2 \
    pytorch-lightning==1.9.4 \
    nemo_toolkit[asr]==1.20.0 \
    pyannote.audio==3.1.1 \
    speechbrain==1.0.3

# Git-based packages (not on PyPI)
pip install --no-cache-dir --no-build-isolation \
    "ctc-forced-aligner @ git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git@abd458dd879305566cd4ed0c8624c95f22e3126a"

pip install --no-cache-dir --no-build-isolation \
    "deepmultilingualpunctuation @ git+https://github.com/oliverguhr/deepmultilingualpunctuation.git@5a0dd7f4fd56687f59405aa8eba1144393d8b74b"

pip install --no-cache-dir --no-build-isolation \
    "demucs @ git+https://github.com/adefossez/demucs.git@b9ab48cad45976ba42b2ff17b229c071f0df9390"

pip install --no-cache-dir --no-build-isolation \
    "whisperx @ git+https://github.com/m-bain/whisperX.git@78dcfaab51005aa703ee21375f81ed31bc248560"

pip install --no-cache-dir faster-whisper==1.0.0

# Remaining deps
pip install --no-cache-dir nltk==3.9.1 wget==3.2 "pandas>=1.5,<3.0"

# --- Phase 5: Install WhisNemo itself ---
echo ""
echo "[Phase 5/5] Installing WhisNemo..."
pip install --no-cache-dir --no-deps --no-build-isolation -e .

echo ""
echo "============================================="
echo "WhisNemo installation complete!"
echo "Test with: whisnemo version"
echo "============================================="
