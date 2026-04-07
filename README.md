# WhisNemo

WhisNemo is a Whisper + NeMo MSDD pipeline for transcription and speaker diarization.

## Supported platform

Current supported target:

- Linux x86_64
- Python 3.10

This package is not yet officially supported on macOS or Windows.

## Install

### Base package

pip install whisnemo

### Full diarization stack

For GPU-enabled Linux install on CUDA 12.8:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "whisnemo[diarize]"

## CLI

whisnemo version
whisnemo diarize -a sample.wav
whisnemo batch -a /path/to/audio_dir --start 1 --end 10

## Notes

- The diarization stack is packaged as an extra: `whisnemo[diarize]`
- Tested on Linux x86_64 with Python 3.10
- The current validated setup uses OpenAI Whisper, NeMo 2.7.2, and the upstream `ctc-forced-aligner`
- GPU support depends on installing a CUDA-compatible PyTorch build before installing the diarization extra
- For now, the recommended tested install path is Linux first
