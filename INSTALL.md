# WhisNemo Installation Guide

## What install.sh does

The dependency stack is complex (NeMo, WhisperX, Demucs, pyannote, etc.)
with strict version pinning requirements. A plain `pip install` cannot
resolve all of this cleanly. The install script handles it in phases:

1. **Build tools**: setuptools, Cython, pybind11
2. **PyTorch**: torch==2.1.0 + torchaudio for CUDA 11.8
3. **Native/git deps**: NeMo, WhisperX, Demucs, CTC aligner, pyannote, speechbrain
4. **WhisNemo itself**: installed with `--no-deps` since everything is already in place

## Changing CUDA version

Edit `install.sh` and change the `TORCH_INDEX` variable:
- CUDA 11.8: `https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`
- CPU only: `https://download.pytorch.org/whl/cpu`

## Non-PyPI dependencies

These packages are only available from GitHub (pinned to specific commits):
- [WhisperX](https://github.com/m-bain/whisperX)
- [CTC Forced Aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [Demucs](https://github.com/adefossez/demucs)
- [Deep Multilingual Punctuation](https://github.com/oliverguhr/deepmultilingualpunctuation)

## Troubleshooting

**`ModuleNotFoundError: No module named 'pkg_resources'`**
→ `pip install setuptools`

**`av` / FFmpeg build errors**
→ WhisNemo uses `ffmpeg-python` (not `av`). If something pulls in `av`, install FFmpeg headers: `conda install ffmpeg` or `apt install libavformat-dev libavcodec-dev`

**`ModelFilter` import error from huggingface_hub**
→ Version mismatch. Run: `pip install huggingface-hub==0.23.2 --no-deps`

**CUDA OOM**
→ Use `whisnemo batch` which auto-splits and retries on OOM

**numpy 2.0 errors (`np.sctypes` removed)**
→ `pip install numpy==1.23.5`
