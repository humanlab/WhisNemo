# WhisNemo

WhisNemo is a Whisper + NeMo MSDD pipeline for transcription and speaker diarization.

## Supported platforms

WhisNemo is validated on the following platforms with Python 3.10:

- **Linux x86_64** with NVIDIA CUDA (12.4 or 12.8) — production target
- **Windows x86_64** with NVIDIA CUDA (12.4 or 12.8) — experimental, validated end-to-end
- **macOS arm64** with Apple Silicon MPS — experimental, validated end-to-end

See the [Install](#install) section for platform-specific instructions.

## Install

WhisNemo's diarization stack depends on PyTorch and several transitive
dependencies (NeMo, OpenAI Whisper, Demucs) that pin specific versions of
torch and numpy. The install uses a constraints file to ensure pip
resolves a compatible plan across platforms.

### Prerequisites (all platforms)

- **Python 3.10**. On Windows, install Python 3.10.11 from python.org
  (later 3.10.x releases are source-only).
- **ffmpeg** on your system PATH. Required by Whisper, Demucs, and pydub
  for audio I/O. Installation:
  - Linux (Debian/Ubuntu): `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: install from [gyan.dev FFmpeg builds](https://www.gyan.dev/ffmpeg/builds/)
    and add the `bin/` directory to your PATH, or use `imageio-ffmpeg`
    as a fallback (see Windows notes below).

All three platforms use the same two-step install pattern: install torch
with a platform-appropriate wheel, then install whisnemo with the
shared `constraints/runtime.txt` file.

### Linux

```bash
python3.10 -m venv .venv-whisnemo
source .venv-whisnemo/bin/activate
pip install --upgrade pip setuptools wheel

# GPU with CUDA 12.8 (adjust cu128 to match your driver's CUDA version):
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# CPU-only alternative:
# pip install torch==2.11.0 torchaudio==2.11.0

# Install whisnemo with the diarize extra, using the constraints file:
pip install -c "https://raw.githubusercontent.com/humanlab/WhisNemo/dumrania/timing-and-postprocess/constraints/runtime.txt" "whisnemo[diarize] @ git+https://github.com/humanlab/WhisNemo.git@dumrania/timing-and-postprocess"
```

### macOS (experimental, Apple Silicon)

```bash
python3.10 -m venv .venv-whisnemo
source .venv-whisnemo/bin/activate
pip install --upgrade pip setuptools wheel

# macOS: torch ships from PyPI directly (Apple Silicon MPS supported):
pip install torch==2.11.0 torchaudio==2.11.0

# Install whisnemo with the diarize extra, using the constraints file:
pip install -c "https://raw.githubusercontent.com/humanlab/WhisNemo/dumrania/timing-and-postprocess/constraints/runtime.txt" "whisnemo[diarize] @ git+https://github.com/humanlab/WhisNemo.git@dumrania/timing-and-postprocess"
```

On Apple Silicon, pass `device="mps"` to `run_diarize` to use GPU
acceleration via Metal Performance Shaders. WhisNemo applies runtime
patches to openai-whisper and nemo-toolkit automatically when MPS is
requested; no user action is required beyond setting the device.

### Windows (experimental)

```powershell
py -3.10 -m venv .venv-whisnemo
.\.venv-whisnemo\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel

# GPU with CUDA 12.8 (requires NVIDIA driver supporting CUDA 12.8+):
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Windows users on older NVIDIA drivers (driver CUDA < 12.8) should use
# torch 2.6.0+cu124 instead - NeMo 2.7.2 remains compatible:
# pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# CPU-only alternative:
# pip install torch==2.11.0 torchaudio==2.11.0

# Install whisnemo with the diarize extra, using the constraints file:
pip install -c "https://raw.githubusercontent.com/humanlab/WhisNemo/dumrania/timing-and-postprocess/constraints/runtime.txt" "whisnemo[diarize] @ git+https://github.com/humanlab/WhisNemo.git@dumrania/timing-and-postprocess"
```

If ffmpeg is not on your system PATH on Windows, use `imageio-ffmpeg`
as a fallback:

```powershell
pip install imageio-ffmpeg
$src = python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
Copy-Item $src (Join-Path (Split-Path $src) "ffmpeg.exe")
```

Then prepend the imageio-ffmpeg binaries directory to your `PATH`, or
add this to your scripts before importing whisnemo:

```python
import os, imageio_ffmpeg
os.environ["PATH"] = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) + os.pathsep + os.environ.get("PATH", "")
```

### Verify the install (all platforms)

```bash
python -c "from whisnemo.core.diarize import run_diarize; print('import OK')"
```

If this prints `import OK` (after several NeMo startup warnings), the
install succeeded.

### Known limitations

- **Demucs vocal separation** is affected by a `torchcodec` / CUDA
  runtime compatibility issue (`libnvrtc.so.13`) on Linux, Windows, and
  macOS. When this occurs, the pipeline falls back gracefully to running
  on the original audio without vocal pre-separation. To explicitly
  disable Demucs and suppress the warning, pass `stemming=False` when
  calling `run_diarize`. A fix is in progress; for most interview audio
  the impact is minimal since vocal separation is a pre-processing aid
  rather than a requirement.
- **macOS MPS transcript segments.** On Apple Silicon with `device="mps"`,
  Whisper can occasionally drop short transcript segments on some files.
  The same files transcribe completely on Linux and Windows (CUDA) and on
  macOS with `device="cpu"`. This is under investigation. If you see
  missing content on a Mac, use `device="cpu"`.
- **GPU validation is per-platform.** Linux is validated on an NVIDIA
  A6000. Windows is validated on an RTX 4060 with torch 2.6.0+cu124.
  macOS is validated on Apple Silicon with MPS.

## CLI
```bash
whisnemo version
whisnemo diarize -a sample.wav
whisnemo batch -a /path/to/audio_dir --start 1 --end 10
```

## Notes

- The diarization stack is packaged as an extra: `whisnemo[diarize]`
- Validated on Linux, Windows, and macOS with Python 3.10
- The current validated setup uses OpenAI Whisper and NeMo 2.7.2.
  Word-level timestamps come from Whisper directly rather than from a
  separate forced-alignment step, which removes a dependency that had
  no Windows build and worked inconsistently on macOS.
- GPU support depends on installing a torch build compatible with your
  platform's GPU stack (CUDA on Linux/Windows, MPS on macOS) before
  installing the diarization extra.
- macOS GPU support (MPS) applies two runtime patches to openai-whisper
  and nemo-toolkit. These patches only activate when `device="mps"` is
  passed to `run_diarize` and are no-ops on other platforms.
