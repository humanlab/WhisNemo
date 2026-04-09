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


### Windows (experimental)

Native Windows support for `whisnemo[diarize]` is experimental and
currently validated for transcription and diarization without Demucs
vocal separation. Linux remains the recommended platform.

#### Prerequisites

- Python 3.10 (install from python.org — Python 3.10.11 is the latest
  Windows installer; later 3.10.x releases are source-only).
- ffmpeg available on your system PATH. The simplest options:
  - Install via [gyan.dev FFmpeg builds](https://www.gyan.dev/ffmpeg/builds/)
    and add the `bin/` directory to your PATH, or
  - Use `imageio-ffmpeg`'s bundled binary (see step 4 below).

#### Install steps

1. Create and activate a Python 3.10 virtual environment:

```powershell
   py -3.10 -m venv .venv-whisnemo
   .\.venv-whisnemo\Scripts\Activate.ps1
   pip install --upgrade pip setuptools wheel
```

2. Install PyTorch with the version pinned by the constraints file.
   For CPU-only:

```powershell
   pip install torch==2.11.0 torchaudio==2.11.0
```

   For GPU (CUDA 12.8):

```powershell
   pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

3. Install whisnemo with the diarize extra, using the Windows
   constraints file to pin torch and numpy:

```powershell
   pip install -c "https://raw.githubusercontent.com/humanlab/WhisNemo/dumrania/timing-and-postprocess/constraints/windows.txt" "whisnemo[diarize] @ git+https://github.com/humanlab/WhisNemo.git@dumrania/timing-and-postprocess"
```

4. (If ffmpeg is not on your system PATH) Install imageio-ffmpeg and
   create an `ffmpeg.exe` symlink in its binaries directory so audio
   libraries can find it:

```powershell
   pip install imageio-ffmpeg
   $src = python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
   Copy-Item $src (Join-Path (Split-Path $src) "ffmpeg.exe")
```

   Then prepend the imageio-ffmpeg binaries directory to your `PATH`
   environment variable, or add this snippet to your scripts before
   importing whisnemo:

```python
   import os, imageio_ffmpeg
   os.environ["PATH"] = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) + os.pathsep + os.environ.get("PATH", "")
```

#### Running on Windows

Demucs vocal separation has known issues on Windows and should be
disabled by passing `stemming=False`:

```python
from whisnemo.core.diarize import run_diarize

run_diarize(
    audio_path="my_audio.wav",
    stemming=False,           # required on Windows for now
    model_name="medium.en",
    device="cuda",            # or "cpu"
    output_formats=["csv", "txt", "srt"],
    num_speakers=2,
    oracle_num_speakers=False,
)
```

#### Validated configuration

- Windows 10/11
- Python 3.10.11
- torch 2.11.0+cpu
- NeMo 2.7.2
- Tested end-to-end with Whisper `tiny.en` and `medium.en` models on
  short and medium-length audio files.

GPU-accelerated runs on Windows are theoretically supported via
`torch==2.11.0+cu128` but have not been validated as of this writing.

## Notes

- The diarization stack is packaged as an extra: `whisnemo[diarize]`
- Tested on Linux x86_64 with Python 3.10
- The current validated setup uses OpenAI Whisper, NeMo 2.7.2, and the upstream `ctc-forced-aligner`
- GPU support depends on installing a CUDA-compatible PyTorch build before installing the diarization extra
- For now, the recommended tested install path is Linux first
