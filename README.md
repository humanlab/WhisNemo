# WhisNemo

**Whisper + NeMo MSDD pipeline for audio transcription and speaker diarization.**

WhisNemo processes interview-length audio files (20–60 minutes) through a production-tested pipeline:
```
Demucs vocal separation → Faster-Whisper transcription → CTC forced alignment
→ NeMo MSDD diarization → speaker-word mapping → TXT / SRT / CSV
```

## Installation

### Prerequisites

- Python 3.10 or 3.11
- conda (for FFmpeg installation)
- NVIDIA GPU with CUDA 11.8 (recommended)

### Quick Install
```bash
conda create -n whisnemo python=3.10 -y
conda activate whisnemo
git clone git@github.com:humanlab/WhisNemo.git
cd WhisNemo
git checkout dumrania/timing-and-postprocess
bash install.sh
```

The install script handles the full dependency stack in the correct order: build tools, PyTorch+CUDA, FFmpeg, NeMo, WhisperX, and all other dependencies. See [INSTALL.md](INSTALL.md) for details and troubleshooting.

### Verify
```bash
whisnemo version
```

## Usage

### Single File
```bash
# Default (CSV output only)
whisnemo diarize -a interview.mp3

# All output formats + stutter removal
whisnemo diarize -a interview.mp3 --formats csv txt srt --remove-stutters

# Custom NeMo parameters
whisnemo diarize -a interview.mp3 --num-speakers 3 --no-oracle-speakers --onset 0.7
```

### Batch Processing
```bash
# Process files 1-50 in a directory
CUDA_VISIBLE_DEVICES=0 whisnemo batch -a /path/to/audio_dir --start 1 --end 50

# With stutter removal and output organization
whisnemo batch -a /path/to/audio_dir --remove-stutters --organize

# Custom parameters
whisnemo batch -a /path/to/audio_dir --num-speakers 4 --no-oracle-speakers --try-limit 3
```

### Other Commands
```bash
# Extract and organize outputs into subfolders
whisnemo extract /path/to/audio_dir /path/to/output_dir

# Remove stutters from existing CSV files
whisnemo stutter /path/to/csv_dir --threshold 0.8
```

## Pipeline Configuration

### Output Formats

Use `--formats` to select which outputs to produce (default: `csv` only):
```bash
whisnemo diarize -a audio.mp3 --formats csv txt srt
```

### NeMo Diarization Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--num-speakers` | 2 | Expected number of speakers |
| `--no-oracle-speakers` | (oracle on) | Let NeMo auto-detect speaker count |
| `--vad-model` | vad_multilingual_marblenet | NeMo VAD model |
| `--speaker-model` | titanet_large | Speaker embedding model |
| `--onset` | 0.8 | VAD onset threshold |
| `--offset` | 0.5 | VAD offset threshold |
| `--pad-offset` | -0.05 | VAD pad offset |
| `--domain-type` | telephonic | telephonic, meeting, or general |

### Batch Processing Options

| Flag | Default | Description |
|------|---------|-------------|
| `--start` | 1 | Start index (1-indexed) |
| `--end` | all | End index (1-indexed) |
| `--try-limit` | 2 | Max attempts before OOM split-retry |
| `--organize` | off | Organize outputs into subfolders after batch |
| `--remove-stutters` | off | Run stutter removal on CSV outputs |
| `--stutter-threshold` | 0.8 | Similarity threshold for stutter removal |

## Architecture

| Stage | Tool | Purpose |
|-------|------|---------|
| Vocal separation | Demucs htdemucs | Remove music/noise |
| Transcription | Faster-Whisper | Speech-to-text |
| Forced alignment | CTC forced aligner | Word-level timestamps |
| Diarization | NeMo MSDD (telephonic) | Speaker identification |
| Punctuation | deepmultilingualpunctuation | Restore sentence boundaries |
| Output | WhisNemo | TXT, SRT, CSV with speaker labels |

### Key Design Decisions

- **Transcript-first**: Whisper runs on full audio first, then NeMo assigns speakers. The reverse produces incoherent transcripts.
- **Oracle speaker count**: `oracle_num_speakers=True` with `num_speakers=2` by default. Override with `--no-oracle-speakers` for auto-detection.
- **OOM recovery**: Batch mode auto-splits audio on CUDA OOM, diarizes each half, corrects timestamps and speaker labels, and rejoins.

## Python API
```python
from whisnemo.core.diarize import run_diarize

run_diarize(
    audio_path="interview.mp3",
    output_formats=["csv", "txt"],
    num_speakers=2,
    device="cuda",
)
```
```python
from whisnemo.core.batch import run_batch

run_batch(
    audio_dir="/path/to/audio",
    start_idx=1,
    end_idx=50,
    remove_stutters=True,
    organize=True,
)
```

## Non-PyPI Dependencies

These packages are installed from GitHub (pinned to specific commits):

- [WhisperX](https://github.com/m-bain/whisperX)
- [CTC Forced Aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [Demucs](https://github.com/adefossez/demucs)
- [Deep Multilingual Punctuation](https://github.com/oliverguhr/deepmultilingualpunctuation)

## License

MIT
