# WhisNemo

**Whisper + NeMo MSDD pipeline for audio transcription and speaker diarization.**

WhisNemo processes interview-length audio files (20–60 minutes) through a production-tested pipeline:

```
Demucs vocal separation → Faster-Whisper transcription → CTC forced alignment
→ NeMo MSDD diarization (subprocess-isolated) → speaker-word mapping → TXT / SRT / CSV
```

## Installation

### Prerequisites

WhisNemo requires a working CUDA-enabled PyTorch installation. Install the correct version for your system **before** installing WhisNemo:

```bash
# Example for CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install WhisNemo

```bash
# Core pipeline
pip install whisnemo

# With Demucs vocal separation (recommended)
pip install whisnemo[stemming]

# Everything
pip install whisnemo[full]

# From source
git clone https://github.com/dumrania/whisnemo.git
cd whisnemo
pip install -e ".[full]"
```

### Install from TestPyPI (pre-release)

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ whisnemo
```

## Quick Start

### Command Line

```bash
# Process a single file
whisnemo diarize -a interview.mp3

# Batch-process a directory
whisnemo batch -a /path/to/audio_dir --device cuda:0

# Extract outputs into organized folders
whisnemo extract /path/to/audio_dir /path/to/output

# Remove stutters from transcript CSVs
whisnemo stutter /path/to/csv_dir --threshold 0.8
```

### Python API

```python
from whisnemo import process_file, process_audio_files

# Single file
csv_path = process_file(
    "interview.mp3",
    output_dir="./results",
    model_name="medium.en",
    device="cuda",
)

# Batch
results = process_audio_files(
    ["file1.mp3", "file2.wav"],
    output_dir="./results",
    skip_existing=True,
)
print(f"Succeeded: {len(results['succeeded'])}")
print(f"Failed:    {len(results['failed'])}")
```

## Architecture

| Stage | Tool | Purpose |
|-------|------|---------|
| Vocal separation | Demucs htdemucs | Remove music/noise |
| Transcription | Faster-Whisper (via WhisperX) | Speech-to-text |
| Forced alignment | CTC forced aligner | Word-level timestamps |
| Diarization | NeMo MSDD | Speaker identification |
| Punctuation | deepmultilingualpunctuation | Restore sentence boundaries |
| Output | Custom | TXT, SRT, CSV with speaker labels |

### Key design decisions

- **Transcript-first architecture**: Whisper runs on full audio first, then NeMo assigns speakers. The reverse (segment-first) produces incoherent transcripts due to short segments.
- **Subprocess-isolated diarization**: NeMo MSDD runs in a child process so CUDA OOM kills only the worker, not the pipeline.
- **Overclustering approach**: `oracle_num_speakers=False` with post-processing outperforms forced speaker count constraints.

## CLI Reference

| Command | Description |
|---------|-------------|
| `whisnemo diarize` | Process single audio file |
| `whisnemo batch` | Batch-process directory |
| `whisnemo extract` | Organize outputs into folders |
| `whisnemo stutter` | Remove repetitions from CSVs |
| `whisnemo version` | Print version |

Each command also has a standalone entry point: `whisnemo-diarize`, `whisnemo-batch`, etc.

## Development

```bash
git clone https://github.com/dumrania/whisnemo.git
cd whisnemo
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/

# Build
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*
```

## License

MIT
