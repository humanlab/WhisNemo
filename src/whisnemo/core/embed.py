"""
Embedding extraction for WhisNemo.

Two embedding backends, selectable via `model=`:

- "whispa"  : WhiSPA speech-psychological embeddings (single vector per segment).
              Requires the WhiSPA repo to be importable (see `whispa_repo_path`).
- "whisper" : Whisper encoder/decoder hidden-state summary statistics
              (mean/median/var/min/max per hidden dim).

Both operate on the *participant* segments of a diarized transcript: the
function reads the transcript, slices the audio into per-segment clips, runs
the chosen model on each clip, and returns one row per segment as a DataFrame.

This module is the single-file / talk-facing path: it is single-process
(safe to call from R via reticulate). A separate batch path can layer
multi-GPU parallelism on top of the same per-segment helpers.
"""

import os
import sys
import gc
import tempfile
import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device handling (mirror diarize.py: graceful fallback on machines without CUDA)
# ---------------------------------------------------------------------------
def _normalize_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            logger.warning("CUDA not available; falling back to MPS.")
            return "mps"
        logger.warning("CUDA not available; falling back to CPU.")
        return "cpu"
    if device == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        logger.warning("MPS not available; falling back to CPU.")
        return "cpu"
    return device


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _timestamp_to_seconds(ts):
    """Accept seconds (numeric) or HH:MM:SS.sss / MM:SS strings."""
    if pd.isna(ts):
        return None
    if isinstance(ts, (int, float, np.integer, np.floating)):
        return float(ts)
    ts = str(ts).strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 1:
        return float(parts[0])
    raise ValueError(f"Unrecognized timestamp format: {ts}")


def _load_transcript(transcript):
    """Accept a DataFrame or a path to a CSV."""
    if isinstance(transcript, pd.DataFrame):
        return transcript.copy()
    return pd.read_csv(transcript)


def _segment_bounds(row):
    """Return (start_sec, end_sec) from a transcript row, or (None, None)."""
    if "start" in row.index and "end" in row.index:
        try:
            return float(row["start"]), float(row["end"])
        except (TypeError, ValueError):
            return None, None
    start = _timestamp_to_seconds(row.get("start_timestamp"))
    end = _timestamp_to_seconds(row.get("end_timestamp"))
    return start, end


def _iter_participant_segments(seg_df, participant_only):
    """Yield (row_index, row, start_sec, end_sec) for valid segments."""
    for r_idx, row in seg_df.iterrows():
        if participant_only and row.get("speaker_role") != "participant":
            continue
        start_sec, end_sec = _segment_bounds(row)
        if start_sec is None or end_sec is None or end_sec <= start_sec:
            continue
        yield r_idx, row, start_sec, end_sec


def _write_segment_clips(audio_path, segments, temp_dir):
    """Slice audio into per-segment WAV clips. Returns list of (r_idx, clip_path)."""
    import librosa
    import soundfile as sf

    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    clips = []
    for r_idx, row, start_sec, end_sec in segments:
        start_index = max(0, int(start_sec * sample_rate))
        end_index = min(len(audio_data), int(end_sec * sample_rate))
        if end_index <= start_index:
            continue
        clip = audio_data[start_index:end_index]
        if len(clip) == 0:
            continue
        clip_path = os.path.join(temp_dir, f"{base}__{r_idx:05d}.wav")
        sf.write(clip_path, clip, sample_rate)
        clips.append((r_idx, clip_path))
    return clips


def _preprocess_clip(clip_path):
    """Load a clip as a mono 16 kHz waveform tensor.

    Uses soundfile + librosa rather than torchaudio.load, because in
    torchaudio 2.11 the load path routes through torchcodec, which fails
    on environments hitting the libnvrtc.so.13 issue. soundfile reads the
    WAV directly without that dependency.
    """
    import soundfile as sf
    import librosa

    data, sr = sf.read(clip_path, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    return torch.from_numpy(data).unsqueeze(0)


# ---------------------------------------------------------------------------
# WhiSPA backend
# ---------------------------------------------------------------------------
def _ensure_whispa_importable(whispa_repo_path=None):
    """Make the WhiSPA repo importable. Returns the WhiSPAModel class.

    Prefers a pip-installed WhiSPA (import works directly). If that fails and
    a whispa_repo_path is provided, falls back to appending that local clone
    path to sys.path and retrying. This lets the module work on any machine
    where WhiSPA is pip-installed, while still supporting a local clone.
    """
    try:
        from pretrain.whispa_model import WhiSPAModel  # noqa: E402
        return WhiSPAModel
    except ImportError:
        pass

    if whispa_repo_path and whispa_repo_path not in sys.path:
        sys.path.append(whispa_repo_path)
        try:
            from pretrain.whispa_model import WhiSPAModel  # noqa: E402
            return WhiSPAModel
        except ImportError as e:
            raise ImportError(
                "Could not import WhiSPA from the provided whispa_repo_path "
                f"({whispa_repo_path}). Underlying error: {e}"
            )

    raise ImportError(
        "Could not import WhiSPA. Install it with "
        "'pip install git+https://github.com/humanlab/WhiSPA.git' "
        "or pass whispa_repo_path pointing at a local clone of the WhiSPA repo."
    )


def _extract_whispa(
    audio_path, seg_df, device, participant_only,
    whisper_model_id, whispa_model_id, whispa_repo_path,
):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    WhiSPAModel = _ensure_whispa_importable(whispa_repo_path)

    processor = WhisperProcessor.from_pretrained(whisper_model_id)
    whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_id).eval().to(device)
    whispa = WhiSPAModel.from_pretrained(whispa_model_id).eval().to(device)

    rows = []
    segments = list(_iter_participant_segments(seg_df, participant_only))
    with tempfile.TemporaryDirectory(prefix="whisnemo_embed_") as temp_dir:
        clips = _write_segment_clips(audio_path, segments, temp_dir)
        seg_meta = {r_idx: (row, s, e) for r_idx, row, s, e in segments}
        for r_idx, clip_path in clips:
            try:
                waveform = _preprocess_clip(clip_path)
                if waveform.numel() == 0:
                    continue
                audio_inputs = processor(
                    waveform.squeeze(), sampling_rate=16000, return_tensors="pt"
                )
                input_features = audio_inputs["input_features"].to(device)
                with torch.no_grad():
                    tokens = whisper.generate(input_features)
                    embedding = whispa(
                        audio_inputs=input_features,
                        text_input_ids=tokens,
                        text_attention_mask=torch.ones(tokens.size(), device=device),
                    )
                vec = embedding.squeeze(0).detach().cpu().numpy()
                row, s, e = seg_meta[r_idx]
                rows.append(_assemble_row(r_idx, row, s, e, vec))
            except Exception as ex:
                logger.warning(f"WhiSPA segment {r_idx} failed: {ex!r}")
                continue

    del whisper, whispa
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    feat_names = [f"f{i:05d}" for i in range(len(rows[0]) - 4)] if rows else []
    return _rows_to_df(rows, feat_names)


# ---------------------------------------------------------------------------
# Whisper backend (encoder / decoder hidden-state summary stats)
# ---------------------------------------------------------------------------
_STATS = ["mea", "med", "var", "min", "max"]


def _summarize_hidden(hidden):
    """hidden: (seq_len, hidden_dim) -> concat of mean/median/var/min/max -> (5*hidden_dim,)."""
    emb_mea = torch.mean(hidden, dim=0)
    emb_med, _ = torch.median(hidden, dim=0)
    emb_var = torch.var(hidden, dim=0, unbiased=False)
    emb_min, _ = torch.min(hidden, dim=0)
    emb_max, _ = torch.max(hidden, dim=0)
    return torch.cat([emb_mea, emb_med, emb_var, emb_min, emb_max]).cpu().numpy()


def _extract_whisper(
    audio_path, seg_df, device, participant_only,
    whisper_model_id, return_enc, return_dec,
):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained(whisper_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_model_id).eval().to(device)
    hidden_size = model.config.hidden_size

    enc_rows, dec_rows = [], []
    segments = list(_iter_participant_segments(seg_df, participant_only))
    with tempfile.TemporaryDirectory(prefix="whisnemo_embed_") as temp_dir:
        clips = _write_segment_clips(audio_path, segments, temp_dir)
        seg_meta = {r_idx: (row, s, e) for r_idx, row, s, e in segments}
        for r_idx, clip_path in clips:
            try:
                waveform = _preprocess_clip(clip_path)
                if waveform.numel() == 0:
                    continue
                row, s, e = seg_meta[r_idx]
                text_value = row.get("message", "")
                if pd.isna(text_value):
                    text_value = ""
                audio_inputs = processor(
                    waveform.squeeze(), sampling_rate=16000, return_tensors="pt"
                ).to(device)
                text_inputs = processor.tokenizer(
                    str(text_value), padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    out = model(
                        input_features=audio_inputs["input_features"],
                        decoder_input_ids=text_inputs["input_ids"],
                        decoder_attention_mask=text_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                if return_enc:
                    enc = out.encoder_last_hidden_state.squeeze()
                    enc_rows.append(_assemble_row(r_idx, row, s, e, _summarize_hidden(enc)))
                if return_dec:
                    dec = out.decoder_hidden_states[-1].squeeze()
                    dec_rows.append(_assemble_row(r_idx, row, s, e, _summarize_hidden(dec)))
            except Exception as ex:
                logger.warning(f"Whisper segment {r_idx} failed: {ex!r}")
                continue

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    feat_cols = [f"f{i:05d}_{stat}" for stat in _STATS for i in range(hidden_size)]
    result = {}
    if return_enc:
        result["encoder"] = _rows_to_df(enc_rows, feat_cols)
    if return_dec:
        result["decoder"] = _rows_to_df(dec_rows, feat_cols)
    # If only one was requested, return that DataFrame directly for convenience.
    if return_enc and not return_dec:
        return result["encoder"]
    if return_dec and not return_enc:
        return result["decoder"]
    return result


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------
def _assemble_row(r_idx, row, start_sec, end_sec, vec):
    speaker = row.get("speaker", "")
    return [r_idx, start_sec, end_sec, speaker] + list(np.asarray(vec).ravel())


def _rows_to_df(rows, feat_names):
    cols = ["segment_id", "start_sec", "end_sec", "speaker"] + feat_names
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def extract_embeddings(
    audio_path,
    transcript,
    model="whispa",
    device="cuda",
    participant_only=True,
    whisper_model_id=None,
    whispa_model_id="Jarhatz/WhiSPA-V1-Small",
    whispa_repo_path=None,
    return_enc=True,
    return_dec=False,
):
    """
    Extract per-segment embeddings for one audio file.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    transcript : str or pandas.DataFrame
        Diarized transcript (path to CSV, or a DataFrame). Must contain
        per-segment start/end timestamps (either numeric "start"/"end"
        columns or "start_timestamp"/"end_timestamp" in HH:MM:SS.sss),
        and a "speaker_role" column if participant_only is True.
    model : str
        "whispa" or "whisper".
    device : str
        "cuda", "cpu", or "mps".
    participant_only : bool
        If True, only segments with speaker_role == "participant" are used.
    whisper_model_id : str
        Whisper model. Defaults to "openai/whisper-small" for whispa
        (which is paired with whisper-small) and "openai/whisper-medium"
        for the whisper backend.
    whispa_model_id : str
        WhiSPA checkpoint on HuggingFace.
    whispa_repo_path : str
        Path to a local clone of the WhiSPA repo, if WhiSPA is not pip-installed.
    return_enc, return_dec : bool
        For model="whisper": which hidden states to summarize. If both are
        True, returns a dict {"encoder": df, "decoder": df}; if only one,
        returns that DataFrame directly.

    Returns
    -------
    pandas.DataFrame (or dict of DataFrames for whisper with both enc+dec)
        One row per participant segment: segment_id, start_sec, end_sec,
        speaker, then the embedding feature columns.
    """
    device = _normalize_device(device)
    seg_df = _load_transcript(transcript)

    if model == "whispa":
        wid = whisper_model_id or "openai/whisper-small"
        return _extract_whispa(
            audio_path, seg_df, device, participant_only,
            wid, whispa_model_id, whispa_repo_path,
        )
    elif model == "whisper":
        wid = whisper_model_id or "openai/whisper-medium"
        if not (return_enc or return_dec):
            raise ValueError("For model='whisper', at least one of return_enc/return_dec must be True.")
        return _extract_whisper(
            audio_path, seg_df, device, participant_only,
            wid, return_enc, return_dec,
        )
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'whispa' or 'whisper'.")
