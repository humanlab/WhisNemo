"""
Main diarization pipeline — ported from whisper-diarization/diarize.py.
"""

import argparse
import ctypes
import logging
import os
import re
import sys
import time
import csv
from datetime import datetime

# --- cuDNN preload (must happen before any torch/NeMo imports) ---
import importlib.util
_torch_spec = importlib.util.find_spec("torch")
if _torch_spec and _torch_spec.origin:
    _torch_lib = os.path.join(os.path.dirname(_torch_spec.origin), "lib")
    _cudnn_path = os.path.join(_torch_lib, "libcudnn.so.8")
    if os.path.isfile(_cudnn_path):
        ctypes.CDLL(_cudnn_path, mode=ctypes.RTLD_GLOBAL)
        os.environ['LD_LIBRARY_PATH'] = _torch_lib
os.environ.pop('CUDA_HOME', None)
os.environ.pop('CUDA_PATH', None)
# --- end cuDNN preload ---

import torch
import torchaudio
import soundfile as sf
import resampy
# from ctc_forced_aligner import (
#     generate_emissions,
#     get_alignments,
#     get_spans,
#     load_alignment_model,
#     load_audio,
#     postprocess_results,
#     preprocess_text,
# )
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from whisnemo.core.helpers import (
    cleanup,
    create_config,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    punct_model_langs,
    whisper_langs,
    write_srt,
)
from whisnemo.core.transcription_helpers import transcribe
from whisnemo.core.timing_utils import setup_timing
from whisnemo.core.format_srt import format_srt_to_csv

mtypes = {"cpu": "int8", "cuda": "float16", "mps": "float32"}

# --- MPS compatibility patches ---
# Apple Silicon MPS backend has two known issues with our upstream dependencies
# that we patch at runtime when device="mps" is requested. These patches are
# no-ops on cpu and cuda, so they only activate when actually needed.
#
# 1. openai-whisper: whisper.timing.dtw calls .double() on an MPS tensor, which
#    fails because MPS does not support float64. We reorder the cast so the move
#    to CPU happens before the cast to double.
#
# 2. nemo-toolkit: MSDD_module.conv_scale_weights uses .view() on a tensor that
#    is non-contiguous on MPS after the preceding convolutions. On CUDA the
#    memory layout happens to be contiguous and .view() works by accident. We
#    replace .view() with .reshape() which handles both cases correctly.
#
# Both patches are idempotent via _MPS_PATCHES_APPLIED.
_MPS_PATCHES_APPLIED = False

def _apply_mps_patches():
    """Apply runtime patches for MPS compatibility. Idempotent."""
    global _MPS_PATCHES_APPLIED
    if _MPS_PATCHES_APPLIED:
        return
    # Ensure PyTorch MPS fallback is enabled for any unsupported ops
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Patch 1: openai-whisper dtw float64 incompatibility
    import whisper.timing as _whisper_timing
    _dtw_cpu = _whisper_timing.dtw_cpu
    def _patched_dtw(x):
        return _dtw_cpu(x.cpu().double().numpy())
    _whisper_timing.dtw = _patched_dtw

    # Patch 2: NeMo MSDD_module.conv_scale_weights view/reshape
    import torch.nn.functional as F
    from nemo.collections.asr.modules.msdd_diarizer import MSDD_module

    def _patched_conv_scale_weights(self, ms_avg_embs_perm, ms_emb_seq_single):
        ms_cnn_input_seq = torch.cat([ms_avg_embs_perm, ms_emb_seq_single], dim=2)
        ms_cnn_input_seq = ms_cnn_input_seq.unsqueeze(2).flatten(0, 1)
        conv_out = self.conv_forward(
            ms_cnn_input_seq, conv_module=self.conv[0], bn_module=self.conv_bn[0], first_layer=True
        )
        for conv_idx in range(1, self.conv_repeat + 1):
            conv_out = self.conv_forward(
                conv_input=conv_out,
                conv_module=self.conv[conv_idx],
                bn_module=self.conv_bn[conv_idx],
                first_layer=False,
            )
        # .view() fails on non-contiguous MPS tensors; .reshape() handles both
        lin_input_seq = conv_out.reshape(self.batch_size, self.length, self.cnn_output_ch * self.emb_dim)
        hidden_seq = self.conv_to_linear(lin_input_seq)
        hidden_seq = self.dropout(F.leaky_relu(hidden_seq))
        scale_weights = self.softmax(self.linear_to_weights(hidden_seq))
        scale_weights = scale_weights.unsqueeze(3).expand(-1, -1, -1, self.num_spks)
        return scale_weights

    MSDD_module.conv_scale_weights = _patched_conv_scale_weights

    _MPS_PATCHES_APPLIED = True
    logging.info("Applied MPS compatibility patches for whisper and nemo-toolkit")


def _normalize_device(device):
    """Return a device string that is actually available on this machine.

    If "cuda" is requested but unavailable (e.g. on macOS), fall back to
    "mps" if available, otherwise "cpu". This prevents NeMo from calling
    CUDA-only APIs on a machine without CUDA. No-op when the requested
    device is available.
    """
    if device == "cuda" and not torch.cuda.is_available():
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            logging.warning("CUDA not available; falling back to MPS (Apple Silicon).")
            return "mps"
        logging.warning("CUDA not available; falling back to CPU.")
        return "cpu"
    if device == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        logging.warning("MPS not available; falling back to CPU.")
        return "cpu"
    return device


def _set_macos_multiprocessing():
    """On macOS, force the 'spawn' multiprocessing start method.

    NeMo's diarization uses fork-based multiprocessing, which is unsafe on
    macOS once OpenMP / threaded libraries are loaded and can cause segfaults.
    Using 'spawn' avoids this. No-op on Linux/Windows (which keep 'fork',
    the faster default). force=True makes this safe to call repeatedly.
    """
    if sys.platform == "darwin":
        import multiprocessing
        try:
            multiprocessing.set_start_method("spawn", force=True)
            logging.info("Set multiprocessing start method to 'spawn' for macOS.")
        except RuntimeError:
            pass



def run_diarize(audio_path, stemming=True, suppress_numerals=False,
                model_name="medium.en", batch_size=8, language=None,
                device="cuda",
                # Output format selection (default: csv only)
                output_formats=None,
                # Stutter removal (opt-in)
                remove_stutters=False, stutter_threshold=0.8,
                # NeMo config params
                num_speakers=2, oracle_num_speakers=True,
                vad_model="vad_multilingual_marblenet",
                speaker_model="titanet_large",
                onset=0.8, offset=0.5, pad_offset=-0.05,
                domain_type="telephonic"):
    """
    Run the full diarization pipeline on a single audio file.

    Args:
        output_formats: List of output formats to produce. Options: "csv", "txt", "srt".
                        Default (None) = ["csv"] only.
        remove_stutters: If True, run stutter removal on the CSV output.
        stutter_threshold: Similarity threshold for stutter removal (0.0-1.0).
        num_speakers: Expected number of speakers in manifest.
        oracle_num_speakers: Whether to use oracle num speakers in clustering.
        vad_model: NeMo VAD model name.
        speaker_model: NeMo speaker embedding model name.
        onset/offset/pad_offset: VAD parameters.
        domain_type: NeMo config domain type (telephonic, meeting, general).
    """
    # Normalize the device: fall back gracefully if the requested
    # accelerator isn't available (e.g. "cuda" requested on a Mac).
    device = _normalize_device(device)

    # On macOS, NeMo's fork-based multiprocessing can segfault; use spawn.
    _set_macos_multiprocessing()

    # Apply MPS compatibility patches if running on Apple Silicon.
    # No-op on cpu and cuda.
    if device == "mps":
        _apply_mps_patches()

    if output_formats is None:
        output_formats = ["csv"]

    audio_filename, timing_csv, log_timing = setup_timing(audio_path, "diarization")
    attempt_num = int(os.environ.get('WHISNEMO_ATTEMPT', '1'))

    # --- 1. Demucs vocal separation ---
    start_time = time.time()
    if stemming:
        return_code = os.system(
            f'"{sys.executable}" -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs"'
        )
        if return_code != 0:
            logging.warning(
                "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
            )
            vocal_target = audio_path
        else:
            vocal_target = os.path.join(
                "temp_outputs", "htdemucs",
                os.path.splitext(os.path.basename(audio_path))[0],
                "vocals.wav",
            )
    else:
        vocal_target = audio_path
    end_time = time.time()
    log_timing("audio_preprocessing_stemming", start_time, end_time)

    # --- 2. Whisper transcription (non-batched) ---
    start_time = time.time()
    whisper_results, language_detected = transcribe(
        vocal_target, language, model_name,
        mtypes[device], suppress_numerals, device,
    )

    audio_waveform, sr = sf.read(vocal_target)
    if audio_waveform.ndim > 1:
        audio_waveform = audio_waveform.mean(axis=1)
    if sr != 16000:
        audio_waveform = resampy.resample(audio_waveform, sr, 16000)
        sr = 16000
    audio_waveform = audio_waveform.astype("float32")

    end_time = time.time()
    log_timing("whisper_transcription", start_time, end_time)

    # --- 3. Build word_timestamps from Whisper's native word-level output ---
    # Replaces the previous CTC forced alignment step. Whisper produces
    # word-level timestamps when word_timestamps=True, which transcription_helpers
    # already sets for supported languages. Quality is slightly worse than CTC at
    # word boundaries, but for 2-speaker structured interviews the difference is
    # typically negligible because turn boundaries are long compared to the
    # ~100-300ms timestamp jitter.
    start_time = time.time()
    word_timestamps = []
    for segment in whisper_results:
        words = segment.get("words") or []
        for w in words:
            # Whisper returns "word" (with leading space); helpers.py expects "text"
            text = w.get("word") or w.get("text") or ""
            text = text.strip()
            if not text:
                continue
            ws = w.get("start")
            we = w.get("end")
            if ws is None or we is None:
                continue
            word_timestamps.append({
                "text": text,
                "start": float(ws),
                "end": float(we),
            })

    if not word_timestamps:
        # Fallback: synthesize from segment-level timestamps if no word-level data
        logging.warning(
            "No word-level timestamps from Whisper; falling back to segment-level."
        )
        for segment in whisper_results:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            word_timestamps.append({
                "text": text,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
            })

    end_time = time.time()
    log_timing("word_timestamps_from_whisper", start_time, end_time)

    # --- 4. Convert to mono for NeMo ---
    start_time = time.time()
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    sf.write(
        os.path.join(temp_path, "mono_file.wav"),
        audio_waveform,
        16000,
    )
    end_time = time.time()
    log_timing("audio_conversion_mono", start_time, end_time)

    # --- 5. NeMo MSDD diarization (with configurable params) ---
    start_time = time.time()
    msdd_model = NeuralDiarizer(cfg=create_config(
        temp_path,
        num_speakers=num_speakers,
        oracle_num_speakers=oracle_num_speakers,
        vad_model=vad_model,
        speaker_model=speaker_model,
        onset=onset,
        offset=offset,
        pad_offset=pad_offset,
        domain_type=domain_type,
    )).to(device)
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()
    end_time = time.time()
    log_timing("nemo_speaker_diarization", start_time, end_time)

    # --- 6. Read RTTM speaker timestamps ---
    start_time = time.time()
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    end_time = time.time()
    log_timing("speaker_timestamp_mapping", start_time, end_time)

    # --- 7. Punctuation restoration ---
    start_time = time.time()
    if language_detected in punct_model_langs:
        punct_model = PunctuationModel(model="kredor/punctuate-all")
        words_list = list(map(lambda x: x["word"], wsm))
        labled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
    else:
        logging.warning(
            f"Punctuation restoration is not available for {language_detected} language. Using the original punctuation."
        )
    end_time = time.time()
    log_timing("punctuation_restoration", start_time, end_time)

    # --- 8. Write outputs (only requested formats) ---
    start_time = time.time()
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    base_path = os.path.splitext(audio_path)[0]

    if "txt" in output_formats:
        with open(f"{base_path}.txt", "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f)
        print(f"  Output: {base_path}.txt")

    if "srt" in output_formats or "csv" in output_formats:
        # SRT is always needed as intermediate for CSV
        srt_path = f"{base_path}.srt"
        with open(srt_path, "w", encoding="utf-8-sig") as srt:
            write_srt(ssm, srt)
        if "srt" in output_formats:
            print(f"  Output: {srt_path}")

        if "csv" in output_formats:
            csv_path = format_srt_to_csv(srt_path)
            print(f"  Output: {csv_path}")

            # --- Optional stutter removal ---
            if remove_stutters:
                from whisnemo.postprocessing.remove_stutters import correct_file_with_similarity
                result = correct_file_with_similarity(csv_path, stutter_threshold)
                if result:
                    print(f"  Stutter removal: removed {result['messages_removed']} repetitions")

        # Clean up SRT if not requested
        if "srt" not in output_formats and os.path.isfile(srt_path):
            os.remove(srt_path)

    cleanup(temp_path)
    end_time = time.time()
    log_timing("output_generation", start_time, end_time)

    print(f"[TIMING] Completed diarization for {audio_filename} - Attempt {attempt_num}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio", help="name of the target audio file", required=True)
    parser.add_argument("--no-stem", action="store_false", dest="stemming", default=True,
                        help="Disables source separation.")
    parser.add_argument("--suppress_numerals", action="store_true", dest="suppress_numerals",
                        default=False, help="Suppresses Numerical Digits.")
    parser.add_argument("--whisper-model", dest="model_name", default="medium.en",
                        help="name of the Whisper model to use")
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=8,
                        help="Batch size for batched inference")
    parser.add_argument("--language", type=str, default=None, choices=whisper_langs,
                        help="Language spoken in the audio")
    parser.add_argument("--device", dest="device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="if you have a GPU use 'cuda', otherwise 'cpu'")
    # Output format
    parser.add_argument("--formats", nargs="+", default=["csv"],
                        choices=["csv", "txt", "srt"],
                        help="Output formats to produce (default: csv)")
    # Stutter removal
    parser.add_argument("--remove-stutters", action="store_true", default=False,
                        help="Run stutter removal on CSV output")
    parser.add_argument("--stutter-threshold", type=float, default=0.8,
                        help="Similarity threshold for stutter removal (0.0-1.0)")
    # NeMo config
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--no-oracle-speakers", action="store_false", dest="oracle_num_speakers", default=True,
                        help="Disable oracle num speakers (let NeMo auto-detect)")
    parser.add_argument("--vad-model", default="vad_multilingual_marblenet")
    parser.add_argument("--speaker-model", default="titanet_large")
    parser.add_argument("--onset", type=float, default=0.8)
    parser.add_argument("--offset", type=float, default=0.5)
    parser.add_argument("--pad-offset", type=float, default=-0.05)
    parser.add_argument("--domain-type", default="telephonic",
                        choices=["telephonic", "meeting", "general"])

    args = parser.parse_args()

    run_diarize(
        audio_path=args.audio,
        stemming=args.stemming,
        suppress_numerals=args.suppress_numerals,
        model_name=args.model_name,
        batch_size=args.batch_size,
        language=args.language,
        device=args.device,
        output_formats=args.formats,
        remove_stutters=args.remove_stutters,
        stutter_threshold=args.stutter_threshold,
        num_speakers=args.num_speakers,
        oracle_num_speakers=args.oracle_num_speakers,
        vad_model=args.vad_model,
        speaker_model=args.speaker_model,
        onset=args.onset,
        offset=args.offset,
        pad_offset=args.pad_offset,
        domain_type=args.domain_type,
    )


if __name__ == "__main__":
    main()
