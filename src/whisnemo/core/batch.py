"""
Batch processing pipeline — Python replacement for whisnemo_pipeline_batch_v1.0.sh.
"""

import logging
import os
import shutil
import sys
import time
from pathlib import Path

from whisnemo.core.timing_utils import log_bash_event
from whisnemo.core.format_srt import format_srt_to_csv
from whisnemo.core.utils import (
    get_audio_duration_ffmpeg,
    split_audio_file,
    get_op_csv_path,
    get_op_srt_path,
    process_updating_timestamps_srt_csv,
    process_speaker_labels,
    concatenate_csv_files,
)
from whisnemo.core.helpers import cleanup

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".wma", ".flac", ".ogg", ".mp4")
SKIP_EXTENSIONS = (".srt", ".txt", ".csv")


def sanitize_filenames(audio_dir: str):
    """Rename files with spaces to underscores."""
    audio_dir = Path(audio_dir)
    for f in audio_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in SKIP_EXTENSIONS:
            continue
        if " " in f.name:
            new_name = f.name.replace(" ", "_")
            new_path = f.parent / new_name
            f.rename(new_path)
            print(f"✓ Renamed: {f.name} -> {new_name}")


def _run_diarize_single(audio_file: str, device: str, attempt: int, **diarize_kwargs):
    """Run diarize on a single file. Returns True on success."""
    os.environ['WHISNEMO_ATTEMPT'] = str(attempt)

    try:
        from whisnemo.core.diarize import run_diarize
        run_diarize(audio_path=audio_file, device=device, **diarize_kwargs)
        return True
    except Exception as e:
        logger.error(f"Diarize failed for {audio_file} (attempt {attempt}): {e}")
        return False


def _run_srt_to_csv(audio_file: str):
    """Convert the SRT output to formatted CSV."""
    srt_path = os.path.splitext(audio_file)[0] + ".srt"
    if os.path.isfile(srt_path):
        try:
            format_srt_to_csv(srt_path)
            return True
        except Exception as e:
            logger.error(f"SRT to CSV conversion failed for {srt_path}: {e}")
            return False
    else:
        logger.warning(f"SRT file not found: {srt_path}")
        return False


def organize_outputs(audio_dir: str):
    """
    Organize outputs into subfolders next to the audio directory.
    Creates audio_dir_outputs/ with transcripts_csv/, transcripts_txt/, etc.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(str(audio_dir) + "_outputs")

    subdirs = {
        "csv": output_dir / "transcripts_csv",
        "txt": output_dir / "transcripts_txt",
        "srt": output_dir / "transcripts_srt",
        "timing": output_dir / "timing_logs",
    }
    for sd in subdirs.values():
        sd.mkdir(parents=True, exist_ok=True)

    counts = {"csv": 0, "txt": 0, "srt": 0, "timing": 0}

    for f in audio_dir.iterdir():
        if not f.is_file():
            continue
        try:
            if f.name.endswith("_formatted.csv") or f.name.endswith("_corrected.csv"):
                shutil.copy2(f, subdirs["csv"] / f.name)
                counts["csv"] += 1
            elif f.suffix == ".txt" and not f.name.startswith("split_ids"):
                shutil.copy2(f, subdirs["txt"] / f.name)
                counts["txt"] += 1
            elif f.suffix == ".srt":
                shutil.copy2(f, subdirs["srt"] / f.name)
                counts["srt"] += 1
        except Exception as e:
            logger.error(f"Failed to organize {f.name}: {e}")

    # Copy timing logs
    timing_src = audio_dir / "timing_logs"
    if timing_src.is_dir():
        for f in timing_src.iterdir():
            if f.is_file():
                shutil.copy2(f, subdirs["timing"] / f.name)
                counts["timing"] += 1

    print(f"\nOutputs organized in: {output_dir}")
    for key, count in counts.items():
        if count > 0:
            print(f"  {key}: {count} files")


def retry_oom_with_splits(audio_file: str, audio_dir: str, device: str,
                          max_split_retries: int = 2, **diarize_kwargs):
    """OOM recovery: split, diarize each, update timestamps, fix speakers, concatenate."""
    file_path = Path(audio_file)
    file_name = file_path.name
    file_stem = file_path.stem
    max_splits = 2

    temp_dir = Path(f"{audio_file}_oomretry")
    temp_done_dir = Path(f"{audio_file}_oomretry_runstatus")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_done_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created Temp Retry: {temp_dir}, Runstatus: {temp_done_dir}")

    shutil.copy2(audio_file, temp_dir / file_name)
    new_main_audio = str(temp_dir / file_name)
    main_done_file = Path(audio_dir + "_runstatus") / f"{file_name}_run.done"

    attempt = 0
    all_splits_done = False

    while attempt <= max_split_retries:
        print(f"Splitting Attempt: {attempt}")

        split_files = split_audio_file(new_main_audio, str(temp_dir), max_splits)
        if not split_files:
            logger.error(f"Failed to split {new_main_audio}")
            attempt += 1
            continue

        for df in temp_done_dir.glob("*.done"):
            df.unlink()

        splits_succeeded = 0
        for split_file in split_files:
            split_file_str = str(split_file)
            os.environ['WHISNEMO_ATTEMPT'] = str(attempt)

            # Force srt output for splits (needed for CSV conversion)
            split_kwargs = dict(diarize_kwargs)
            split_kwargs["output_formats"] = ["srt"]
            split_kwargs.pop("remove_stutters", None)

            success = _run_diarize_single(split_file_str, device, attempt, **split_kwargs)
            if success:
                done_file = temp_done_dir / f"{split_file.name}.done"
                done_file.touch()
                splits_succeeded += 1

        if splits_succeeded == max_splits:
            print(f"Split Retry Attempt {attempt}: all splits diarized successfully")
            all_splits_done = True
            break

        attempt += 1

    if not all_splits_done:
        print(f"Split retry failed for {audio_file} after {max_split_retries} attempts")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if temp_done_dir.exists():
            shutil.rmtree(temp_done_dir)
        return

    for srt_file in temp_dir.glob("*.srt"):
        format_srt_to_csv(str(srt_file))

    csv_audio_mapping = {}
    for split_file in sorted(temp_dir.glob("*__split_*_formatted.csv")):
        audio_stem = split_file.name.replace("_formatted.csv", "")
        for ext in [".wav", ".mp3", ".m4a"]:
            audio_candidate = temp_dir / f"{audio_stem}{ext}"
            if audio_candidate.exists():
                csv_audio_mapping[str(split_file)] = str(audio_candidate)
                break

    if not csv_audio_mapping:
        logger.error("No CSV-audio mappings found after splitting")
        shutil.rmtree(temp_dir)
        shutil.rmtree(temp_done_dir)
        return

    try:
        ts_updated = process_updating_timestamps_srt_csv(
            csv_audio_mapping, '_formatted.csv', '_formatted_timestamp_updated.csv'
        )
        spk_updated = process_speaker_labels(
            ts_updated, '_formatted_timestamp_updated.csv', '_formatted_speaker_label.csv'
        )
        final_csv = concatenate_csv_files(
            spk_updated, '_split_1_formatted_speaker_label.csv', '_formatted.csv'
        )

        final_dest = os.path.join(audio_dir, f"{file_stem}_formatted.csv")
        shutil.copy2(final_csv, final_dest)
        print(f"Final CSV: {final_dest}")

        main_done_file.parent.mkdir(parents=True, exist_ok=True)
        main_done_file.touch()
    except Exception as e:
        logger.error(f"Post-split processing failed: {e}")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if temp_done_dir.exists():
            shutil.rmtree(temp_done_dir)


def run_batch(
    audio_dir: str,
    device: str = "cuda",
    try_limit: int = 2,
    start_idx: int = 1,
    end_idx: int = None,
    organize: bool = False,
    # Pass-through to run_diarize
    output_formats=None,
    remove_stutters=False,
    stutter_threshold=0.8,
    num_speakers=2,
    oracle_num_speakers=True,
    vad_model="vad_multilingual_marblenet",
    speaker_model="titanet_large",
    onset=0.8,
    offset=0.5,
    pad_offset=-0.05,
    domain_type="telephonic",
    stemming=True,
    model_name="medium.en",
    batch_size=8,
    language=None,
    suppress_numerals=False,
):
    """
    Batch-process audio files — full replacement for whisnemo_pipeline_batch_v1.0.sh.
    """
    if output_formats is None:
        output_formats = ["csv"]

    audio_dir = audio_dir.rstrip("/")

    # Diarize kwargs to pass through
    diarize_kwargs = dict(
        stemming=stemming,
        model_name=model_name,
        batch_size=batch_size,
        language=language,
        suppress_numerals=suppress_numerals,
        output_formats=output_formats,
        remove_stutters=remove_stutters,
        stutter_threshold=stutter_threshold,
        num_speakers=num_speakers,
        oracle_num_speakers=oracle_num_speakers,
        vad_model=vad_model,
        speaker_model=speaker_model,
        onset=onset,
        offset=offset,
        pad_offset=pad_offset,
        domain_type=domain_type,
    )

    # Setup timing
    timing_dir = os.path.join(audio_dir, "timing_logs")
    os.makedirs(timing_dir, exist_ok=True)
    batch_label = f"batch_{start_idx}_to_{end_idx or 'end'}"

    log_bash_event(timing_dir, "batch_start", batch_label, "1")
    print(f"[TIMING] Batch started - Processing files {start_idx} to {end_idx or 'all'}")

    done_dir = audio_dir + "_runstatus"
    os.makedirs(done_dir, exist_ok=True)

    sanitize_filenames(audio_dir)

    audio_files = sorted(
        f for f in Path(audio_dir).iterdir()
        if f.is_file() and f.suffix.lower() not in SKIP_EXTENSIONS
        and f.suffix.lower() in AUDIO_EXTENSIONS
    )

    if end_idx is None:
        end_idx = len(audio_files)

    print(f"Found {len(audio_files)} audio files, processing index {start_idx} to {end_idx}")

    for ctr, file_path in enumerate(audio_files, 1):
        if ctr < start_idx or ctr > end_idx:
            continue

        file_name = file_path.name
        audio_file = str(file_path)
        done_file = os.path.join(done_dir, f"{file_name}_run.done")

        if os.path.isfile(done_file):
            print(f"Skipping File: {file_name} (already completed)")
            continue

        print(f"Started File: {file_name} @ Index: {ctr}")

        attempt = 1
        success = False
        while attempt <= try_limit:
            print(f"Attempt: {attempt}; File: {audio_file}")

            if _run_diarize_single(audio_file, device, attempt, **diarize_kwargs):
                Path(done_file).touch()
                success = True
                print(f"Finished File: {file_name}")
                break
            else:
                print(f"At Attempt: {attempt}; File Failed: {audio_file}")
                attempt += 1

        if not success:
            print(f"Exceeded try limit of {try_limit}. Splitting and retrying.")
            retry_oom_with_splits(audio_file, audio_dir, device,
                                  max_split_retries=2, **diarize_kwargs)

    log_bash_event(timing_dir, "batch_end", batch_label, "1")
    print(f"[TIMING] Batch completed - Processed files {start_idx} to {end_idx}")
    print(f"Finished Iteration From {start_idx} to {end_idx} in Audio Dir: {audio_dir}")

    # Optional: organize outputs into subfolders
    if organize:
        organize_outputs(audio_dir)
