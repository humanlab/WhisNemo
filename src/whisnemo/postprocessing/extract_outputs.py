"""
Extract and organize WhisNemo pipeline output files from audio directories.

Collects TXT, SRT, CSV transcripts, timing logs, and run_status files into
a clean directory structure.
"""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_outputs(source_dir: str, dest_dir: str, copy_mode: str = "copy") -> dict:
    """
    Extract WhisNemo output files from *source_dir* into organized *dest_dir*.

    Args:
        source_dir: Source audio directory containing pipeline outputs.
        dest_dir: Destination directory.
        copy_mode: One of 'copy', 'move', or 'symlink'.

    Returns:
        Summary dict with file counts per category.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    subdirs = {
        "transcripts_txt": dest_path / "transcripts_txt",
        "transcripts_srt": dest_path / "transcripts_srt",
        "transcripts_csv": dest_path / "transcripts_csv",
        "timing_logs": dest_path / "timing_logs",
        "run_status": dest_path / "run_status",
    }
    for sd in subdirs.values():
        sd.mkdir(parents=True, exist_ok=True)

    patterns = [
        ("*.txt", "transcripts_txt"),
        ("*.srt", "transcripts_srt"),
        ("*_formatted.csv", "transcripts_csv"),
        ("*_timing.csv", "timing_logs"),
        ("*.done", "run_status"),
    ]

    counts = {}
    for pattern, key in patterns:
        count = 0
        for fp in source_path.rglob(pattern):
            if fp.is_file():
                dest_file = subdirs[key] / fp.name
                # Handle name conflicts
                if dest_file.exists():
                    base, suffix = fp.stem, fp.suffix
                    c = 1
                    while dest_file.exists():
                        dest_file = subdirs[key] / f"{base}_{c}{suffix}"
                        c += 1
                try:
                    _transfer(fp, dest_file, copy_mode)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed: {fp.name}: {e}")
        counts[key] = count

    # Also pull from timing_logs/ and *_runstatus/ subdirectories
    for td in source_path.rglob("timing_logs"):
        if td.is_dir():
            for tf in td.glob("*"):
                if tf.is_file():
                    _transfer(tf, subdirs["timing_logs"] / tf.name, copy_mode)

    for rd in source_path.rglob("*_runstatus"):
        if rd.is_dir():
            dest_rs = subdirs["run_status"] / rd.name
            try:
                if copy_mode == "copy":
                    shutil.copytree(rd, dest_rs, dirs_exist_ok=True)
                elif copy_mode == "move":
                    shutil.move(str(rd), str(dest_rs))
                elif copy_mode == "symlink":
                    os.symlink(rd.absolute(), dest_rs)
            except Exception as e:
                logger.error(f"Failed runstatus dir {rd.name}: {e}")

    logger.info(f"Extraction complete → {dest_dir}")
    return counts


def _transfer(src: Path, dst: Path, mode: str):
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        os.symlink(src.absolute(), dst)
