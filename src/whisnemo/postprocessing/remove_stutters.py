"""
Similarity-based consecutive stutter / repetition removal for transcript CSVs.

Usage as library:
    from whisnemo.postprocessing.remove_stutters import correct_file_with_similarity
    result = correct_file_with_similarity("transcript.csv", similarity_threshold=0.8)

Usage as CLI:
    whisnemo-stutter /path/to/csv_folder --threshold 0.8
"""

import difflib
import glob
import json
import logging
import os
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def remove_consecutive_repetitive_messages(
    df: pd.DataFrame,
    text_column: str = "message",
    similarity_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Remove consecutive repetitive messages using sequence similarity.

    Preserves timestamps by merging time ranges when messages are removed.
    Does **not** group by speaker (preserves chronological order).
    """
    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' not found. Available: {list(df.columns)}")
        return df

    df_work = df.copy().reset_index(drop=True)

    # Detect timestamp columns
    start_col = end_col = None
    for col in df_work.columns:
        if "start" in col.lower() and "time" in col.lower():
            start_col = col
        elif "end" in col.lower() and "time" in col.lower():
            end_col = col

    has_timestamps = start_col is not None and end_col is not None

    def _merge_ts(kept_row, removed_indices):
        if not has_timestamps or not removed_indices:
            return kept_row
        all_starts = [kept_row[start_col]] + [df_work.iloc[i][start_col] for i in removed_indices if i < len(df_work)]
        all_ends = [kept_row[end_col]] + [df_work.iloc[i][end_col] for i in removed_indices if i < len(df_work)]
        try:
            kept_row[start_col] = min(all_starts)
            kept_row[end_col] = max(all_ends)
        except Exception:
            pass
        return kept_row

    keep = []
    n = len(df_work)
    i = 0

    while i < n:
        current_msg = str(df_work.iloc[i][text_column])
        current_row = df_work.iloc[i].copy()

        # Check against last kept message
        if keep:
            prev_msg = str(keep[-1][text_column])
            ratio = difflib.SequenceMatcher(None, prev_msg, current_msg).ratio()
            if ratio >= similarity_threshold:
                if has_timestamps:
                    keep[-1] = _merge_ts(keep[-1], [i])
                i += 1
                continue

        # Check for 2-message pattern repetition
        if i + 3 < n:
            msg_next = str(df_work.iloc[i + 1][text_column])
            msg_next2 = str(df_work.iloc[i + 2][text_column])
            msg_next3 = str(df_work.iloc[i + 3][text_column])

            pair1 = " ".join([current_msg, msg_next])
            pair2 = " ".join([msg_next2, msg_next3])
            ratio_pattern = difflib.SequenceMatcher(None, pair1, pair2).ratio()
            exact = current_msg == msg_next2 and msg_next == msg_next3

            if ratio_pattern >= similarity_threshold or exact:
                i += 2  # skip the repeated pair
                continue

        keep.append(current_row)
        i += 1

    result = pd.DataFrame(keep) if keep else pd.DataFrame()
    logger.info(f"Stutter removal: {len(df_work)} → {len(result)} messages ({len(df_work) - len(result)} removed)")
    return result.reset_index(drop=True)


def correct_file_with_similarity(
    file_path: str,
    similarity_threshold: float = 0.8,
) -> dict | None:
    """
    Correct a single CSV file.  Writes ``*_corrected.csv`` alongside original.

    Returns:
        Summary dict on correction, or None if no changes needed / error.
    """
    logger.info(f"Processing: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return None

    # Auto-detect text column
    text_col = None
    for candidate in ("text", "message", "transcript", "content", "utterance"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        text_col = df.columns[0]

    corrected = remove_consecutive_repetitive_messages(df, text_col, similarity_threshold)

    if len(corrected) == len(df):
        logger.info(f"No repetitions found in {os.path.basename(file_path)}")
        return None

    corrected_path = file_path.replace(".csv", "_corrected.csv")
    corrected.to_csv(corrected_path, index=False)
    logger.info(f"Corrected file saved: {corrected_path}")

    return {
        "file": os.path.basename(file_path),
        "original_messages": len(df),
        "corrected_messages": len(corrected),
        "messages_removed": len(df) - len(corrected),
        "text_column": text_col,
        "similarity_threshold": similarity_threshold,
    }


def process_folder(
    folder: str,
    similarity_threshold: float = 0.8,
    file_pattern: str = "*.csv",
) -> list[dict]:
    """
    Process all CSV files in *folder*.  Returns list of correction summaries.
    """
    pattern = os.path.join(folder, file_pattern)
    csv_files = [
        f for f in glob.glob(pattern)
        if not any(
            os.path.basename(f).startswith(pfx) for pfx in ("stutter_", "correction_", "similarity_")
        )
        and not os.path.basename(f).endswith(("_corrected.csv", "_backup.csv"))
    ]

    if not csv_files:
        logger.warning("No CSV files found to process")
        return []

    logger.info(f"Found {len(csv_files)} files to process")
    results = []
    for fp in csv_files:
        r = correct_file_with_similarity(fp, similarity_threshold)
        if r:
            results.append(r)

    # Save summary
    if results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(results).to_csv(
            os.path.join(folder, f"similarity_correction_summary_{ts}.csv"), index=False
        )

    return results
