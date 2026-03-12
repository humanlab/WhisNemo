"""
Convert SRT files to formatted CSV.
Ported from src/format_nemo_srt_to_csv.py
"""

import os
import sys
import time

import pandas as pd

from whisnemo.core.timing_utils import setup_timing


def format_srt_to_csv(srt_path: str) -> str:
    """
    Convert an SRT file to a formatted CSV with columns:
    speaker, start_timestamp, end_timestamp, message

    Returns:
        Path to the output CSV file.
    """
    op_file_path = srt_path[:-4] + "_formatted.csv"

    audio_filename, timing_csv, log_timing = setup_timing(srt_path, "SRT to CSV conversion")

    start_time = time.time()
    with open(srt_path, 'r') as file:
        file_content = file.read()
    end_time = time.time()
    log_timing("srt_file_reading", start_time, end_time)

    start_time = time.time()
    segments = file_content.strip().split("\n\n")
    fin_data_arr = []
    for segment in segments:
        lines = segment.split("\n")
        if len(lines) >= 3:
            time_range = lines[1]
            start_timestamp, end_timestamp = time_range.replace(",", ".").split(" --> ")
            text_lines = lines[2:]
            text = " ".join(text_lines)
            speaker, message = text.split(":", maxsplit=1)
            fin_data_arr.append([speaker, start_timestamp, end_timestamp, message])
        else:
            print(f"Something Wrong in this segment: {segment}")
    end_time = time.time()
    log_timing("srt_parsing", start_time, end_time)

    df = pd.DataFrame(fin_data_arr, columns=['speaker', 'start_timestamp', 'end_timestamp', 'message'])
    df.to_csv(op_file_path, index=False)
    print(f"Finished {op_file_path}")
    print(f"[TIMING] Completed SRT to CSV conversion for {audio_filename}")

    return op_file_path


def main():
    """CLI entry point: python -m whisnemo.core.format_srt <srt_file>"""
    if len(sys.argv) < 2:
        print("Usage: python -m whisnemo.core.format_srt <srt_file>")
        sys.exit(1)
    format_srt_to_csv(sys.argv[1])


if __name__ == "__main__":
    main()
