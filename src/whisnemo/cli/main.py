"""
``whisnemo`` — top-level CLI.

Usage:
    whisnemo diarize -a audio.mp3
    whisnemo batch -a /path/to/audio_dir --device cuda:0 --start 1 --end 50
    whisnemo extract /source /dest
    whisnemo stutter /path/to/csvs --threshold 0.8
    whisnemo version
"""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="whisnemo",
        description="WhisNemo: Whisper + NeMo MSDD transcription & diarization pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- diarize (single file) ---
    p_diarize = subparsers.add_parser("diarize", help="Process a single audio file")
    p_diarize.add_argument("-a", "--audio", required=True, help="Audio file path")
    p_diarize.add_argument("--no-stem", action="store_false", dest="stemming", default=True)
    p_diarize.add_argument("--whisper-model", default="medium.en")
    p_diarize.add_argument("--batch-size", type=int, default=8)
    p_diarize.add_argument("--language", default=None)
    p_diarize.add_argument("--suppress-numerals", action="store_true", default=False)
    p_diarize.add_argument("--device", default="cuda")

    # --- batch (directory, replaces bash script) ---
    p_batch = subparsers.add_parser("batch", help="Batch-process a directory (replaces bash script)")
    p_batch.add_argument("-a", "--audio-dir", required=True, help="Audio directory")
    p_batch.add_argument("--device", default="cuda")
    p_batch.add_argument("--try-limit", type=int, default=2, help="Max attempts before OOM split-retry")
    p_batch.add_argument("--start", type=int, default=1, help="Start index (1-indexed, inclusive)")
    p_batch.add_argument("--end", type=int, default=None, help="End index (1-indexed, inclusive)")

    # --- extract ---
    p_extract = subparsers.add_parser("extract", help="Extract pipeline outputs into organized dirs")
    p_extract.add_argument("source_dir")
    p_extract.add_argument("dest_dir")
    p_extract.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")

    # --- stutter ---
    p_stutter = subparsers.add_parser("stutter", help="Remove consecutive stutters from CSVs")
    p_stutter.add_argument("folder")
    p_stutter.add_argument("--threshold", type=float, default=0.8)
    p_stutter.add_argument("--pattern", default="*.csv")

    # --- version ---
    subparsers.add_parser("version", help="Print version")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.command == "version":
        from whisnemo import __version__
        print(f"whisnemo {__version__}")

    elif args.command == "diarize":
        from whisnemo.core.diarize import run_diarize
        from whisnemo.core.format_srt import format_srt_to_csv
        import os

        run_diarize(
            audio_path=args.audio,
            stemming=args.stemming,
            model_name=args.whisper_model,
            batch_size=args.batch_size,
            language=args.language,
            suppress_numerals=args.suppress_numerals,
            device=args.device,
        )
        # Also convert SRT to CSV (like your bash script does)
        srt_path = os.path.splitext(args.audio)[0] + ".srt"
        if os.path.isfile(srt_path):
            format_srt_to_csv(srt_path)

    elif args.command == "batch":
        from whisnemo.core.batch import run_batch
        run_batch(
            audio_dir=args.audio_dir,
            device=args.device,
            try_limit=args.try_limit,
            start_idx=args.start,
            end_idx=args.end,
        )

    elif args.command == "extract":
        from whisnemo.postprocessing.extract_outputs import extract_outputs
        extract_outputs(args.source_dir, args.dest_dir, args.mode)

    elif args.command == "stutter":
        from whisnemo.postprocessing.remove_stutters import process_folder
        if not (0.0 <= args.threshold <= 1.0):
            print("Error: threshold must be between 0.0 and 1.0")
            sys.exit(1)
        results = process_folder(args.folder, args.threshold, args.pattern)
        if results:
            total_removed = sum(r["messages_removed"] for r in results)
            print(f"\nCorrected {len(results)} files, removed {total_removed} total messages")
        else:
            print("\nNo corrections needed")


if __name__ == "__main__":
    main()
