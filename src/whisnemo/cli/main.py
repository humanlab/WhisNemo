"""
``whisnemo`` — top-level CLI.
"""

import argparse
import logging
import sys


def _add_diarize_args(parser):
    """Shared diarize arguments for both diarize and batch commands."""
    # Audio processing
    parser.add_argument("--no-stem", action="store_false", dest="stemming", default=True)
    parser.add_argument("--whisper-model", default="medium.en")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--language", default=None)
    parser.add_argument("--suppress-numerals", action="store_true", default=False)
    parser.add_argument("--device", default="cuda")
    # Output formats
    parser.add_argument("--formats", nargs="+", default=["csv"],
                        choices=["csv", "txt", "srt"],
                        help="Output formats (default: csv)")
    # Stutter removal
    parser.add_argument("--remove-stutters", action="store_true", default=False,
                        help="Run stutter removal on CSV output")
    parser.add_argument("--stutter-threshold", type=float, default=0.8)
    # NeMo config
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--no-oracle-speakers", action="store_false",
                        dest="oracle_num_speakers", default=True,
                        help="Let NeMo auto-detect speaker count")
    parser.add_argument("--vad-model", default="vad_multilingual_marblenet")
    parser.add_argument("--speaker-model", default="titanet_large")
    parser.add_argument("--onset", type=float, default=0.8)
    parser.add_argument("--offset", type=float, default=0.5)
    parser.add_argument("--pad-offset", type=float, default=-0.05)
    parser.add_argument("--domain-type", default="telephonic",
                        choices=["telephonic", "meeting", "general"])


def main():
    parser = argparse.ArgumentParser(
        prog="whisnemo",
        description="WhisNemo: Whisper + NeMo MSDD transcription & diarization pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- diarize ---
    p_diarize = subparsers.add_parser("diarize", help="Process a single audio file")
    p_diarize.add_argument("-a", "--audio", required=True)
    _add_diarize_args(p_diarize)

    # --- batch ---
    p_batch = subparsers.add_parser("batch", help="Batch-process a directory")
    p_batch.add_argument("-a", "--audio-dir", required=True)
    p_batch.add_argument("--try-limit", type=int, default=2)
    p_batch.add_argument("--start", type=int, default=1)
    p_batch.add_argument("--end", type=int, default=None)
    p_batch.add_argument("--organize", action="store_true", default=False,
                         help="Organize outputs into subfolders after batch")
    _add_diarize_args(p_batch)

    # --- extract ---
    p_extract = subparsers.add_parser("extract", help="Extract outputs into organized dirs")
    p_extract.add_argument("source_dir")
    p_extract.add_argument("dest_dir")
    p_extract.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")

    # --- stutter ---
    p_stutter = subparsers.add_parser("stutter", help="Remove stutters from CSVs")
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
        run_diarize(
            audio_path=args.audio,
            stemming=args.stemming,
            model_name=args.whisper_model,
            batch_size=args.batch_size,
            language=args.language,
            suppress_numerals=args.suppress_numerals,
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

    elif args.command == "batch":
        from whisnemo.core.batch import run_batch
        run_batch(
            audio_dir=args.audio_dir,
            device=args.device,
            try_limit=args.try_limit,
            start_idx=args.start,
            end_idx=args.end,
            organize=args.organize,
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
            stemming=args.stemming,
            model_name=args.whisper_model,
            batch_size=args.batch_size,
            language=args.language,
            suppress_numerals=args.suppress_numerals,
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
            total = sum(r["messages_removed"] for r in results)
            print(f"\nCorrected {len(results)} files, removed {total} total messages")
        else:
            print("\nNo corrections needed")


if __name__ == "__main__":
    main()
