"""``whisnemo-batch`` — batch-process a directory of audio files."""
import argparse
import logging
import sys

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="WhisNemo batch processing (replaces bash script)",
        epilog="Example: whisnemo-batch -a ../data --device cuda:0 --try-limit 2 --start 1 --end 50"
    )
    parser.add_argument("-a", "--audio-dir", required=True, help="Audio directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--try-limit", type=int, default=2, help="Max attempts before OOM split-retry")
    parser.add_argument("--start", type=int, default=1, help="Start index (1-indexed)")
    parser.add_argument("--end", type=int, default=None, help="End index (1-indexed)")

    args = parser.parse_args()

    from whisnemo.core.batch import run_batch
    run_batch(
        audio_dir=args.audio_dir,
        device=args.device,
        try_limit=args.try_limit,
        start_idx=args.start,
        end_idx=args.end,
    )

if __name__ == "__main__":
    main()
