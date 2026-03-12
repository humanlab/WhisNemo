"""``whisnemo-stutter``"""
import argparse, logging, sys

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Remove stutters from transcript CSVs")
    parser.add_argument("folder")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--pattern", default="*.csv")
    args = parser.parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        print("Error: threshold must be between 0.0 and 1.0")
        sys.exit(1)
    from whisnemo.postprocessing.remove_stutters import process_folder
    results = process_folder(args.folder, args.threshold, args.pattern)
    if results:
        total = sum(r["messages_removed"] for r in results)
        print(f"\nCorrected {len(results)} files, removed {total} total messages")
    else:
        print("\nNo corrections needed")

if __name__ == "__main__":
    main()
