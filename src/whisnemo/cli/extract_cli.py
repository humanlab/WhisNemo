"""``whisnemo-extract``"""
import argparse, logging

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Extract WhisNemo pipeline outputs")
    parser.add_argument("source_dir")
    parser.add_argument("dest_dir")
    parser.add_argument("--mode", choices=["copy", "move", "symlink"], default="copy")
    args = parser.parse_args()
    from whisnemo.postprocessing.extract_outputs import extract_outputs
    extract_outputs(args.source_dir, args.dest_dir, args.mode)

if __name__ == "__main__":
    main()
