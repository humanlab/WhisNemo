"""``whisnemo-diarize`` — process a single audio file."""
import sys

def main():
    from whisnemo.core.diarize import main as diarize_main
    diarize_main()

if __name__ == "__main__":
    main()
