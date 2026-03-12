#!/usr/bin/env python3
import time
import csv
import os
import sys
from datetime import datetime
from contextlib import contextmanager

def setup_timing(file_path, process_name="processing"):
    """
    Setup timing for a Python script
    """
    audio_filename = os.path.splitext(os.path.basename(file_path))[0]
    for suffix in ["_whisnemo", "_formatted", "_split"]:
        audio_filename = audio_filename.replace(suffix, "")

    timing_dir = os.path.join(os.path.dirname(file_path), "timing_logs")
    os.makedirs(timing_dir, exist_ok=True)
    timing_csv = os.path.join(timing_dir, f"{audio_filename}_timing.csv")

    attempt_num = int(os.environ.get('WHISNEMO_ATTEMPT', '1'))

    if not os.path.exists(timing_csv):
        with open(timing_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'attempt_num', 'block_name', 'start_time', 'end_time', 'duration_seconds', 'timestamp'])

    def log_timing(block_name, start_time, end_time):
        duration = end_time - start_time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(timing_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([audio_filename, attempt_num, block_name, start_time, end_time, f"{duration:.2f}", timestamp])
        print(f"[TIMING] {block_name}: {duration:.2f}s")

    print(f"[TIMING] Starting {process_name} for {audio_filename} - Attempt {attempt_num}")

    return audio_filename, timing_csv, log_timing


class TimingLogger:
    def __init__(self, output_dir, filename):
        self.output_dir = output_dir
        self.filename = filename
        self.attempt_num = 1
        self.current_blocks = {}
        self.timing_data = []
        self.csv_file = os.path.join(output_dir, f"{filename}_timing.csv")
        self.ensure_csv_header()

    def ensure_csv_header(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'attempt_num', 'block_name', 'start_time', 'end_time', 'duration_seconds', 'timestamp'])

    def set_attempt(self, attempt_num):
        self.attempt_num = attempt_num

    @contextmanager
    def time_block(self, block_name):
        start_time = time.time()
        start_timestamp = datetime.now().isoformat()
        print(f"[TIMING] {self.filename} - Attempt {self.attempt_num} - Starting {block_name} at {start_timestamp}")

        try:
            yield
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.time()
            end_timestamp = datetime.now().isoformat()
            duration = end_time - start_time
            print(f"[TIMING] {self.filename} - Attempt {self.attempt_num} - Finished {block_name} at {end_timestamp} (Duration: {duration:.2f}s)")
            self.write_timing_record(block_name, start_time, end_time, duration, start_timestamp)

    def write_timing_record(self, block_name, start_time, end_time, duration, timestamp):
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.filename, self.attempt_num, block_name,
                start_time, end_time, f"{duration:.2f}", timestamp
            ])


_bash_event_times = {}

def log_bash_event(timing_dir, event_type, filename, attempt_num="1", additional_info=""):
    os.makedirs(timing_dir, exist_ok=True)
    csv_file = os.path.join(timing_dir, "batch_timing.csv")

    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'event_type', 'filename', 'attempt_num', 'start_time', 'end_time', 'duration_seconds', 'additional_info'])

    current_time = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    event_key = f"{filename}_{attempt_num}_{event_type.replace('_end', '').replace('_start', '')}"

    if event_type.endswith('_start'):
        _bash_event_times[event_key] = current_time
        duration = ""
        end_time = ""
        start_time = current_time
        print(f"[TIMING] {event_type}: {filename} (attempt {attempt_num}) started at {timestamp}")
    elif event_type.endswith('_end'):
        start_time = _bash_event_times.get(event_key, current_time)
        end_time = current_time
        duration = f"{end_time - start_time:.2f}"
        print(f"[TIMING] {event_type}: {filename} (attempt {attempt_num}) completed - Duration: {duration}s")
        _bash_event_times.pop(event_key, None)
    else:
        start_time = current_time
        end_time = ""
        duration = ""
        print(f"[TIMING] {event_type}: {filename} (attempt {attempt_num}) at {timestamp}")

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, event_type, filename, attempt_num,
            start_time if start_time else "",
            end_time if end_time else "",
            duration, additional_info
        ])


def main():
    if len(sys.argv) < 5:
        print("Usage: python timing_utils.py log_event <timing_dir> <event_type> <filename> <attempt_num> [additional_info]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "log_event":
        timing_dir = sys.argv[2]
        event_type = sys.argv[3]
        filename = sys.argv[4]
        attempt_num = sys.argv[5] if len(sys.argv) > 5 else "1"
        additional_info = sys.argv[6] if len(sys.argv) > 6 else ""
        log_bash_event(timing_dir, event_type, filename, attempt_num, additional_info)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
