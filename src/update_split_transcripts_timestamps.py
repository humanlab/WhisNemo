import sys
import os
import pandas as pd
from datetime import timedelta
import ffmpeg

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds using ffmpeg."""
    probe = ffmpeg.probe(file_path)
    duration = float(probe['format']['duration'])
    return timedelta(seconds=duration)

def find_audio_file(csv_file, prev_suffix):
    """Find the corresponding audio file for a given CSV."""
    base_name = csv_file.replace(prev_suffix, "")
    extensions = ['.wav', '.mp3', '.m4a']
    
    for ext in extensions:
        audio_file = base_name + ext
        print(f"Searching: {audio_file}")
        if os.path.exists(audio_file):
            return audio_file
    
    raise FileNotFoundError(f"Audio file not found for {csv_file}")

def process_csv(file_path, time_delta):
    """Read a CSV, add the time delta to start and end timestamps."""
    df = pd.read_csv(file_path)
    
    # Assuming 'start_timestamp' and 'end_timestamp' columns are in HH:MM:SS.sss format
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], format='%H:%M:%S.%f') + time_delta
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], format='%H:%M:%S.%f') + time_delta
    
    # Convert back to HH:MM:SS.sss format
    df['start_timestamp'] = df['start_timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]  # Keep milliseconds
    df['end_timestamp'] = df['end_timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]    # Keep milliseconds
    
    return df

def write_corrected_timestamps_csv(df, file_path, prev_suffix, new_suffix):
    new_file_path = file_path.replace(prev_suffix, new_suffix)
    df.to_csv(new_file_path, index=False)


def main():
    # Get command line arguments
    if len(sys.argv) != 4:
        print("Usage: script.py <csv_file1,csv_file2> <prev_suffix> <new_suffix>")
        print("""Sample: python script.py 'PP516_20231016_JM_split_1_formatted.csv,PP516_20231016_JM_split_2_formatted.csv' '_formatted.csv' '_timestamp_updated_formatted.csv'""")
        sys.exit(1)
    
    csv_files = sys.argv[1].split(',')
    prev_suffix = sys.argv[2]
    new_suffix = sys.argv[3]
    
    cumulative_time_delta = timedelta(0)  # Initial cumulative time delta is zero
    print(sys.argv[1])
    # Process each CSV file and calculate cumulative duration
    for i, csv_file in enumerate(csv_files):
        print(f"Trying File: {csv_file}")
        # Apply the cumulative time delta to the CSV file
        df = process_csv(csv_file, cumulative_time_delta)
        write_corrected_timestamps_csv(df, csv_file, prev_suffix, new_suffix)

        # Find the corresponding audio file and get its duration
        try:
            audio_file_path = find_audio_file(csv_file, prev_suffix)
            audio_duration = get_audio_duration(audio_file_path)
            cumulative_time_delta += audio_duration
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
    
    print(f"Finished Updating Timestamps for: \"{sys.argv[1]}\"")
    

if __name__ == "__main__":
    main()
