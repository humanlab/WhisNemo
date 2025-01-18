import os
import csv
import ffmpeg
import logging
from pathlib import Path  # For handling file paths and directories.
from datetime import datetime, timedelta

def get_audio_duration_ffmpeg(audio_file_path):
    try:
        probe = ffmpeg.probe(audio_file_path)
        duration = float(probe['format']['duration'])
        return duration
    except ffmpeg.Error as e:
        logging.error(f"Error occurred: {e.stderr.decode()}")
        raise
    

def calculate_split_durations(audio_duration, num_splits):
    """
    Calculate the start times and duration of each split.
    
    Parameters:
        audio_duration (float): Total duration of the audio in seconds.
        num_splits (int): Number of splits to create.
    
    Returns:
        List[Tuple[float, float]]: A list of (start_time, duration) tuples for each split.
    """
    split_duration = audio_duration / num_splits
    splits = [(i * split_duration, split_duration) for i in range(num_splits)]
    return splits


def split_audio_file(audio_file_path, output_dir, num_splits):
    """
    Splits an audio file into equal parts using `ffmpeg-python`.
    
    Parameters:
        audio_file_path (str): Path to the input audio file.
        output_dir (str): Directory to save the split audio files.
        num_splits (int): Number of splits to create.
    
    Returns:
        List[Path]: List of paths to the split audio files.
    """
    audio_duration = get_audio_duration_ffmpeg(audio_file_path)
    if audio_duration is not None:
        print(f"Duration of the audio file is {audio_duration:.2f} seconds.")
        
    # Calculate split start times and durations
    split_durations = calculate_split_durations(audio_duration, num_splits)
    
    # Prepare output directory and paths
    audio_file_path = Path(audio_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_files = [
        output_dir / f"{audio_file_path.stem}__split_{i+1}.wav" 
        for i in range(num_splits)
    ]

    # Split the audio file
    for (start_time, duration), split_file in zip(split_durations, split_files):
        try:
            ffmpeg.input(str(audio_file_path), ss=start_time, t=duration).output(
                str(split_file), codec="copy"
            ).run(overwrite_output=True)
        except ffmpeg.Error as e:
            print(f"Error splitting audio: {e.stderr.decode()}")
            return []

    print(f"Audio successfully split into {num_splits} parts.")
    return split_files


# def read_srt_csv_to_adjust_timestamp(audio_srt_csv_path, time_delta):
#     """Process a CSV file and adjust timestamps."""
#     updated_rows = []

#     # Read and process each row
#     with open(audio_srt_csv_path, 'r') as csvfile:
#         reader = csv.DictReader(csvfile, quotechar='"')
#         fieldnames = reader.fieldnames  # Capture field names for writing later
        
#         for row in reader:
#             # Adjust timestamps
#             start_time = datetime.strptime(row['start_timestamp'], '%H:%M:%S.%f') + time_delta
#             end_time = datetime.strptime(row['end_timestamp'], '%H:%M:%S.%f') + time_delta
            
#             # Format adjusted timestamps back to string
#             row['start_timestamp'] = start_time.strftime('%H:%M:%S.%f')[:-3]  # Keep milliseconds
#             row['end_timestamp'] = end_time.strftime('%H:%M:%S.%f')[:-3]      # Keep milliseconds
#             updated_rows.append(row)
    
#     return fieldnames, updated_rows

# def write_corrected_timestamps_csv(fieldnames, rows, file_path, prev_suffix, new_suffix):
#     """Write updated rows to a new CSV file."""
#     new_file_path = file_path.replace(prev_suffix, new_suffix)
#     with open(new_file_path, 'w', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quotechar='"')
#         writer.writeheader()
#         writer.writerows(rows)

#     return new_file_path

# def process_updating_timestamps_srt_csv(file_mapping, prev_suffix, new_suffix):
#     """
#     Process a dictionary of CSV file paths (keys) and their corresponding audio file paths (values) 
#     to update timestamps.
#     """
#     cumulative_time_delta = timedelta(0)  # Initial cumulative time delta is zero

#     updated_mapping = {}

#     for csv_file, audio_file_path in file_mapping.items():
#         print(f"Processing file: {csv_file}")
#         try:
#             # Apply the cumulative time delta to the CSV file
#             fieldnames, updated_rows = read_srt_csv_to_adjust_timestamp(csv_file, cumulative_time_delta)
#             corrected_timestamp_csv = write_corrected_timestamps_csv(fieldnames, updated_rows, csv_file, prev_suffix, new_suffix)
#             updated_mapping[corrected_timestamp_csv] = audio_file_path

#             # Get the duration of the corresponding audio file
#             audio_duration = get_audio_duration_ffmpeg(audio_file_path)
#             cumulative_time_delta += timedelta(seconds=audio_duration)
#         except FileNotFoundError as e:
#             print(f"Error: {e}")
#             raise
         
#     print(f"Finished updating timestamps for: \"{', '.join(file_mapping.keys())}\"")
#     return updated_mapping


def read_srt_csv_to_adjust_timestamp(audio_srt_csv_path, time_delta):
    """
    Read a CSV file (with columns [speaker, start_timestamp, end_timestamp, message]),
    parse timestamps, adjust them by `time_delta`, and return updated rows.
    """
    updated_rows = []

    # Read and process each row
    with open(audio_srt_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        fieldnames = reader.fieldnames  # Capture field names for writing later
        
        for row in reader:
            # If timestamps might be malformed, consider wrapping in try/except
            try:
                start_time = datetime.strptime(row['start_timestamp'], '%H:%M:%S.%f') + time_delta
                end_time = datetime.strptime(row['end_timestamp'], '%H:%M:%S.%f') + time_delta

                # Format adjusted timestamps back to string (keeping milliseconds)
                row['start_timestamp'] = start_time.strftime('%H:%M:%S.%f')[:-3]  # e.g. 00:01:23.456
                row['end_timestamp'] = end_time.strftime('%H:%M:%S.%f')[:-3]
                
                updated_rows.append(row)
            except ValueError as e:
                logging.warning(f"Skipping row due to malformed timestamp. Row data: {row}. Error: {e}")
    
    return fieldnames, updated_rows


def write_corrected_timestamps_csv(fieldnames, rows, file_path, prev_suffix, new_suffix):
    """
    Write updated rows to a new CSV file, replacing `prev_suffix` in the filename with `new_suffix`.
    """
    new_file_path = file_path.replace(prev_suffix, new_suffix)
    with open(new_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(
            csvfile, 
            fieldnames=fieldnames, 
            delimiter=',', 
            quotechar='"', 
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)

    return new_file_path


def process_updating_timestamps_srt_csv(file_mapping, prev_suffix, new_suffix):
    """
    Process a dictionary of CSV file paths (keys) and their corresponding audio file paths (values) 
    to update timestamps cumulatively across multiple files.
    """
    cumulative_time_delta = timedelta(0)  # Initial cumulative time delta

    updated_mapping = {}

    for csv_file, audio_file_path in file_mapping.items():
        print(f"Processing file: {csv_file}")
        try:
            # Read and adjust timestamps
            fieldnames, updated_rows = read_srt_csv_to_adjust_timestamp(csv_file, cumulative_time_delta)
            
            # Write out the corrected timestamps CSV
            corrected_timestamp_csv = write_corrected_timestamps_csv(fieldnames, updated_rows, csv_file, prev_suffix, new_suffix)
            updated_mapping[corrected_timestamp_csv] = audio_file_path

            # Get the duration of the corresponding audio file
            audio_duration = get_audio_duration_ffmpeg(audio_file_path)
            cumulative_time_delta += timedelta(seconds=audio_duration)
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise  # Re-raise to stop execution or handle upstream if needed

    print(f"Finished updating timestamps for: \"{', '.join(file_mapping.keys())}\"")
    return updated_mapping


def swap_speaker_labels(rows, swap_mapping):
    """
    Swap speaker labels in the rows based on the given mapping.
    If a speaker is not found in the mapping, it remains unchanged.
    """
    for row in rows:
        # Only swap if 'speaker' is present in the row
        if 'speaker' in row:
            row['speaker'] = swap_mapping.get(row['speaker'], row['speaker'])
    return rows


def determine_speaker_with_max_questions(rows):
    """
    Determine the speaker with the most questions in the rows.
    Returns None if no questions are found at all.
    """
    question_counts = {}
    for row in rows:
        # Validate that 'message' and 'speaker' keys exist
        if 'message' not in row or 'speaker' not in row:
            logging.warning(f"Skipping row due to missing 'message' or 'speaker': {row}")
            continue
        
        # Check if the message contains a question mark
        if "?" in row['message']:
            speaker = row['speaker']
            question_counts[speaker] = question_counts.get(speaker, 0) + 1
    
    if not question_counts:
        return None  # No questions found in any row
    
    # Find the speaker with the maximum questions
    max_speaker = max(question_counts, key=question_counts.get)
    return max_speaker


def write_corrected_speaker_labels_csv(rows, fieldnames, file_path, prev_suffix, new_suffix):
    """
    Write the updated speaker labels to a new CSV file, ensuring
    that special characters are handled correctly by using a DictWriter 
    with proper quoting.
    """
    new_file_path = file_path.replace(prev_suffix, new_suffix)
    
    # Write the CSV with proper quoting and UTF-8 encoding
    with open(new_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(
            csvfile, 
            fieldnames=fieldnames, 
            delimiter=',', 
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        
        for row in rows:
            # Optional: ensure only known fieldnames are written (in case of extra keys)
            # You can do something like:
            clean_row = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(clean_row)
            
            # # Or simply write the entire row if guaranteed the columns match:
            # writer.writerow(row)
    
    return new_file_path

def check_speaker_existence(rows, speaker_label):
    """Check if a specific speaker label exists in the rows."""
    return any(row['speaker'] == speaker_label for row in rows)


def process_speaker_labels(file_mapping, prev_suffix, new_suffix):
    """
    Process the CSV files and ensure consistent speaker labeling.
    
    Steps:
    1) The first file is used to establish the 'ground_truth_max_speaker'.
    2) Check if "Speaker 1" exists in the first file to decide the swap mapping.
    3) For subsequent files, if the speaker with the most questions differs 
       from ground_truth, swap the labels.
    4) Write out the updated rows to a new CSV file with the new suffix.
    
    Parameters:
        file_mapping (dict): { csv_file_path : audio_file_path }
        prev_suffix (str) : The suffix on the input files to replace.
        new_suffix (str) : The suffix for the output files.
        
    Returns:
        updated_mapping (dict): { new_csv_path : audio_file_path }
    """
    ground_truth_max_speaker = None
    speaker_labels_swap_flag = False
    speaker_1_check_flag = True

    updated_mapping = {}

    for i, (csv_file, audio_file_path) in enumerate(file_mapping.items()):
        print(f"Processing file: {csv_file}")
        
        # Read the CSV file (with robust settings)
        with open(csv_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(
                infile,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            # Ensure fieldnames exist
            if not reader.fieldnames:
                logging.error(f"No header found in CSV: {csv_file}")
                continue
            
            fieldnames = reader.fieldnames
            rows = list(reader)
        
        # If the CSV is empty or missing crucial columns, skip
        if not rows:
            logging.warning(f"No data found in CSV: {csv_file}")
            continue
        
        # The first file sets the ground truth
        if i == 0:
            ground_truth_max_speaker = determine_speaker_with_max_questions(rows)
            speaker_1_check_flag = check_speaker_existence(rows, "Speaker 1")
            
            # Define the swap mapping
            # If "Speaker 1" exists, we swap "Speaker 0" <-> "Speaker 1"
            # Otherwise, we only remap "Speaker 1" -> "Speaker 0" 
            # (in case there is a label "Speaker 1" in subsequent files)
            if speaker_1_check_flag:
                swap_mapping = {"Speaker 0": "Speaker 1", "Speaker 1": "Speaker 0"}
            else:
                swap_mapping = {"Speaker 1": "Speaker 0"}
            
            # Write out the first file as-is (defining ground truth)
            corrected_speaker_labels_csv = write_corrected_speaker_labels_csv(
                rows, fieldnames, csv_file, prev_suffix, new_suffix
            )
            updated_mapping[corrected_speaker_labels_csv] = audio_file_path
            continue
        
        # For subsequent files, check if we need to swap
        current_max_speaker = determine_speaker_with_max_questions(rows)
        
        # If there's a mismatch with the ground truth, do the swap
        if current_max_speaker != ground_truth_max_speaker and current_max_speaker is not None:
            speaker_labels_swap_flag = True
            rows = swap_speaker_labels(rows, swap_mapping)
        
        # Write the updated rows
        corrected_speaker_labels_csv = write_corrected_speaker_labels_csv(
            rows, fieldnames, csv_file, prev_suffix, new_suffix
        )
        updated_mapping[corrected_speaker_labels_csv] = audio_file_path

    print(f"Finished Speaker Labels Swaps Process for: \"{', '.join(file_mapping.keys())}\"")
    print(f"Labels were swapped: {speaker_labels_swap_flag}, Speaker 1 was present in the first file: {speaker_1_check_flag}")

    return updated_mapping


def concatenate_csv_files(csv_files_audio_path_dict, prev_suffix, new_suffix):
    """
    Concatenate multiple CSV files and save the combined output to a new file, 
    without using pandas.

    Parameters:
    -----------
    csv_files_audio_path_dict : dict
        A dictionary whose keys are CSV file paths and values are associated audio file paths.
    prev_suffix : str
        The suffix of the input files (e.g., "_timestamp_updated_formatted.csv").
    new_suffix : str
        The suffix for the output file (e.g., "_concatenated.csv").

    Returns:
    --------
    str
        The path to the output concatenated CSV file.
    """
    # Gather the CSV file paths
    csv_files = list(csv_files_audio_path_dict.keys())

    if not csv_files:
        raise ValueError("No CSV files provided for concatenation.")

    all_rows = []
    headers = None

    # Read each CSV file and append its rows
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(
                    infile,
                    delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL
                )

                try:
                    # Attempt to read the header row
                    file_headers = next(reader)
                except StopIteration:
                    logging.warning(f"CSV file '{csv_file}' is empty. Skipping.")
                    continue  # Skip empty files

                # Check if headers is still None; if so, accept this file's header as the standard
                if headers is None:
                    headers = file_headers
                else:
                    # Ensure the current file's header matches the previously established headers
                    if file_headers != headers:
                        raise ValueError(
                            f"Headers in '{csv_file}' do not match the previously read headers.\n"
                            f"Expected: {headers}\nGot:      {file_headers}"
                        )

                # Append all rows (data) except the header
                for row in reader:
                    # Optional: check row length matches the number of headers
                    if len(row) != len(headers):
                        logging.warning(
                            f"Row in '{csv_file}' has a different number of columns ({len(row)}) "
                            f"than expected ({len(headers)}). Row data: {row}"
                        )
                    all_rows.append(row)
        except FileNotFoundError as e:
            logging.error(f"CSV file not found: {csv_file}")
            raise
        except Exception as e:
            logging.error(f"Error reading '{csv_file}': {e}")
            raise

    if not headers:
        raise ValueError("No valid CSV files with headers found to concatenate.")

    # Determine the output file path
    output_file = csv_files[0].replace(prev_suffix, new_suffix)

    # Write the concatenated rows to the output file
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(
                outfile,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            writer.writerow(headers)  # Write the header row
            writer.writerows(all_rows)  # Write all concatenated rows
    except Exception as e:
        logging.error(f"Failed to write concatenated CSV '{output_file}': {e}")
        raise

    return output_file


def get_op_srt_path(audio_path: str, output_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    srt_path = os.path.join(output_dir, f"{base_name}.srt")
    return srt_path


def get_op_csv_path(audio_path: str, output_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    csv_path = os.path.join(output_dir, f"{base_name}_formatted.csv")
    return csv_path


def get_op_txt_path(audio_path: str, output_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    return txt_path


def is_split_file(file_name: str) -> bool:
    """Determine if the file is a split file based on its name."""
    return "__split_" in Path(file_name).stem