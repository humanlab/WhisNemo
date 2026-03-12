import os
import csv
import ffmpeg
import logging
from pathlib import Path
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
    split_duration = audio_duration / num_splits
    splits = [(i * split_duration, split_duration) for i in range(num_splits)]
    return splits


def split_audio_file(audio_file_path, output_dir, num_splits):
    audio_duration = get_audio_duration_ffmpeg(audio_file_path)
    if audio_duration is not None:
        print(f"Duration of the audio file is {audio_duration:.2f} seconds.")

    split_durations = calculate_split_durations(audio_duration, num_splits)

    audio_file_path = Path(audio_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_files = [
        output_dir / f"{audio_file_path.stem}__split_{i+1}.wav"
        for i in range(num_splits)
    ]

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


def read_srt_csv_to_adjust_timestamp(audio_srt_csv_path, time_delta):
    updated_rows = []

    with open(audio_srt_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        fieldnames = reader.fieldnames

        for row in reader:
            try:
                start_time = datetime.strptime(row['start_timestamp'], '%H:%M:%S.%f') + time_delta
                end_time = datetime.strptime(row['end_timestamp'], '%H:%M:%S.%f') + time_delta

                row['start_timestamp'] = start_time.strftime('%H:%M:%S.%f')[:-3]
                row['end_timestamp'] = end_time.strftime('%H:%M:%S.%f')[:-3]

                updated_rows.append(row)
            except ValueError as e:
                logging.warning(f"Skipping row due to malformed timestamp. Row data: {row}. Error: {e}")

    return fieldnames, updated_rows


def write_corrected_timestamps_csv(fieldnames, rows, file_path, prev_suffix, new_suffix):
    new_file_path = file_path.replace(prev_suffix, new_suffix)
    with open(new_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, delimiter=',',
            quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)

    return new_file_path


def process_updating_timestamps_srt_csv(file_mapping, prev_suffix, new_suffix):
    cumulative_time_delta = timedelta(0)
    updated_mapping = {}

    for csv_file, audio_file_path in file_mapping.items():
        print(f"Processing file: {csv_file}")
        try:
            fieldnames, updated_rows = read_srt_csv_to_adjust_timestamp(csv_file, cumulative_time_delta)
            corrected_timestamp_csv = write_corrected_timestamps_csv(fieldnames, updated_rows, csv_file, prev_suffix, new_suffix)
            updated_mapping[corrected_timestamp_csv] = audio_file_path

            audio_duration = get_audio_duration_ffmpeg(audio_file_path)
            cumulative_time_delta += timedelta(seconds=audio_duration)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

    print(f"Finished updating timestamps for: \"{', '.join(file_mapping.keys())}\"")
    return updated_mapping


def swap_speaker_labels(rows, swap_mapping):
    for row in rows:
        if 'speaker' in row:
            row['speaker'] = swap_mapping.get(row['speaker'], row['speaker'])
    return rows


def determine_speaker_with_max_questions(rows):
    question_counts = {}
    for row in rows:
        if 'message' not in row or 'speaker' not in row:
            logging.warning(f"Skipping row due to missing 'message' or 'speaker': {row}")
            continue
        if "?" in row['message']:
            speaker = row['speaker']
            question_counts[speaker] = question_counts.get(speaker, 0) + 1

    if not question_counts:
        return None

    max_speaker = max(question_counts, key=question_counts.get)
    return max_speaker


def write_corrected_speaker_labels_csv(rows, fieldnames, file_path, prev_suffix, new_suffix):
    new_file_path = file_path.replace(prev_suffix, new_suffix)

    with open(new_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, delimiter=',',
            quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for row in rows:
            clean_row = {field: row.get(field, "") for field in fieldnames}
            writer.writerow(clean_row)

    return new_file_path


def check_speaker_existence(rows, speaker_label):
    return any(row['speaker'] == speaker_label for row in rows)


def process_speaker_labels(file_mapping, prev_suffix, new_suffix):
    ground_truth_max_speaker = None
    speaker_labels_swap_flag = False
    speaker_1_check_flag = True

    updated_mapping = {}

    for i, (csv_file, audio_file_path) in enumerate(file_mapping.items()):
        print(f"Processing file: {csv_file}")

        with open(csv_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(
                infile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            if not reader.fieldnames:
                logging.error(f"No header found in CSV: {csv_file}")
                continue

            fieldnames = reader.fieldnames
            rows = list(reader)

        if not rows:
            logging.warning(f"No data found in CSV: {csv_file}")
            continue

        if i == 0:
            ground_truth_max_speaker = determine_speaker_with_max_questions(rows)
            speaker_1_check_flag = check_speaker_existence(rows, "Speaker 1")

            if speaker_1_check_flag:
                swap_mapping = {"Speaker 0": "Speaker 1", "Speaker 1": "Speaker 0"}
            else:
                swap_mapping = {"Speaker 1": "Speaker 0"}

            corrected_speaker_labels_csv = write_corrected_speaker_labels_csv(
                rows, fieldnames, csv_file, prev_suffix, new_suffix
            )
            updated_mapping[corrected_speaker_labels_csv] = audio_file_path
            continue

        current_max_speaker = determine_speaker_with_max_questions(rows)

        if current_max_speaker != ground_truth_max_speaker and current_max_speaker is not None:
            speaker_labels_swap_flag = True
            rows = swap_speaker_labels(rows, swap_mapping)

        corrected_speaker_labels_csv = write_corrected_speaker_labels_csv(
            rows, fieldnames, csv_file, prev_suffix, new_suffix
        )
        updated_mapping[corrected_speaker_labels_csv] = audio_file_path

    print(f"Finished Speaker Labels Swaps Process for: \"{', '.join(file_mapping.keys())}\"")
    print(f"Labels were swapped: {speaker_labels_swap_flag}, Speaker 1 was present in the first file: {speaker_1_check_flag}")

    return updated_mapping


def concatenate_csv_files(csv_files_audio_path_dict, prev_suffix, new_suffix):
    csv_files = list(csv_files_audio_path_dict.keys())

    if not csv_files:
        raise ValueError("No CSV files provided for concatenation.")

    all_rows = []
    headers = None

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(
                    infile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                try:
                    file_headers = next(reader)
                except StopIteration:
                    logging.warning(f"CSV file '{csv_file}' is empty. Skipping.")
                    continue

                if headers is None:
                    headers = file_headers
                else:
                    if file_headers != headers:
                        raise ValueError(
                            f"Headers in '{csv_file}' do not match.\n"
                            f"Expected: {headers}\nGot:      {file_headers}"
                        )

                for row in reader:
                    if len(row) != len(headers):
                        logging.warning(
                            f"Row in '{csv_file}' has {len(row)} columns, expected {len(headers)}. Row: {row}"
                        )
                    all_rows.append(row)
        except FileNotFoundError:
            logging.error(f"CSV file not found: {csv_file}")
            raise

    if not headers:
        raise ValueError("No valid CSV files with headers found to concatenate.")

    output_file = csv_files[0].replace(prev_suffix, new_suffix)

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(
            outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(headers)
        writer.writerows(all_rows)

    return output_file


def get_op_srt_path(audio_path: str, output_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(output_dir, f"{base_name}.srt")


def get_op_csv_path(audio_path: str, output_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(output_dir, f"{base_name}_formatted.csv")


def get_op_txt_path(audio_path: str, output_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(output_dir, f"{base_name}.txt")


def is_split_file(file_name: str) -> bool:
    return "__split_" in Path(file_name).stem
