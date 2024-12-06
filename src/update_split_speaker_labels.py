import sys
import os
import pandas as pd
from datetime import timedelta
import ffmpeg

def swap(df, swap_mapping):
    # Swap speaker 0 to speaker 1 and vice versa
    df['speaker'] = df['speaker'].map(swap_mapping)
    return df


def determine_speaker_with_max_questions(df):
    counter_speakers = df[df['message'].str.contains("\?")].groupby('speaker').size().reset_index(name='count')
    max_speaker = counter_speakers.loc[counter_speakers['count'].idxmax()].speaker
    return max_speaker


def write_corrected_speaker_labels_csv(df, file_path, prev_suffix, new_suffix):
    new_file_path = file_path.replace(prev_suffix, new_suffix)
    df.to_csv(new_file_path, index=False)


def determine_speaker_1_exist(df, value_to_check):
    return value_to_check in df['speaker'].values


def main():
    # Get command line arguments
    if len(sys.argv) != 4:
        print("Usage: script.py <csv_file1,csv_file2> <prev_suffix> <new_suffix>")
        print("""Sample: python script.py 'PP516_20231016_JM_split_1_timestamp_updated_formatted.csv,PP516_20231016_JM_split_2_timestamp_updated_formatted.csv' '_timestamp_updated_formatted.csv' '_speaker_label_formatted.csv'""")
        sys.exit(1)
    
    csv_files = sys.argv[1].split(',')
    prev_suffix = sys.argv[2]
    new_suffix = sys.argv[3]
    
    ground_truth_max_speaker = None
    speaker_labels_swap_flag = False
    speaker_1_check_flag = True
    # Process each CSV file and calculate cumulative duration
    for i, csv_file in enumerate(csv_files):
        # Apply the cumulative time delta to the CSV file
        df = pd.read_csv(csv_file)

        if i == 0:
            ground_truth_max_speaker = determine_speaker_with_max_questions(df)
            speaker_1_check_flag = determine_speaker_1_exist(df, "Speaker 1")
            if speaker_1_check_flag:
                swap_mapping = {"Speaker 0": "Speaker 1", "Speaker 1": "Speaker 0"}
            else:
                swap_mapping = {"Speaker 1": "Speaker 0"}
            write_corrected_speaker_labels_csv(df, csv_file, prev_suffix, new_suffix)
            continue
        
        curr_df_max_speaker = determine_speaker_with_max_questions(df)
        if ground_truth_max_speaker != curr_df_max_speaker:
            speaker_labels_swap_flag = True
            df = swap(df, swap_mapping)
        
        write_corrected_speaker_labels_csv(df, csv_file, prev_suffix, new_suffix)
        
    
    print(f"Finished Speaker Labels Swaps Process for: \"{sys.argv[1]}\" And Were Swapped: {speaker_labels_swap_flag} And Had Both Speakers: {speaker_1_check_flag}")
    

if __name__ == "__main__":
    main()
