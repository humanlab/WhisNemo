#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 cuda_device_map audio_file1 [audio_file2 ...] [try_limit=2]"
    echo "Example: ./whisnemo_specific_v1.0.sh 0 file1.wav file2.mp4 file3.wav 1"
    echo "The above runs the script on the specified audio files using CUDA device 0, with a try limit of 1 (default is 2)."
    exit 1
fi

script_dir="$(dirname "$(realpath "$0")")"

# Function to wrap up OOM splits
wrap_up_oom_splits() {
    local l_temp_dir="$1"
    local l_temp_done_dir="$2"

    # Delete temp directories
    rm -rf "$l_temp_dir"
    rm -rf "$l_temp_done_dir"
    echo "Deleted temporary directories"
}

# Function to retry OOM fail file
retry_oom_fail_file() {
    local file="$1"
    local file_name="$(basename "${file}")"
    local file_name_wo_extension="${file_name%.*}"
    local max_splits=2
    local l_try_limit=2
    local l_attempt_counter=0

    # Create Temp Directories
    local l_temp_dir="${file}_oomretry"
    local l_temp_done_dir="${file}_oomretry_runstatus"
    mkdir -p "${l_temp_dir}" "${l_temp_done_dir}"
    echo "Created Temp Retry: ${l_temp_dir}, Runstatus directories: ${l_temp_done_dir}"

    # Copy main audio file to temp dir
    cp "${file}" "${l_temp_dir}" \
    && echo "Copied main audio file: ${file} to Temp Retry directory"
    local new_main_audio_file="${l_temp_dir}/${file_name}"
    local main_audio_done_file="$(dirname "${file}")_runstatus/${file_name}_run.done"

    # Retry twice splitting files
    while [ $l_attempt_counter -le $l_try_limit ]; do
        echo "Splitting Attempt Counter: ${l_attempt_counter}"
        local split_audio_prefix="${file_name_wo_extension}_split_"
        
        # Get duration of main audio
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${new_main_audio_file}")
        echo "Duration of Main Audio File: ${new_main_audio_file} is: ${duration}" 
        # Calculate split duration
        split_duration=$(echo "${duration} / ${max_splits}" | bc -l)
        
        # Split audio files
        for i in $(seq 1 $max_splits); do
            start_time=$(echo "($i - 1) * ${split_duration}" | bc -l)
            
            if [ $i -eq $max_splits ]; then
                # For the last split, don't specify duration
                ffmpeg -y -i "${new_main_audio_file}" -ss ${start_time} -c copy "${l_temp_dir}/${split_audio_prefix}${i}.wav"
            else
                # For all other splits, specify duration
                ffmpeg -y -i "${new_main_audio_file}" -ss ${start_time} -t ${split_duration} -c copy "${l_temp_dir}/${split_audio_prefix}${i}.wav"
            fi
            echo "Duration of File: ${l_temp_dir}/${split_audio_prefix}${i}.wav" \
            && echo "$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${l_temp_dir}/${split_audio_prefix}${i}.wav")"
        done

        # Remove old .done files
        rm -rf "${l_temp_done_dir}/${split_audio_prefix}"*.done
        echo "Removed previously created .done files"

        # Process split files
        for i in $(seq 1 $max_splits); do
            split_file="${l_temp_dir}/${split_audio_prefix}${i}.wav"
            split_done_file="${l_temp_done_dir}/${split_audio_prefix}${i}.done"
            
            CUDA_VISIBLE_DEVICES=${cuda_device_map} python "${script_dir}/../whisper-diarization/diarize.py" -a "${split_file}" \
            && touch "${split_done_file}" &
        done

        # Wait for all background jobs to finish
        wait
        echo "Finished whisper-diarization of split files: $(ls -1 "${l_temp_dir}/${split_audio_prefix}"*.wav)"
        ls -1 "${l_temp_done_dir}/"
        echo "Listed .done dir"
        
        # Check if all splits were processed successfully
        if [ $(ls -1 "${l_temp_done_dir}"/*.done | wc -l) -eq ${max_splits} ]; then
            echo "Split Retry Attempt: ${l_attempt_counter}; Finished diarizing splits"
            break
        fi

        l_attempt_counter=$((l_attempt_counter + 1))
    done

    # Process SRT files
    for srt_file in ${l_temp_dir}/*.srt; do
        python "${script_dir}/format_nemo_srt_to_csv.py" "${srt_file}"
    done

    # Update timestamps and speaker labels
    python "${script_dir}/update_split_transcripts_timestamps.py" "$(ls ${l_temp_dir}/*_formatted.csv | sort -V | tr '\n' ',' | sed 's/,$//')" '_formatted.csv' '_formatted_timestamp_updated.csv' \
    && python "${script_dir}/update_split_speaker_labels.py" "$(ls ${l_temp_dir}/*_formatted_timestamp_updated.csv | sort -V | tr '\n' ',' | sed 's/,$//')" '_formatted_timestamp_updated.csv' '_formatted_speaker_label.csv' \
    && python "${script_dir}/combine_split_transcripts.py" "$(ls ${l_temp_dir}/*_formatted_speaker_label.csv | sort -V | tr '\n' ',' | sed 's/,$//')" '_split_1_formatted_speaker_label.csv' '_formatted.csv' \
    && cp "${l_temp_dir}/${file_name_wo_extension}_formatted.csv" "$(dirname "${file}")" \
    && touch "${main_audio_done_file}" 
    
    wrap_up_oom_splits "$l_temp_dir" "$l_temp_done_dir"
}

# Main script starts here
cuda_device_map="$1"
try_limit="${!#}" # Last argument is the try limit if provided
shift

# Process each audio file
for file in "$@"; do
    if [ ! -f "$file" ]; then
        echo "Skipping invalid file path: $file"
        continue
    fi

    file_name="$(basename "${file}")"
    done_dir="$(dirname "${file}")_runstatus/"
    mkdir -p "$done_dir"
    done_file="${done_dir}${file_name}_run.done"
    srt_file="${file%.*}.srt"

    if [ -f "${done_file}" ]; then
        echo "Skipping File: ${file} as it was already completed before"
        continue
    fi

    echo "Started File: ${file_name}"

    attempt_counter=1
    while [ $attempt_counter -le $try_limit ]; do
        echo "Attempt: ${attempt_counter}; File: ${file}"

        CUDA_VISIBLE_DEVICES=${cuda_device_map} python "${script_dir}/../whisper-diarization/diarize.py" -a "${file}" && \
        touch "${done_file}" && \
        python "${script_dir}/format_nemo_srt_to_csv.py" "${srt_file}"


        if [ $? -eq 0 ]; then
            echo "Finished File: ${file_name}"
            break
        else
            echo "At Attempt: ${attempt_counter}; File Failed: ${file}"
            attempt_counter=$((attempt_counter + 1))
        fi
    done

    if [ $attempt_counter -gt $try_limit ]; then
        echo "Attempted More than try limit of: ${try_limit}"
        echo "Splitting File and retrying"
        retry_oom_fail_file "${file}"
    fi
done

echo "Finished processing all files."
