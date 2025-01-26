#!/usr/bin/env python3

import os
import re
import time
import logging
import argparse
import torch
import torchaudio
import pandas as pd
from typing import List, Tuple, Union, Optional
import subprocess
import shutil  # For file operations such as copying files.
from pathlib import Path  # For handling file paths and directories.
import numpy as np  # For numerical operations.
from itertools import groupby  # For grouping elements in an iterable.
from datetime import datetime, timedelta
import gc
import sys
import csv

# -----------------------
# NEMO / HELPER IMPORTS
# -----------------------
# Adjust your imports as needed to match your local environment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from helpers import (
    cleanup,
    create_config,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    punct_model_langs,
    whisper_langs,
    write_srt,
)
from transcription_helpers import transcribe_batched

from utils import (
    get_audio_duration_ffmpeg,
    calculate_split_durations,
    split_audio_file,
    read_srt_csv_to_adjust_timestamp,
    write_corrected_timestamps_csv,
    process_updating_timestamps_srt_csv,
    swap_speaker_labels,
    determine_speaker_with_max_questions,
    write_corrected_speaker_labels_csv,
    check_speaker_existence,
    process_speaker_labels,
    concatenate_csv_files,
    get_op_srt_path,
    get_op_csv_path,
    get_op_txt_path,
    is_split_file,
)

# ---------------------------------------------------------------------
# PART 1: COMMON/EXISTING FUNCTIONS (SAME AS YOUR ORIGINAL CODE BASE)
# ---------------------------------------------------------------------

def preprocess_audio_for_diarization(audio_path: str, temp_dir: str, stemming: bool = True) -> str:
    """
    Preprocess the audio file for diarization by isolating vocals using Demucs.
    """
    if stemming:
        return_code = os.system(
            f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "{temp_dir}"'
        )
        if return_code != 0:
            error_message = f"Source splitting failed for {audio_path}. Stopping further processing."
            logging.error(error_message)
            raise RuntimeError(error_message)
        output_path = os.path.join(
            temp_dir, "htdemucs", os.path.splitext(os.path.basename(audio_path))[0], "vocals.wav"
        )
        if not os.path.isfile(output_path):
            raise FileNotFoundError(f"Expected vocals file not found: {output_path}")
        return output_path
    return audio_path


def transcribe_audio(
    audio_file: str,
    language: Optional[str],
    batch_size: int,
    model_name: str,
    suppress_numerals: bool,
    device: str,
) -> Tuple[dict, str, torch.Tensor]:
    """
    Transcribe audio using Whisper model.
    """
    mtypes = {"cpu": "int8", "cuda": "float16"}
    return transcribe_batched(
        audio_file, language, batch_size, model_name, mtypes[device], suppress_numerals, device
    )


def perform_forced_alignment(
    audio_waveform: torch.Tensor,
    whisper_results: dict,
    device: str,
    language: str,
    batch_size: int,
) -> Tuple[List[dict], torch.Tensor]:
    """
    Perform forced alignment on the transcribed audio.
    """
    try:
        alignment_model, alignment_tokenizer, alignment_dictionary = load_alignment_model(
            device, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        processed_audio_waveform = torch.from_numpy(audio_waveform).to(alignment_model.dtype).to(alignment_model.device)

        emissions, stride = generate_emissions(alignment_model, processed_audio_waveform, batch_size=batch_size)

        # Release alignment_model
        del alignment_model
        torch.cuda.empty_cache()

        full_transcript = "".join(segment["text"] for segment in whisper_results)
        tokens_starred, text_starred = preprocess_text(
            full_transcript, romanize=True, language=langs_to_iso[language]
        )
        segments, scores, blank_id = get_alignments(emissions, tokens_starred, alignment_dictionary)
        spans = get_spans(tokens_starred, segments, alignment_tokenizer.decode(blank_id))
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        return word_timestamps, processed_audio_waveform

    except torch.cuda.CudaError as e:
        logging.error(f"CUDA error during forced alignment: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during forced alignment: {e}")
        raise


def map_words_to_speakers(
    word_timestamps: List[dict], speaker_ts: List[List[int]], language: str
) -> List[dict]:
    """
    Map words in the transcription to speakers based on timestamps, adding punctuation if available.
    """
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    if language in punct_model_langs:
        punct_model = PunctuationModel(model="kredor/punctuate-all")
        words_list = list(map(lambda x: x["word"], wsm))
        labeled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
    else:
        logging.warning(
            f"Punctuation restoration is not available for {language}. Using the original punctuation."
        )
    return get_realigned_ws_mapping_with_punctuation(wsm)


def save_results(
    ssm: List[dict], txt_path: str, srt_path: str
) -> None:
    """
    Save transcription results in TXT and SRT formats.
    """
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(srt_path), exist_ok=True)

    with open(txt_path, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(srt_path, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)


def convert_srt_to_csv(srt_path: str, csv_path: str):
    """
    Convert an SRT file to a CSV file without using pandas.
    
    Each SRT segment is expected to have:
      1) A sequence number (ignored).
      2) A time range on a line, e.g. "00:00:01,000 --> 00:00:04,000".
      3) One or more lines of text with a format "Speaker: message".
      
    This function:
      - Reads and splits the file by double newlines to isolate segments.
      - Parses each segment to extract speaker, start_timestamp, end_timestamp, and message.
      - Writes these fields to a CSV file using the Python standard library `csv` module
        which properly handles special characters (like commas) by quoting fields.
    """
    try:
        # 1) Read the entire SRT file content
        with open(srt_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        # 2) Split the content by double newlines to separate segments
        segments = file_content.strip().split("\n\n")
        
        fin_data_arr = []

        # 3) Parse each segment
        for segment in segments:
            lines = segment.split("\n")
            
            # Need at least 3 lines: sequence#, time range, and text
            if len(lines) >= 3:
                try:
                    # The second line should be the time range
                    time_range = lines[1]
                    # Replace commas with periods, then split on ' --> '
                    start_timestamp, end_timestamp = time_range.replace(",", ".").split(" --> ")
                    
                    # The remaining lines (index 2 onward) should be the text
                    text_lines = lines[2:]
                    text = " ".join(text_lines).strip()
                    
                    # Check if there's a colon to separate speaker and message
                    if ":" in text:
                        speaker, message = text.split(":", maxsplit=1)
                        speaker = speaker.strip()
                        message = message.strip()

                        # Append to our final data list
                        fin_data_arr.append([speaker, start_timestamp.strip(), end_timestamp.strip(), message])
                    else:
                        logging.warning(f"No colon found in text; skipping segment:\n{segment}")
                except ValueError as ve:
                    logging.warning(f"Skipping malformed segment:\n{segment}\nError: {ve}")
            else:
                logging.warning(f"Skipping malformed segment:\n{segment}")

        # 4) Write to CSV using the standard library csv module (handles commas/special chars)
        with open(csv_path, "w", encoding="utf-8", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Write the header
            writer.writerow(["speaker", "start_timestamp", "end_timestamp", "message"])
            
            # Write each row of parsed data
            for row in fin_data_arr:
                # row = [speaker, start_timestamp, end_timestamp, message]
                # Make sure none of these are empty or None, if you want to skip incomplete ones
                if all(row):
                    writer.writerow(row)

        logging.info(f"CSV saved at {csv_path}")

    except FileNotFoundError:
        logging.error(f"SRT file not found: {srt_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while converting SRT to CSV: {e}")
        raise


def read_diarization_output(temp_dir: str) -> List[List[int]]:
    """
    Read the diarization output from the temporary directory (RTTM file).
    """
    speaker_ts = []
    rttm_file = os.path.join(temp_dir, "pred_rttms", "mono_file.rttm")

    try:
        with open(rttm_file, "r") as f:
            for line in f.readlines():
                line_list = line.split(" ")
                start = int(float(line_list[5]) * 1000)  # sec to ms
                end = start + int(float(line_list[8]) * 1000)
                speaker_id = int(line_list[11].split("_")[-1])
                speaker_ts.append([start, end, speaker_id])
        logging.info(f"Read diarization output: {rttm_file}")
    except FileNotFoundError:
        logging.error(f"RTTM file not found: {rttm_file}")
        raise
    except Exception as e:
        logging.error(f"Error reading diarization output: {e}")
        raise

    return speaker_ts


def process_splits(audio_file: str, args: dict) -> None:
    """
    Split an audio file and process the resulting splits if OOM can't be resolved on the entire file.
    If a split file also fails (e.g., OOM again), abort further splitting, clean up, and return.
    """
    audio_file_path = Path(audio_file)
    temp_dir = audio_file_path.parent / f"{audio_file_path.stem}_oomretry"
    os.makedirs(temp_dir, exist_ok=True)

    NUM_SPLITS = 2
    split_files = split_audio_file(audio_file, temp_dir, NUM_SPLITS)
    logging.info(f"Split audio file into {NUM_SPLITS} part(s): {split_files}")

    csv_audio_file_mapping = {}
    split_failed = False

    for split_file in split_files:
        logging.info(f"Processing split file: {split_file}")
        split_audio_output_dir = os.path.dirname(split_file) if args["output_dir"] is None else args["output_dir"]
        split_audio_file_csv_path = get_op_csv_path(split_file, split_audio_output_dir)
        
        try:
            # Attempt to process the split
            process_retry(str(split_file), args)
            csv_audio_file_mapping[split_audio_file_csv_path] = str(split_file)

        except MemoryError as e:
            logging.error(f"OOM in split file '{split_file}'. Aborting further splitting. Error: {e}")
            split_failed = True
            break  # Stop processing other splits
        except Exception as e:
            logging.error(f"Unexpected error in processing split file '{split_file}'. Aborting. Error: {e}")
            split_failed = True
            break
        # If no exception, we proceed to the next split

    if split_failed:
        # If any split failed, optionally do partial cleanup
        logging.info(f"Split handling failed for '{audio_file}'. Cleaning up partial artifacts...")
        
        # Remove the temp_dir with partially processed files
        logging.info(f"Cleaning up split temp dir: {temp_dir}")
        cleanup(temp_dir)
        
        # If you have partial CSVs, remove them
        for partial_file in csv_audio_file_mapping.keys():
            logging.info(f"Removing partial file: {partial_file}")
            cleanup(partial_file)
        
        return  # Return early, skipping final concatenation

    # If we reach here, all splits succeeded. We can proceed with final steps:
    logging.info(f"CSV Files & Split Audio File Mapping: {csv_audio_file_mapping}")

    # Update split transcripts timestamps
    update_timestamps_prev_suffix = "_formatted.csv"
    update_timestamps_new_suffix = "_formatted_timestamp_updated.csv"
    updated_timestamps_csv_audio_dict = process_updating_timestamps_srt_csv(
        csv_audio_file_mapping, 
        update_timestamps_prev_suffix, 
        update_timestamps_new_suffix
    )
    logging.info(f"Timestamp Updated CSV Files & Split Audio File Mapping: {updated_timestamps_csv_audio_dict}")
    
    # Update speaker labels
    update_speaker_labels_prev_suffix = "_formatted_timestamp_updated.csv"
    update_speaker_labels_new_suffix = "_formatted_speaker_label.csv"
    updated_speaker_labels_csv_audio_dict = process_speaker_labels(
        updated_timestamps_csv_audio_dict,
        update_speaker_labels_prev_suffix,
        update_speaker_labels_new_suffix
    )
    logging.info(f"Speaker Updated CSV Files & Split Audio File Mapping: {updated_speaker_labels_csv_audio_dict}")
    
    # Concatenate CSV Files
    concatenate_csv_files_prev_suffix = "_split_1_formatted_speaker_label.csv"
    concatenate_csv_files_new_suffix = "_formatted.csv"
    final_concatenated_csv = concatenate_csv_files(
        updated_speaker_labels_csv_audio_dict, 
        concatenate_csv_files_prev_suffix, 
        concatenate_csv_files_new_suffix
    )
    logging.info(f"Final Concatenated CSV file: {final_concatenated_csv}")
    
    # Cleanup oomretry dir
    logging.info(f"Cleaning up: {temp_dir}")
    cleanup(temp_dir)

    # Cleanup intermediate Split File Outputs
    cleanup_files = list(csv_audio_file_mapping.keys()) 
    cleanup_files += list(updated_timestamps_csv_audio_dict.keys()) 
    cleanup_files += list(updated_speaker_labels_csv_audio_dict.keys())

    for file in cleanup_files:
        logging.info(f"Cleaning up: {file}")
        cleanup(file)

    logging.info(f"CUDA OOM handling complete for {audio_file}. Final CSV File: {final_concatenated_csv}")

# ---------------------------------------------------------------------
# PART 2: SUBPROCESS WORKER MODE FOR DIARIZATION
# ---------------------------------------------------------------------

def diarize_worker(temp_dir: str, device: str):
    """
    Subprocess worker:
     - Loads 'audio_waveform.pt' from `temp_dir`.
     - Runs NeMo MSDD diarization.
     - Exits with code 0 on success, 1 on error (including OOM).
    """
    try:
        waveform_path = os.path.join(temp_dir, "audio_waveform.pt")
        audio_waveform = torch.load(waveform_path)  # shape: (channels, samples)

        rttm_dir = os.path.join(temp_dir, "pred_rttms")
        os.makedirs(rttm_dir, exist_ok=True)

        # Save the audio waveform to "mono_file.wav"
        torchaudio.save(
            os.path.join(temp_dir, "mono_file.wav"),
            audio_waveform.cpu().unsqueeze(0).float(),
            16000,
            channels_first=True,
        )

        msdd_model = None
        try:
            msdd_model = NeuralDiarizer(cfg=create_config(temp_dir)).to(device)
            msdd_model.diarize()

            rttm_file = os.path.join(temp_dir, "pred_rttms", "mono_file.rttm")
            if not os.path.isfile(rttm_file):
                raise FileNotFoundError(f"RTTM file not generated: {rttm_file}")

            print("Diarization completed successfully.")

        except RuntimeError as e:
            # Catch a typical OOM
            if "CUDA out of memory" in str(e):
                logging.error("CUDA OOM encountered in diarize_worker.")
            raise

        finally:
            if msdd_model is not None:
                del msdd_model
            del audio_waveform
            gc.collect()
            torch.cuda.empty_cache()

    except Exception as e:
        logging.exception("Error in diarize_worker.")
        sys.exit(1)  # Non-zero exit code means failure

    # If everything succeeded, exit code 0
    sys.exit(0)


def run_diarization_in_subprocess(audio_waveform: torch.Tensor, temp_dir: str, device: str) -> bool:
    """
    Main process function:
      - Saves the waveform to 'audio_waveform.pt'.
      - Spawns this script again in worker mode with --diarize-worker.
      - Returns True on success, False if OOM or other errors.
    """
    os.makedirs(temp_dir, exist_ok=True)

    # Save waveform so worker can load it
    wf_path = os.path.join(temp_dir, "audio_waveform.pt")
    torch.save(audio_waveform, wf_path)

    # Prepare subprocess command to re-run this same file in worker mode
    cmd = [
        sys.executable,
        __file__,           # The same file
        "--diarize-worker", # Flag to indicate worker mode
        "--temp-dir", temp_dir,
        "--device", device
    ]

    logging.info(f"Launching diarization worker subprocess: {cmd}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    output_lines = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            output_lines.append(line)
            print(line, end="")  # Show logs in real-time (optional)

    return_code = proc.wait()
    combined_output = "".join(output_lines)

    if return_code != 0:
        if "out of memory" in combined_output or "OOM" in combined_output:
            logging.error("Worker failed with CUDA OOM.")
        else:
            logging.error("Worker failed with a different error.")
        return False

    logging.info("Worker completed successfully.")
    return True

# ---------------------------------------------------------------------
# PART 3: MAIN PROCESSING LOGIC (TRANSCRIBE, ALIGN, DIARIZE, RETRIES)
# ---------------------------------------------------------------------

def process_file(audio_file: str, args: dict) -> None:
    """
    Process a single audio file for transcription, alignment, and diarization
    (with the diarization step in a subprocess).
    """
    output_dir = os.path.dirname(audio_file) if args["output_dir"] is None else args["output_dir"]
    txt_path = get_op_txt_path(audio_file, output_dir)
    srt_path = get_op_srt_path(audio_file, output_dir)
    csv_path = get_op_csv_path(audio_file, output_dir)

    # Unique temp directory for the entire pipeline
    temp_dir = os.path.join(os.getcwd(), f"temp_outputs_{time.strftime('%Y%m%d_%H%M%S')}")

    os.makedirs(temp_dir, exist_ok=True)
    try:
        # 1. Optional Preprocessing with Demucs
        vocal_target = preprocess_audio_for_diarization(audio_file, temp_dir, args["stemming"])
        logging.info(f"Preprocessing completed for file: {audio_file}")

        # 2. Transcription with Whisper
        whisper_results, language, audio_waveform = transcribe_audio(
            vocal_target, args["language"], args["batch_size"], args["model_name"], 
            args["suppress_numerals"], args["device"]
        )
        logging.info(f"Transcription completed for file: {audio_file}")

        # 3. Forced Alignment
        word_timestamps, audio_waveform = perform_forced_alignment(
            audio_waveform, whisper_results, args["device"], language, batch_size=args["batch_size"]
        )
        logging.info(f"Forced alignment completed for file: {audio_file}")

        # 4. Diarization in a Subprocess
        diar_success = run_diarization_in_subprocess(audio_waveform, temp_dir, args["device"])
        if not diar_success:
            # Raise a MemoryError so we can catch it specifically for OOM logic
            raise MemoryError("Diarization worker failed (OOM or other error).")

        # 5. Speaker Mapping
        speaker_ts = read_diarization_output(temp_dir)
        wsm = map_words_to_speakers(word_timestamps, speaker_ts, language)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        # 6. Save Results
        save_results(ssm, txt_path, srt_path)
        convert_srt_to_csv(srt_path, csv_path)
        logging.info(f"Results saved for file: {audio_file}")

    finally:
        # Always clean up the temp folder
        cleanup(temp_dir)


def process_retry(audio_file: str, args: dict) -> None:
    """
    Manage retries for processing a single audio file
    (including detection of OOM and possibly splitting the audio).
    """
    retries = 0
    max_retries = args["max_retries"] if not is_split_file(audio_file) else args["max_oom_retries"]
    cuda_oom_error = False

    while retries < max_retries:
        try:
            process_file(audio_file, args)
            return
        except MemoryError as e:
            # We interpret MemoryError as a "CUDA OOM or worker out of memory"
            logging.warning(f"OOM detected for {audio_file}: {e}. Retrying...")
            torch.cuda.empty_cache()
            cuda_oom_error = True
            retries += 1
        except FileNotFoundError as e:
            # Typically no retry on missing file
            logging.error(f"File-related error for {audio_file}: {e}. Skipping retries.")
            raise
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")
            retries += 1
        finally:
            logging.info(f"Retry {retries}/{max_retries} completed for {audio_file}.")

    # If we've exhausted retries, and it's truly an OOM scenario, attempt splitting
    if retries == max_retries and not is_split_file(audio_file):
        logging.warning(f"Final OOM handling for {audio_file} - attempting to split audio.")
        process_splits(audio_file, args)


def process_audio_files(audio_files: List[str], args: dict) -> None:
    """
    Process multiple audio files. Skip files that already have final CSV unless --rerun-completed-audio-file is set.
    """
    for audio_file in audio_files:
        output_csv = get_op_csv_path(audio_file, args["output_dir"])
        if args["rerun_completed_audio_file"] or not os.path.isfile(output_csv):
            logging.info(f"Processing audio file: {audio_file}")
            process_retry(audio_file, args)
        else:
            logging.info(f"Skipping already completed file: {audio_file}")


def main(args: dict) -> None:
    """
    Main entry point for processing audio files.
    """
    supported_audio_formats = (".wav", ".mp3", ".m4a")  # Extend if needed
    if os.path.isdir(args["audio"]):
        audio_files = [
            os.path.join(args["audio"], f)
            for f in os.listdir(args["audio"])
            if f.endswith(supported_audio_formats)
        ]
    else:
        audio_files = args["audio"].split(",")
        print(f"Files to be diarized are: {audio_files}")

    process_audio_files(audio_files, args)


# ---------------------------------------------------------------------
# PART 4: SCRIPT ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # If we have the special flag for worker mode, run diarize_worker
    if "--diarize-worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--diarize-worker", action="store_true")
        parser.add_argument("--temp-dir", required=True)
        parser.add_argument("--device", default="cuda")
        sub_args = parser.parse_args()
        diarize_worker(temp_dir=sub_args.temp_dir, device=sub_args.device)
        sys.exit(0)

    # Otherwise, parse normal arguments for the main pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio", required=True, 
                        help="Target audio file or directory.")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Directory to save results. Default: same as audio file's directory.")
    parser.add_argument("--no-stem", action="store_false", dest="stemming", default=True,
                        help="Disable source separation with Demucs.")
    parser.add_argument("--suppress_numerals", action="store_true", dest="suppress_numerals", default=False,
                        help="Suppress numerical digits in transcripts (converting to words).")
    parser.add_argument("--whisper-model", dest="model_name", default="medium.en",
                        help="Name of the Whisper model to use.")
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=8,
                        help="Batch size for batched inference.")
    parser.add_argument("--max-retries", type=int, dest="max_retries", default=1,
                        help="Max Retries Limit before splitting on OOM.")
    parser.add_argument("--max-oom-retries", type=int, dest="max_oom_retries", default=2,
                        help="Max Retries after audio is split on OOM.")
    parser.add_argument("--language", type=str, default=None, choices=whisper_langs,
                        help="Language spoken in the audio. None = auto-detect.")
    parser.add_argument("--device", dest="device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="If you have a GPU, use 'cuda', else 'cpu'.")
    parser.add_argument("--rerun-completed-audio-file", action="store_true", 
                        dest="rerun_completed_audio_file", default=False,
                        help="Rerun pipeline for already completed audio files.")
    
    args = parser.parse_args()
    args_dict = vars(args)  # Convert Namespace to dictionary
    print(args_dict) 
    logging.basicConfig(level=logging.INFO)

    main(args_dict)
