import sys
import os
import pandas as pd
from datetime import timedelta
import ffmpeg

def main():
    # Get command line arguments
    if len(sys.argv) != 4:
        print("Usage: script.py <csv_files> <prev_suffix> <new_suffix>")
        sys.exit(1)
    
    csv_files = sys.argv[1].split(',')
    prev_suffix = sys.argv[2]
    new_suffix = sys.argv[3]
    
    all_dataframes = []
    
    # Process each CSV file and calculate cumulative duration
    for i, csv_file in enumerate(csv_files):
        # Apply the cumulative time delta to the CSV file
        df = pd.read_csv(csv_file)
        all_dataframes.append(df)
        
    # Concatenate all dataframes
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Write the concatenated dataframe to the output file
    output_file = csv_files[0].replace(prev_suffix, new_suffix)
    final_df.to_csv(output_file)

if __name__ == "__main__":
    main()
