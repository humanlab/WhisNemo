import pandas as pd
import re
import sys
file_path = sys.argv[1] #"/cronus_data/araghavan/InterviewTranscriberData/mock/mock_recording_001_whisnemo.srt" #sys.argv[1]
op_file_path = file_path[:-4] + "_formatted.csv"
with open(file_path, 'r') as file:
    file_content = file.read()


segments = file_content.strip().split("\n\n")
fin_data_arr = []
for segment in segments:
    lines = segment.split("\n")
    if len(lines) >= 3:
        time_range = lines[1]
        start_timestamp, end_timestamp = time_range.replace(",",".").split(" --> ")
        text_lines = lines[2:]
        text = " ".join(text_lines)
        speaker, message = text.split(":", maxsplit=1)
        fin_data_arr.append([speaker, start_timestamp, end_timestamp, message])
    else:
        print(f"Something Wrong in In this segment: {segment}")


df = pd.DataFrame(fin_data_arr, columns=['speaker', 'start_timestamp', 'end_timestamp', 'message'])
df.to_csv(op_file_path, index=False)
print(f"Finished {op_file_path}")