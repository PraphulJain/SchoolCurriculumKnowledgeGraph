import json
import pandas as pd
import os

json_data = json.load(open('5th Grade_Transcripts_Normal.json'))

txt_data = ""

for element in json_data:
    txt_data += element[1]

with open("5th Grade_Transcripts_Normal.txt", "w") as txt_file:
    txt_file.write(txt_data)