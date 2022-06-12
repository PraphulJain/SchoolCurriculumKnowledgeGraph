import os
import time
from models import models
model = models()

#________________________________

folderPath = "data/"
#________________________________

# Add check for text_files when combining with Ayan's code
if not os.path.isdir(folderPath + 'csv_files/'):
    os.mkdir(folderPath + 'csv_files/')
if not os.path.isdir(folderPath + 'triples_files/'):
    os.mkdir(folderPath + 'triples_files/')
if not os.path.isdir(folderPath + 'kg_files/'):
    os.mkdir(folderPath + 'kg_files/')

files = os.listdir(folderPath+"text_files/")

for file in files:
    if file[-4:] == ".txt":
        txt_file_path = folderPath + 'text_files/' + file
        csv_file_path = folderPath + 'csv_files/' + file[:-4] + '.csv'
        triples_path = folderPath + 'triples_files/' + file[:-4] + '.csv'
        kg_file_path = folderPath + 'kg_files/' + file[:-4] + '.html'

        start_time = time.time()
        #
        model.text_to_csv(txt_file_path, csv_file_path)
        print("time taken to create csv file for " + file + " = " + str(time.time()-stime) + " secs\n")

        stime = time.time()
        #
        model.csv_to_triples(csv_file_path, triples_path)

        #
        model.extract_key_concepts(triples_path, triples_path)
        print("time taken to create Triples for " + file + " = " + str(time.time()-stime) + " secs\n")

        stime = time.time()
        #
        model.triples_to_kg(triples_path, kg_file_path)
        print("time taken to create Knowledge graph for " + file + " = " + str(time.time()-stime) + " secs")
        print("total time taken = " + str(time.time()-start_time) + " secs\n")