import os
import time
from models import models

model = models()

########################################################
# extract_key_concepts


# ___________________________________

folder_path = 'Path here'
# ___________________________________

#store file names in list by using os.listdir on target folder
k = os.listdir(folder_path)
tot_pages = 0

if not os.path.isdir(folder_path + 'text_files/'):
    os.mkdir(folder_path + 'text_files/')
if not os.path.isdir(folder_path + 'csv_files/'):
    os.mkdir(folder_path + 'csv_files/')
if not os.path.isdir(folder_path + 'triples_files/'):
    os.mkdir(folder_path + 'triples_files/')
if not os.path.isdir(folder_path + 'kg_files/'):
    os.mkdir(folder_path + 'kg_files/')

#run loop to generate corres. text files of PDFs and store in text_files folder inside target_folder
for i in range(len(k)):
    if(k[i][-4:] == '.pdf'): #since there is text_file folder in same folder, we check for pdf and then process
        txt_file_path = folder_path + 'text_files/' + k[i][:-4] + '.txt'
        #print(k[i] + " and " + txt_file_path)
        pdf_file_path = folder_path + k[i]
        csv_file_path = folder_path + 'csv_files/' + k[i][:-4] + '.csv'
        triples_path = folder_path + 'triples_files/' + k[i][:-4] + '.csv'
        kg_file_path = folder_path + 'kg_files/' + k[i][:-4] + '.html'
        
        start_time = time.time()

        #a new text file is created and flushed. needs to be only when text file is yet to be created.
        file = open(txt_file_path,"w+") #opens a fresh txt file at given path
        file.write("")
        file.close()

        tot_pages = tot_pages + model.pdf_to_text(pdf_file_path,txt_file_path,2) # 3rd param is no of cols. feed it MANUALLY. either 1 or 2
        print("time taken to create text file for " + k[i] + " = " + str(time.time()-start_time) + " secs\n")

        stime = time.time()
        #
        model.text_to_csv(txt_file_path, csv_file_path)
        print("time taken to create csv file for " + k[i] + " = " + str(time.time()-stime) + " secs\n")

        stime = time.time()
        #
        model.csv_to_triples(csv_file_path, triples_path)

        #
        model.extract_key_concepts(triples_path, triples_path)
        print("time taken to create Triples for " + k[i] + " = " + str(time.time()-stime) + " secs\n")

        stime = time.time()
        #
        model.triples_to_kg(triples_path, kg_file_path)
        print("time taken to create Knowledge graph for " + k[i] + " = " + str(time.time()-stime) + " secs")
        print("total time taken = " + str(time.time()-start_time) + " secs\n")

#Linking knowledge graphs by hyperlinks
stime = time.time()
folder_path = folder_path + "/kg_files/"
k = os.listdir(folder_path)
k.sort()
dlinks = [] #location of html files in drive folder.
for i in range(len(k)):
  file_path = folder_path + k[i]
  dlinks.append(file_path)

model.link_html_files(dlinks,dlinks, k)
print("Time taken to create hyperlinks = " + str(time.time()-stime))
