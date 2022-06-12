# GENERIC IMPORTS
from bs4 import BeautifulSoup as BS
from pyvis.network import Network
from spacyEntityLinker import EntityLinker
import spacy
from openie import StanfordOpenIE
from nltk import tokenize
import time
import math
import requests
import io
import os
import re
import numpy as np
import pandas as pd

# IMPORTS FOR PDF2TEXT
from pdf2image import convert_from_path
from PIL import Image
import cv2
import pytesseract

# IMPORTS FOR TEXT2CSV
import nltk
import spacy
import neuralcoref
import csv
nltk.download('punkt')


class models:
    
    def pdf_to_text(self, pdf_file_path, text_file_path, no_of_cols):
        '''
        Argument: path of pdf, path of text file, no of cols

        does: image to np array, tesserract ocr on np array, writing output into text file 

        Returns: no of pages in the pdf - so as to keep track of total no of pages processed in the entire program

        '''
        images = convert_from_path(pdf_file_path)
        print('# pages in this chapter: ' +
            str(len(images)) + '; Pages done: ', end=' ')
        for idx in range(len(images)):
            np_im = np.array(images[idx])
            imgray = cv2.cvtColor(np_im, cv2.COLOR_BGR2GRAY)
            ret, without_pb = cv2.threshold(imgray, 200, 255, 0)

            file = open(text_file_path, "a")  # append mode
            if no_of_cols == 1:
                text = pytesseract.image_to_string(without_pb)
                file.write(text)

            if no_of_cols == 2:
                np_im1 = without_pb[:, :int(without_pb.shape[1]/2)]
                text = pytesseract.image_to_string(np_im1)
                file.write(text)

                np_im2 = without_pb[:, int(without_pb.shape[1]/2):]
                text = pytesseract.image_to_string(np_im2)
                file.write(text)

            file.close
            print(idx+1, end=' ')
        print('\n')

        return len(images)


    # reading file contents and storing it as one string
    def coref_resolution(self, text):
        """Function that executes coreference resolution on a given text"""
        nlp = spacy.load(
            'en_core_web_sm')  # using sm for now. change to lg later if need be

        # session crashes when this is run. check versions and resolve later.
        neuralcoref.add_to_pipe(nlp)

        doc = nlp(text)
        # fetches tokens with whitespaces from spacy document
        tok_list = list(token.text_with_ws for token in doc)
        for cluster in doc._.coref_clusters:
            # get tokens from representative cluster name
            cluster_main_words = set(cluster.main.text.split(' '))
            for coref in cluster:
                if coref != cluster.main:  # if coreference element is not the representative element of that cluster
                    if coref.text != cluster.main.text and bool(set(coref.text.split(' ')).intersection(cluster_main_words)) == False:
                        # if coreference element text and representative element text are not equal and none of the coreference element words are in representative element. This was done to handle nested coreference scenarios
                        tok_list[coref.start] = cluster.main.text + \
                            doc[coref.end-1].whitespace_
                        for i in range(coref.start+1, coref.end):
                            tok_list[i] = ""

        return "".join(tok_list)


    # POST SEGMENTATION AND FORMATTING
    def text_to_csv(self, txt_file_path, csv_file_path):
        file = open(txt_file_path, 'r+')
        k = file.readlines()
        text = ''
        for i in range(len(k)):
            text += k[i]
        new_text = self.coref_resolution(text)
        # print(new_text)
        sent = tokenize.sent_tokenize(new_text)
        print('no of sentences/rows in csv file: ' + str(len(sent)))
        for i in range(len(sent)):
            sent[i] = sent[i].replace('\n', ' ')

        fields = ['sentence']  # temporarily only one field.
        rows = []
        for i in range(len(sent)):
            tem = []
            tem.append(sent[i])
            rows.append(tem)

        with open(csv_file_path, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)  # creating a csv writer object
            csvwriter.writerow(fields)      # writing the fields
            csvwriter.writerows(rows)  # writing the data rows

        print('\n' + csv_file_path + ' created!!')


    def csv_to_triples(self, csv_file_path, save_path):
        sentences = pd.read_csv(csv_file_path).values.tolist()

        data = []
        client = StanfordOpenIE()
        for sentence in sentences:
            triples = client.annotate(sentence[0])
            for triple in triples:
                data.append([sentence[0], triple['subject'],
                            triple['relation'], triple['object']])

        #print(pd.DataFrame(data, columns=['Sentence', 'Subject', 'Relation', 'Object'])[:10])
        pd.DataFrame(data, columns=['Sentence', 'Subject', 'Relation', 'Object']).to_csv(
            save_path, index=False)


    def extract_key_concepts(self, filepath, savepath):

        newdf = {'Sentence': [], 'Subject': [], 'Relation': [], 'Object': [
        ], 'Subject_keys': [], 'Object_keys': [], 'Subject_flag': [], 'Object_flag': []}

        entityLinker = EntityLinker()
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe(entityLinker, last=True, name="entityLinker")

        df = pd.read_csv(filepath)

        cur_sent = ""
        for row in df.iterrows():
            subs = nlp(row[1]['Subject'])
            objs = nlp(row[1]['Object'])
            sublist = []
            objlist = []
            sub_entities = subs._.linkedEntities
            for sent in sub_entities:
                sublist.append(sent.get_label())

            obj_entities = objs._.linkedEntities
            for sent in obj_entities:
                objlist.append(sent.get_label())

            if len(sublist) == 0:
                sublist.append(row[1]['Subject'])
            if len(objlist) == 0:
                objlist.append(row[1]['Object'])

            if len(sublist) > 0 and len(objlist) > 0:
                combinations = [(x, y) for x in sublist for y in objlist]

                if cur_sent != row[1]['Sentence']:
                    cur_sent = row[1]['Sentence']
                    flagsubdone = []
                    flagobjdone = []
                for i in range(len(combinations)):
                    newdf['Sentence'].append(row[1]['Sentence'])
                    newdf['Subject'].append(row[1]['Subject'])
                    newdf['Relation'].append(row[1]['Relation'])
                    newdf['Object'].append(row[1]['Object'])
                    newdf['Subject_keys'].append(combinations[i][0])
                    newdf['Object_keys'].append(combinations[i][1])
                    subflag = 1 if combinations[i][0] not in flagsubdone else 0
                    objflag = 1 if combinations[i][1] not in flagobjdone else 0
                    flagsubdone.append(combinations[i][0])
                    flagobjdone.append(combinations[i][1])

                    newdf['Subject_flag'].append(subflag)
                    newdf['Object_flag'].append(objflag)

        newdata = pd.DataFrame(newdf)

        newdata.to_csv(savepath)

        return savepath


    def triples_to_kg(self, triples_path, save_path):
        df = pd.read_csv(triples_path)

        x_df = df[['Subject_keys', 'Relation',
                'Object_keys', 'Subject_flag', 'Object_flag']]
        #G=nx.from_pandas_edgelist(x_df, "Subject", "Object", edge_attr="Relation", create_using=nx.MultiDiGraph(), edge_key="Relation")

        nt = Network("1000px", "1000px", directed=True)

        for i in range(len(x_df)):
            if x_df['Subject_keys'][i] not in nt.get_nodes():
                nt.add_node(x_df['Subject_keys'][i],
                            label=x_df['Subject_keys'][i], size=5)
            if x_df['Object_keys'][i] not in nt.get_nodes():
                nt.add_node(x_df['Object_keys'][i],
                            label=x_df['Object_keys'][i], size=5)
            nt.add_edge(x_df['Subject_keys'][i], x_df['Object_keys']
                        [i], title=x_df['Relation'][i])

        nodes = nt.get_nodes()
        edges = nt.get_edges()

        weighted_nt = Network("1000px", "1000px", directed=True)
        for i in range(len(nodes)):
            size = 5
            for j in range(len(x_df)):
                if x_df['Subject_keys'][j] == nodes[i] and x_df['Subject_flag'][j] == 1:
                    size = size + 1
                if x_df['Object_keys'][j] == nodes[i] and x_df['Object_flag'][j] == 1:
                    size = size + 1

            weighted_nt.add_node(nodes[i], label=nodes[i],
                                size=size, mass=size, title="123")

        for edge in edges:
            weighted_nt.add_edge(edge['from'], edge['to'], title=edge['title'])

        # nt.from_nx(G)
        weighted_nt.save_graph(save_path)


    def link_html_files(self, hl, dl, fn):  # hl for hyperlinks, dl for drive links
        '''
        Argument : path of folder which contains html files.

        Does: adds prev and next html files as hyperlinks

        Returns: None

        '''

        n = len(dl)  # hl and dl length will be equal
        for i in range(n):
            with open(dl[i]) as f_in:
                text = f_in.read()
                soup = BS(text, 'html.parser')

            if(i != 0):
                prev = soup.new_tag('a', href=fn[i-1])
                prev.string = "Previous KG"
                soup.body.append(prev)
                # no need to add line breaks if no prev button. just add next button alone
                br1 = soup.new_tag('br')
                soup.body.append(br1)
                br2 = soup.new_tag('br')
                soup.body.append(br2)

            if(i != n-1):
                next = soup.new_tag('a', href=fn[i+1])
                next.string = "Next KG"
                soup.body.append(next)

            with open(dl[i], "w") as out_f:
                out_f.write(str(soup))

        print("donee")
