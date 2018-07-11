from django.shortcuts import render

from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn

class Clustering(APIView):
    def get(self, request):
        def convert_tag(tag):   
            tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
            try:
                return tag_dict[tag[0]]
            except KeyError:
                return None

        def doc_to_synsets(doc):
            synsetlist =[]
            tokens=nltk.word_tokenize(doc)
            pos=nltk.pos_tag(tokens)
            for tup in pos:
                try:
                    synsetlist.append(wn.synsets(tup[0], convert_tag(tup[1]))[0])
                except:
                    continue        
            return synsetlist

        def similarity_score(s1, s2):
            highscores = []
            for synset1 in s1:
                highest_yet=0
                for synset2 in s2:
                    try:
                        simscore=synset1.path_similarity(synset2)
                        if simscore>highest_yet:highest_yet=simscore
                    except:
                        continue

                if highest_yet>0:
                    highscores.append(highest_yet)  
            return sum(highscores)/len(highscores)  if len(highscores) > 0 else 0

        def document_path_similarity(doc1, doc2):
            synsets1 = doc_to_synsets(doc1)
            synsets2 = doc_to_synsets(doc2)
            return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


        def similarity(x,df):
            sim_score = []
            for i in df['Questions']:
                sim_score.append(document_path_similarity(x,i))
            return sim_score
        #df = pd.DataFrame({'Questions': ['What are you doing?','What are you doing tonight?','What are you doing now?','What is your name?','What is your nick name?','What is your full name?','Shall we meet?',
        #                    'How are you doing?' ]})
        df = pd.DataFrame({'Questions': [
                        "1.Xiaomi Redmi Note 5", 
                        "1.Xiaomi Redmi Note 5", 
                        "1.Xiaomi Redmi Note 5",
                        "2.Xiaomi Redmi Note 5", 
                        "3.Xiaomi Redmi Note 5", 
                        "Xiaomi Redmi Note 5 3GB/32GB", 
                        "Xiaomi Redmi Note 5 3GB/32GB", 
                        "Xiaomi Redmi Note 5 Pro", 
                        "Xiaomi Redmi Note 5 Pro", 
                        "Xiaomi Redmi Note 5 Pro 3GB/32GB", 
                        "Xiaomi Redmi Note 5 Pro 3GB/32GB", 
                        "Xiaomi Redmi Note 5 Pro 3GB/32GB", 
                        "Xiaomi Redmi Note 5 Pro 3GB/32GB", 
                        "Xiaomi Redmi Note 5 Pro 3GB/32GB", 
                        "Xiaomi Redmi Note 5 Pro 3GB/32GB", 
                        "Xiaomi Redmi Note 5 Pro 4GB/64GB", 
                        "Xiaomi Redmi Note 5 Pro 4GB/64GB", 
                        "Xiaomi Redmi Note 5 Pro 4GB/64GB", 
                        "Xiaomi Redmi Note 5 Pro 6GB/64GB", 
                        "Xiaomi Redmi Note 5 Pro 6GB/64GB", 
                        "Xiaomi Redmi Note 5 Pro 6GB/64GB", 
                        "Xiaomi Redmi Note 5 Pro 6GB/64GB", 
                        "Xiaomi Redmi Note 5A", 
                        "Xiaomi Redmi Note 5A Prime", 
                        "Xiaomi Redmi Note 5A Prime", 
                        "Xiaomi Redmi Note 5A Prime 3GB/32GB"
                    ]})
        df['similarity'] = df['Questions'].apply(lambda x : similarity(x,df)).astype(str)
        result = []
        for _, i in df.groupby('similarity')['Questions']:
            result.append(i)
        return Response(result)
