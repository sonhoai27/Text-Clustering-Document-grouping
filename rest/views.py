from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework.renderers import JSONRenderer, BrowsableAPIRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
import nltk
import pandas as pd
# import numpy as np
from nltk.corpus import wordnet as wn

import click
import re
import numpy
import random
import scipy.cluster.hierarchy as hcluster

class Clustering(APIView):
    renderer_classes = (JSONRenderer,)
    def post(self, request):
        print(request.data['list'])
        def convert_tag(tag):
            tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
            try:
                return tag_dict[tag[0]]
            except KeyError:
                return None

        def doc_to_synsets(doc):
            synsetlist = []
            tokens = nltk.word_tokenize(doc)
            pos = nltk.pos_tag(tokens)
            for tup in pos:
                try:
                    synsetlist.append(wn.synsets(
                        tup[0], convert_tag(tup[1]))[0])
                except:
                    continue
            print(synsetlist)
            return synsetlist

        def similarity_score(s1, s2):
            highscores = []
            for synset1 in s1:
                highest_yet = 0
                for synset2 in s2:
                    try:
                        simscore = synset1.path_similarity(synset2)
                        if simscore > highest_yet:
                            highest_yet = simscore
                    except:
                        continue

                if highest_yet > 0:
                    highscores.append(highest_yet)
            return sum(highscores)/len(highscores) if len(highscores) > 0 else 0

        def document_path_similarity(doc1, doc2):
            synsets1 = doc_to_synsets(doc1['name'])
            synsets2 = doc_to_synsets(doc2['name'])
            return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2

        def similarity(x, df):
            sim_score = []
            for i in df['Questions']:
                sim_score.append(document_path_similarity(x, i))
            return sim_score
        # df = pd.DataFrame({'Questions': ['What are you doing?','What are you doing tonight?','What are you doing now?','What is your name?','What is your nick name?','What is your full name?','Shall we meet?',
        #                    'How are you doing?' ]})
        df = pd.DataFrame({'Questions': request.data['list']})
        df['similarity'] = df['Questions'].apply(
            lambda x: similarity(x, df)).astype(str)
        result = []
        for _,i in df.groupby('similarity')['Questions']:
            result.append(i)
        return Response(result)

    def get(self, request):
        temp_data = [31,68,74,46,47,83,29,11,9,52,1272,1288,1297,1285,1294,1251,1265,1257,1280,1265,1292,1297,1271,1273,1253,1273,1291,1251,1295,1298,1264,1281,1294,1280,1250,1279,1298,1290,1294,1299,1266,1260,1298,1292,1280,1259,1266,1276,1253,1252,1271,1280,1284,1266,1254,1259,1291,1268,1253,1298,1288,1271,1298,1300,1274,1294,1263,1298,1270,1254,1266,1269,1283,1285,1286,1276,1257,1266,1272,1298,1261,1251,1272,1260,1291,1269,1260,1294,1287,1256,1253,1284,1269,1287,1292,1269,1272,1275,1250,1289,56,35,19,80,47,22,92,8,10,24,87,76,60,63,64,0,1295,1268,1280,1281,1277,1300,1278,1273,1250,1296,1266,1269,1282,1281,1272,1260,1292,1272,1253,1255,1299,1269,1268,1294,1250,1299,1292,1254,1281,1289,1259,1290,1271,1280,1272,1300,1258,1290,1289,1300,1299,1261,1300,1276,1290,1299,1280,1267,1283,1282,1269,1260,1285,1252,1250,1263,1297,1300,1292,1266,1260,1263,1292,1296,1289,1297,1251,1261,1250,1294,1278,1284,1291,1281,1269,1261,1257,1267,1265,1288,1291,1257,1296,1251,1260,1272,1294,1285,1269,1283,1297,1287,1253,1292,1299,1295,1286,1288,1283,1290,20,73,81,6,49,88,96,61,49,94,57,16,61,16,17,19,1280,1257,1259,1277,1257,1262,1263,1280,1292,1250,1287,1272,1258,1253,1285,1285,1257,1291,1273,1260,1267,1250,1280,1281,1263,1269,1292,1250,1282,1263,1274,1288,1296,1266,1291,1271,1273,1281,1261,1289,1269,1287,1296,1283,1280,1298,1259,1270,1259,1289,1269,1284,1295,1297,1256,1300,1281,1296,1284,1288,1285,1296,1277,1251,1279,1295,1281,1264,1280,1263,69,8,30,45,89,61,80,45,9,18,19,11,1255,1299,1296,1293,1287,1250,1265,1291,1281,1250,1286,1286,1251,1287,1266,1288,1254,1260,1260,1254,1267,1299,1273,1250,1300,1250,1279,1255,1293,1292,1278,1277,1252,1299,1278,1258,1268,1274,1285,1258,1279,1270,1278,1286,1278,1253,1267,1300,1295,1298,1285,1288,1274,1272,1252,1256,1283,1289,1251,1258,1253,1257,1297,1269,1292,1253,1273,1281,1251,1280,1253,1274,1275,1287,1296,1298,1296,1291,1284,1261,1267,1290,1273,1281,1263,1270,1264,1269,1278,1284,67,8,40,59,97,64,45,72,45,90,94,7,33,58,97,97,1252,1297,1265,1278,1272,1252,1258,1261,1287,1260,1260,1258,1280,1263,1256,1296,1269,1270,1296,1282,696,678,665,700,700,691,689,688,650,663,662,698,655,660,662,684,690,657,653,663,670,691,687,675,694,670,676,659,661,664,664,689,683,675,687,691,676,659,689,657,659,656,654,679,669,687,666,662,691,1260,1276,1252,1295,1257,1277,1281,1257,1295,1269,1265,1290,1266,1269,1286,1254,1260,1265,1290,1294,1286,1279,1254,1256,1276,1285,1282,1251,1282,1261,1253,56,74,85,94,18,83,38,80,8,4,78,43,7,79,68,78,1275,1250,1268,1297,1284,1255,1294,1262,1250,1252,680,693,677,676,670,653,670,661,658,695,665,671,656,686,662,691,675,658,671,650,667,653,652,686,667,682,694,654,689,682,667,658,651,652,692,652,655,651,650,698,655,650,679,672,697,696,696,683,1277,1264,1274,1260,1285,1285,1283,1259,1260,1288,1281,1284,1281,1257,1285,1295,1273,1264,1283,1284,1300,1299,1257,1297,1254,1257,1270,1257,1295,34,5,73,42,27,36,91,85,19,50,34,21,73,38,18,73]

        ndata = [[td, td] for td in temp_data]
        data = numpy.array(ndata)

        # clustering
        thresh = (11.0/100.0) * (max(temp_data) - min(temp_data))  #Threshold 11% of the total range of data

        clusters = hcluster.fclusterdata(data, thresh, criterion="distance")

        total_clusters = max(clusters)

        clustered_index = []
        for i in range(total_clusters):
            clustered_index.append([])

        for i in range(len(clusters)):
            clustered_index[clusters[i] - 1].append(i)

        clustered_range = []
        for x in clustered_index:
            clustered_index_x = [temp_data[y] for y in x]
            clustered_range.append((min(clustered_index_x) , max(clustered_index_x)))
        return Response(clustered_range)

class price(APIView):
    renderer_classes = (JSONRenderer,)
    def post(self, request):
        temp_data = request.data['list']
        print(temp_data)
        ndata = [[td, td] for td in temp_data]
        data = numpy.array(ndata)

        # clustering
        thresh = (11.0/100.0) * (max(temp_data) - min(temp_data))  #Threshold 11% of the total range of data

        clusters = hcluster.fclusterdata(data, thresh, criterion="distance")

        total_clusters = max(clusters)

        clustered_index = []
        for i in range(total_clusters):
            clustered_index.append([])

        for i in range(len(clusters)):
            clustered_index[clusters[i] - 1].append(i)

        clustered_range = []
        for x in clustered_index:
            clustered_index_x = [temp_data[y] for y in x]
            clustered_range.append((min(clustered_index_x) , max(clustered_index_x)))
        
        return Response(clustered_range)