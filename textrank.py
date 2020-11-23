import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx

def  textranksumm(text):
  sentences=sent_tokenize(text)
  sentences_clean=[re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]
  stop_words = stopwords.words('english')
  sentence_tokens=[[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]
  w2v=Word2Vec(sentence_tokens,size=1,min_count=1,iter=1000)
  sentence_embeddings=[[w2v[word][0] for word in words] for words in sentence_tokens]
  max_len=max([len(tokens) for tokens in sentence_tokens])
  sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]

  similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
  for i,row_embedding in enumerate(sentence_embeddings):
    for j,column_embedding in enumerate(sentence_embeddings):
        similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)
  nx_graph = nx.from_numpy_array(similarity_matrix)
  scores = nx.pagerank(nx_graph)
  top_sentence={sentence:scores[index] for index,sentence in enumerate(sentences)}
  top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:4])
  x=""
  for sent in sentences:
    if sent in top.keys():
        x+=sent
  return x