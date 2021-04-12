import csv
import random
from feature_batch import Random_embedding,Glove_embedding
import torch
from comparison_plot_batch import NN_embedding_plot

# Data Reading
with open('train.tsv') as f:
    tsvreader = csv.reader (f, delimiter ='\t')
    temp = list ( tsvreader )

with open('glove.6B.50d.txt','rb') as f:  # for glove embedding
    lines=f.readlines()

# Construct dictionary with glove
trained_dict=dict()
n=len(lines)
for i in range(n):
    line=lines[i].split()
    trained_dict[line[0].decode("utf-8").upper()]=[float(line[j]) for j in range(1,51)]

# Initialization
iter_times=50
alpha=0.001

# Start
data = temp[1:]
batch_size=500

# random embedding
random.seed(2021)
random_embedding=Random_embedding(data=data)
random_embedding.get_words()
random_embedding.get_id()

# trained embedding : glove
random.seed(2021)
glove_embedding=Glove_embedding(data=data,trained_dict=trained_dict)
glove_embedding.get_words()
glove_embedding.get_id()

NN_embedding_plot(random_embedding,glove_embedding,alpha,batch_size,iter_times)