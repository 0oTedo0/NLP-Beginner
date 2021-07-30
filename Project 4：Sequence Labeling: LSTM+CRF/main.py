from feature import pre_process, Glove_embedding, get_batch
from torch import optim
from Neural_Network import Named_Entity_Recognition
from comparison_plot import  NN_plot
import random,numpy,torch

random.seed(2021)
numpy.random.seed(2021)
torch.cuda.manual_seed(2021)
torch.manual_seed(2021)

with open('train.txt', 'r') as f:
    temp = f.readlines()

data = temp[2:]
train_zip = pre_process(data)

with open('test.txt', 'r') as f:
    temp = f.readlines()

data = temp[2:]
test_zip = pre_process(data)

with open('glove.6B.50d.txt', 'rb') as f:  # for glove embedding
    lines = f.readlines()

# Construct dictionary with glove

trained_dict = dict()
n = len(lines)
for i in range(n):
    line = lines[i].split()
    trained_dict[line[0].decode("utf-8").upper()] = [float(line[j]) for j in range(1, 51)]

random_embedding = Glove_embedding(train_zip, test_zip,trained_dict=None)
random_embedding.get_words()
random_embedding.get_id()

glove_embedding = Glove_embedding(train_zip, test_zip,trained_dict=trained_dict)
glove_embedding.get_words()
glove_embedding.get_id()


iter_times = 100
learning_rate=0.001
batch_size=100

NN_plot(random_embedding,glove_embedding,50,50,learning_rate,batch_size,iter_times)