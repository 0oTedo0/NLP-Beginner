from feature_batch import Random_embedding, Glove_embedding, get_batch
import random
from comparison_plot_batch import NN_plot, NN_embdding
from Neural_Network_batch import ESIM

with open('snli_1.0_train.txt', 'r') as f:
    temp = f.readlines()

with open('glove.6B.50d.txt', 'rb') as f:  # for glove embedding
    lines = f.readlines()

# Construct dictionary with glove

trained_dict = dict()
n = len(lines)
for i in range(n):
    line = lines[i].split()
    trained_dict[line[0].decode("utf-8").upper()] = [float(line[j]) for j in range(1, 51)]

data = temp[1:]
# max_item = 100000
# data = data[:max_item]
learning_rate = 0.001
len_feature = 50
len_hidden = 50
iter_times = 50
batch_size = 1000

# random embedding
random.seed(2021)
random_embedding = Random_embedding(data=data)
random_embedding.get_words()
random_embedding.get_id()

# trained embedding : glove
random.seed(2021)
glove_embedding = Glove_embedding(data=data, trained_dict=trained_dict)
glove_embedding.get_words()
glove_embedding.get_id()

NN_plot(random_embedding, glove_embedding, len_feature, len_hidden, learning_rate, batch_size, iter_times)
