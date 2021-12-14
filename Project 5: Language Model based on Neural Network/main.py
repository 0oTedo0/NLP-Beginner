from feature import get_batch,Word_Embedding
from torch import optim
import random,numpy,torch
from Neural_Network import Language
import torch.nn.functional as F

random.seed(2021)
numpy.random.seed(2021)
torch.cuda.manual_seed(2021)
torch.manual_seed(2021)

with open('poetryFromTang.txt', 'rb') as f:
    temp = f.readlines()

a=Word_Embedding(temp)
a.data_process()
train=get_batch(a.matrix,1)
learning_rate=0.004
iter_times=10

strategies=['lstm','gru']
train_loss_records=list()
models=list()
for i in range(2):
  random.seed(2021)
  numpy.random.seed(2021)
  torch.cuda.manual_seed(2021)
  torch.manual_seed(2021)
  model=Language(50,len(a.word_dict),50,a.tag_dict,a.word_dict,strategy=strategies[i])
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  loss_fun = F.cross_entropy
  train_loss_record=list()

  model=model.cuda()
  for iteration in range(iter_times):
      total_loss = 0
      model.train()
      for i, batch in enumerate(train):
          x=batch.cuda()
          x,y=x[:,:-1],x[:,1:]
          pred = model(x).transpose(1,2)
          optimizer.zero_grad()
          loss = loss_fun(pred, y)
          total_loss+=loss.item()/(x.shape[1]-1)
          loss.backward()
          optimizer.step()
      train_loss_record.append(total_loss/len(train))
      print("---------- Iteration", iteration + 1, "----------")
      print("Train loss:", total_loss/len(train))
  train_loss_records.append(train_loss_record)
  models.append(model)

def cat_poem(l):
  """拼接诗句"""
  poem=list()
  for item in l:
    poem.append(''.join(item))
  return poem

model=models[0]
# 生成固定诗句
poem=cat_poem(model.generate_random_poem(16,4,random=False))
for sent in poem:
  print(sent)

# 生成随机诗句
torch.manual_seed(2021)
poem=cat_poem(model.generate_random_poem(12,6,random=True))
for sent in poem:
  print(sent)

# 生成固定藏头诗
poem=cat_poem(model.generate_hidden_head("春夏秋冬",max_len=20,random=False))
for sent in poem:
  print(sent)

# 生成随机藏头诗
torch.manual_seed(0)
poem=cat_poem(model.generate_hidden_head("春夏秋冬",max_len=20,random=True))
for sent in poem:
  print(sent)

# 画图
import matplotlib.pyplot
x = list(range(1, iter_times + 1))
matplotlib.pyplot.plot(x, train_loss_records[0], 'r--',label='Lstm')
matplotlib.pyplot.plot(x, train_loss_records[1], 'g--',label='Gru')
matplotlib.pyplot.legend()
matplotlib.pyplot.title("Average Train Loss")
matplotlib.pyplot.xlabel("Iterations")
matplotlib.pyplot.ylabel("Loss")
matplotlib.pyplot.savefig('main_plot.jpg')
matplotlib.pyplot.show()