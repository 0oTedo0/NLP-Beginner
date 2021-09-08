import matplotlib.pyplot
import torch
import torch.nn.functional as F
from feature_batch import get_batch
from torch import optim
from Neural_Network_batch import ESIM
import random
import numpy


def NN_embdding(model, train, test, learning_rate, iter_times):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = F.cross_entropy
    train_loss_record = list()
    test_loss_record = list()
    train_record = list()
    test_record = list()
    # torch.autograd.set_detect_anomaly(True)

    for iteration in range(iter_times):
      torch.cuda.empty_cache()
      model.train()
      for i, batch in enumerate(train):
        torch.cuda.empty_cache()
        x1, x2, y = batch
        pred = model(x1, x2).cuda()
        optimizer.zero_grad()
        y=y.cuda()
        loss = loss_fun(pred, y).cuda()
        loss.backward()
        optimizer.step()
      with torch.no_grad():
        model.eval()
        train_acc = list()
        test_acc = list()
        train_loss = 0
        test_loss = 0
        for i, batch in enumerate(train):
          torch.cuda.empty_cache()
          x1, x2, y = batch
          y=y.cuda()
          pred = model(x1, x2).cuda()
          loss = loss_fun(pred, y).cuda()
          train_loss += loss.item()
          _, y_pre = torch.max(pred, -1)
          acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
          train_acc.append(acc)

        for i, batch in enumerate(test):
          torch.cuda.empty_cache()
          x1, x2, y = batch
          y=y.cuda()
          pred = model(x1, x2).cuda()
          loss = loss_fun(pred, y).cuda()
          test_loss += loss.item()
          _, y_pre = torch.max(pred, -1)
          acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
          test_acc.append(acc)

      trains_acc = sum(train_acc) / len(train_acc)
      tests_acc = sum(test_acc) / len(test_acc)

      train_loss_record.append(train_loss / len(train_acc))
      test_loss_record.append(test_loss/ len(test_acc))
      train_record.append(trains_acc.cpu())
      test_record.append(tests_acc.cpu())
      print("---------- Iteration", iteration + 1, "----------")
      print("Train loss:", train_loss/ len(train_acc))
      print("Test loss:", test_loss/ len(test_acc))
      print("Train accuracy:", trains_acc)
      print("Test accuracy:", tests_acc)

    return train_loss_record, test_loss_record, train_record, test_record


def NN_plot(random_embedding, glove_embedding, len_feature, len_hidden, learning_rate, batch_size, iter_times):
    train_random = get_batch(random_embedding.train_s1_matrix, random_embedding.train_s2_matrix,
                             random_embedding.train_y,batch_size)
    test_random = get_batch(random_embedding.test_s1_matrix, random_embedding.test_s2_matrix,
                            random_embedding.test_y,batch_size)
    train_glove = get_batch(glove_embedding.train_s1_matrix, glove_embedding.train_s2_matrix,
                            glove_embedding.train_y,batch_size)
    test_glove = get_batch(glove_embedding.test_s1_matrix, glove_embedding.test_s2_matrix,
                           glove_embedding.test_y,batch_size)
    random.seed(2021)
    numpy.random.seed(2021)
    torch.cuda.manual_seed(2021)
    torch.manual_seed(2021)
    random_model = ESIM(len_feature, len_hidden, random_embedding.len_words, longest=random_embedding.longest)
    random.seed(2021)
    numpy.random.seed(2021)
    torch.cuda.manual_seed(2021)
    torch.manual_seed(2021)
    glove_model = ESIM(len_feature, len_hidden, glove_embedding.len_words, longest=glove_embedding.longest,
                       weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))
    random.seed(2021)
    numpy.random.seed(2021)
    torch.cuda.manual_seed(2021)
    torch.manual_seed(2021)
    trl_ran, tsl_ran, tra_ran, tea_ran = NN_embdding(random_model, train_random, test_random, learning_rate,
                                                     iter_times)
    random.seed(2021)
    numpy.random.seed(2021)
    torch.cuda.manual_seed(2021)
    torch.manual_seed(2021)
    trl_glo, tsl_glo, tra_glo, tea_glo = NN_embdding(glove_model, train_glove, test_glove, learning_rate,
                                                     iter_times)
    x = list(range(1, iter_times + 1))
    matplotlib.pyplot.subplot(2, 2, 1)
    matplotlib.pyplot.plot(x, trl_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, trl_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 2)
    matplotlib.pyplot.plot(x, tsl_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, tsl_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 3)
    matplotlib.pyplot.plot(x, tra_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, tra_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.subplot(2, 2, 4)
    matplotlib.pyplot.plot(x, tea_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, tea_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.tight_layout()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 8, forward=True)
    matplotlib.pyplot.savefig('main_plot.jpg')
    matplotlib.pyplot.show()
