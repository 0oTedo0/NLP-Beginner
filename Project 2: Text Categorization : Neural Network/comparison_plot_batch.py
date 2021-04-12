import matplotlib.pyplot
import torch
import torch.nn.functional as F
from torch import optim
from Neural_Network_batch import MY_RNN,MY_CNN
from feature_batch import get_batch


def NN_embdding(model, train,test, learning_rate, iter_times):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = F.cross_entropy
    train_loss_record=list()
    test_loss_record=list()
    long_loss_record=list()
    train_record=list()
    test_record=list()
    long_record=list()
    # torch.autograd.set_detect_anomaly(True)

    for iteration in range(iter_times):
        model.train()
        for i, batch in enumerate(train):
            x, y = batch
            y=y.cuda()
            pred = model(x).cuda()
            optimizer.zero_grad()
            loss = loss_fun(pred, y).cuda()
            loss.backward()
            optimizer.step()

        model.eval()
        train_acc = list()
        test_acc = list()
        long_acc = list()
        length = 20
        train_loss = 0
        test_loss = 0
        long_loss=0
        for i, batch in enumerate(train):
            x, y = batch
            y=y.cuda()
            pred = model(x).cuda()
            loss = loss_fun(pred, y).cuda()
            train_loss += loss.item()
            _, y_pre = torch.max(pred, -1)
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
            train_acc.append(acc)

        for i, batch in enumerate(test):
            x, y = batch
            y=y.cuda()
            pred = model(x).cuda()
            loss = loss_fun(pred, y).cuda()
            test_loss += loss.item()
            _, y_pre = torch.max(pred, -1)
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
            test_acc.append(acc)
            if(len(x[0]))>length:
              long_acc.append(acc)
              long_loss+=loss.item()

        trains_acc = sum(train_acc) / len(train_acc)
        tests_acc = sum(test_acc) / len(test_acc)
        longs_acc = sum(long_acc) / len(long_acc)

        train_loss_record.append(train_loss / len(train_acc))
        test_loss_record.append(test_loss / len(test_acc))
        long_loss_record.append(long_loss/len(long_acc))
        train_record.append(trains_acc.cpu())
        test_record.append(tests_acc.cpu())
        long_record.append(longs_acc.cpu())
        print("---------- Iteration", iteration + 1, "----------")
        print("Train loss:", train_loss/ len(train_acc))
        print("Test loss:", test_loss/ len(test_acc))
        print("Train accuracy:", trains_acc)
        print("Test accuracy:", tests_acc)
        print("Long sentence accuracy:", longs_acc)

    return train_loss_record,test_loss_record,long_loss_record,train_record,test_record,long_record


def NN_embedding_plot(random_embedding,glove_embedding,learning_rate, batch_size, iter_times):
    train_random = get_batch(random_embedding.train_matrix,
                             random_embedding.train_y, batch_size)
    test_random = get_batch(random_embedding.test_matrix,
                            random_embedding.test_y, batch_size)
    train_glove = get_batch(glove_embedding.train_matrix,
                            glove_embedding.train_y, batch_size)
    test_glove = get_batch(random_embedding.test_matrix,
                           glove_embedding.test_y, batch_size)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    random_rnn = MY_RNN(50, 50, random_embedding.len_words)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    random_cnn = MY_CNN(50, random_embedding.len_words, random_embedding.longest)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    glove_rnn = MY_RNN(50, 50, glove_embedding.len_words, weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    glove_cnn = MY_CNN(50, glove_embedding.len_words, glove_embedding.longest,weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    trl_ran_rnn,tel_ran_rnn,lol_ran_rnn,tra_ran_rnn,tes_ran_rnn,lon_ran_rnn=\
        NN_embdding(random_rnn,train_random,test_random,learning_rate,  iter_times)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    trl_ran_cnn,tel_ran_cnn,lol_ran_cnn, tra_ran_cnn, tes_ran_cnn, lon_ran_cnn = \
        NN_embdding(random_cnn, train_random,test_random, learning_rate, iter_times)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    trl_glo_rnn,tel_glo_rnn,lol_glo_rnn, tra_glo_rnn, tes_glo_rnn, lon_glo_rnn = \
        NN_embdding(glove_rnn, train_glove,test_glove, learning_rate, iter_times)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    trl_glo_cnn,tel_glo_cnn,lol_glo_cnn, tra_glo_cnn, tes_glo_cnn, lon_glo_cnn= \
        NN_embdding(glove_cnn,train_glove,test_glove, learning_rate, iter_times)
    x=list(range(1,iter_times+1))
    matplotlib.pyplot.subplot(2, 2, 1)
    matplotlib.pyplot.plot(x, trl_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, trl_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, trl_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, trl_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 2)
    matplotlib.pyplot.plot(x, tel_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, tel_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, tel_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, tel_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 3)
    matplotlib.pyplot.plot(x, tra_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, tra_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, tra_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, tra_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.subplot(2, 2, 4)
    matplotlib.pyplot.plot(x, tes_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, tes_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, tes_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, tes_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
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
    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.plot(x, lon_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, lon_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, lon_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, lon_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Long Sentence Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.plot(x, lol_ran_rnn, 'r--', label='RNN+random')
    matplotlib.pyplot.plot(x, lol_ran_cnn, 'g--', label='CNN+random')
    matplotlib.pyplot.plot(x, lol_glo_rnn, 'b--', label='RNN+glove')
    matplotlib.pyplot.plot(x, lol_glo_cnn, 'y--', label='CNN+glove')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Long Sentence Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.tight_layout()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 8, forward=True)
    matplotlib.pyplot.savefig('sub_plot.jpg')
    matplotlib.pyplot.show()