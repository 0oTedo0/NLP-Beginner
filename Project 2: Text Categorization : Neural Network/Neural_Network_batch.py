import torch
import torch.nn as nn
import torch.nn.functional as F


class MY_RNN(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words, typenum=5, weight=None, layer=1, nonlinearity='tanh',
                 batch_first=True, drop_out=0.5):
        super(MY_RNN, self).__init__()
        self.len_feature = len_feature
        self.len_hidden = len_hidden
        self.len_words = len_words
        self.layer = layer
        self.dropout=nn.Dropout(drop_out)
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        self.rnn = nn.RNN(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, nonlinearity=nonlinearity,
                          batch_first=batch_first, dropout=drop_out).cuda()
        # Fully connected layer
        self.fc = nn.Linear(len_hidden, typenum).cuda()
        # An extra softmax layer may be redundant
        # self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.LongTensor(x).cuda()
        batch_size = x.size(0)
        out_put = self.embedding(x)
        out_put=self.dropout(out_put)

        # h0 = torch.randn(self.layer, batch_size, self.len_hidden).cuda()
        h0 = torch.autograd.Variable(torch.zeros(self.layer, batch_size, self.len_hidden)).cuda()
        _, hn = self.rnn(out_put, h0)
        out_put = self.fc(hn).squeeze(0)
        # out_put = self.act(out_put)
        return out_put


class MY_CNN(nn.Module):
    def __init__(self, len_feature, len_words, longest, len_kernel=50, typenum=5, weight=None,drop_out=0.5):
        super(MY_CNN, self).__init__()
        self.len_feature = len_feature
        self.len_words = len_words
        self.len_kernel = len_kernel
        self.longest = longest
        self.dropout = nn.Dropout(drop_out)
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        self.conv1 = nn.Sequential(nn.Conv2d(1, longest, (2, len_feature), padding=(1, 0)), nn.ReLU()).cuda()
        self.conv2 = nn.Sequential(nn.Conv2d(1, longest, (3, len_feature), padding=(1, 0)), nn.ReLU()).cuda()
        self.conv3 = nn.Sequential(nn.Conv2d(1, longest, (4, len_feature), padding=(2, 0)), nn.ReLU()).cuda()
        self.conv4 = nn.Sequential(nn.Conv2d(1, longest, (5, len_feature), padding=(2, 0)), nn.ReLU()).cuda()
        # Fully connected layer
        self.fc = nn.Linear(4 * longest, typenum).cuda()
        # An extra softmax layer may be redundant
        # self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.LongTensor(x).cuda()
        out_put = self.embedding(x).view(x.shape[0], 1, x.shape[1], self.len_feature)
        out_put=self.dropout(out_put)

        conv1 = self.conv1(out_put).squeeze(3)
        pool1 = F.max_pool1d(conv1, conv1.shape[2])

        conv2 = self.conv2(out_put).squeeze(3)
        pool2 = F.max_pool1d(conv2, conv2.shape[2])

        conv3 = self.conv3(out_put).squeeze(3)
        pool3 = F.max_pool1d(conv3, conv3.shape[2])

        conv4 = self.conv4(out_put).squeeze(3)
        pool4 = F.max_pool1d(conv4, conv4.shape[2])

        pool = torch.cat([pool1, pool2, pool3, pool4], 1).squeeze(2)
        out_put = self.fc(pool)
        # out_put = self.act(out_put)
        return out_put
