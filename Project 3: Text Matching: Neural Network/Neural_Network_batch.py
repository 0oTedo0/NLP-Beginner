import torch
import torch.nn as nn


class Input_Encoding(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words,longest, weight=None, layer=1, batch_first=True, drop_out=0.5):
        super(Input_Encoding, self).__init__()
        self.len_feature = len_feature
        self.len_hidden = len_hidden
        self.len_words = len_words
        self.layer = layer
        self.longest=longest
        self.dropout = nn.Dropout(drop_out)
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x)
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight)
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True)

    def forward(self, x):
        x = torch.LongTensor(x)
        x = self.embedding(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class Local_Inference_Modeling(nn.Module):
    def __init__(self):
        super(Local_Inference_Modeling, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

    def forward(self, a_bar, b_bar):
        e = torch.matmul(a_bar, b_bar.transpose(1, 2))

        a_tilde = self.softmax_2(e)
        a_tilde = a_tilde.bmm(b_bar)
        b_tilde = self.softmax_1(e)
        b_tilde = b_tilde.transpose(1, 2).bmm(a_bar)

        m_a = torch.cat([a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde], dim=-1)
        m_b = torch.cat([b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde], dim=-1)

        return m_a, m_b


class Inference_Composition(nn.Module):
    def __init__(self, len_feature, len_hidden_m, len_hidden, layer=1, batch_first=True, drop_out=0.5):
        super(Inference_Composition, self).__init__()
        self.linear = nn.Linear(len_hidden_m, len_feature)
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        return x


class Prediction(nn.Module):
    def __init__(self, len_v, len_mid, type_num=4, drop_out=0.5):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(nn.Dropout(drop_out), nn.Linear(len_v, len_mid), nn.Tanh(),
                                 nn.Linear(len_mid, type_num))

    def forward(self, a,b):

        v_a_avg=a.sum(1)/a.shape[1]
        v_a_max = a.max(1)[0]

        v_b_avg = b.sum(1) / b.shape[1]
        v_b_max = b.max(1)[0]

        out_put = torch.cat((v_a_avg, v_a_max,v_b_avg,v_b_max), dim=-1)

        return self.mlp(out_put)


class ESIM(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words,longest, type_num=4, weight=None, layer=1, batch_first=True,
                 drop_out=0.5):
        super(ESIM, self).__init__()
        self.len_words=len_words
        self.longest=longest
        self.input_encoding = Input_Encoding(len_feature, len_hidden, len_words,longest, weight=weight, layer=layer,
                                             batch_first=batch_first, drop_out=drop_out)
        self.local_inference_modeling = Local_Inference_Modeling()
        self.inference_composition = Inference_Composition(len_feature, 8 * len_hidden, len_hidden, layer=layer,
                                                           batch_first=batch_first, drop_out=drop_out)
        self.prediction=Prediction(len_hidden*8,len_hidden,type_num=type_num,drop_out=drop_out)

    def forward(self,a,b):
        a_bar=self.input_encoding(a)
        b_bar=self.input_encoding(b)

        m_a,m_b=self.local_inference_modeling(a_bar,b_bar)

        v_a=self.inference_composition(m_a)
        v_b=self.inference_composition(m_b)

        out_put=self.prediction(v_a,v_b)

        return out_put
