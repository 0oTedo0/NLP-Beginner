import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch


def data_split(data, test_rate=0.3):
    """ Take some data , and split them into training set and test set."""
    train = list()
    test = list()
    for datum in data:
        if random.random() > test_rate:
            train.append(datum)
        else:
            test.append(datum)
    return train, test


class Random_embedding():
    def __init__(self, data, test_rate=0.3):
        self.dict_words = dict()
        data.sort(key=lambda x:len(x[2].split()))
        self.data = data
        self.len_words = 0
        self.train, self.test = data_split(data, test_rate=test_rate)
        self.train_y = [int(term[3]) for term in self.train]  # Categories in training set
        self.test_y = [int(term[3]) for term in self.test]  # Categories in test set
        self.train_matrix = list()
        self.test_matrix = list()
        self.longest=0

    def get_words(self):
        for term in self.data:
            s = term[2]
            s = s.upper()
            words = s.split()
            for word in words:  # Process every word
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)+1
        self.len_words=len(self.dict_words)

    def get_id(self):
        for term in self.train:
            s = term[2]
            s = s.upper()
            words = s.split()
            item=[self.dict_words[word] for word in words]
            self.longest=max(self.longest,len(item))
            self.train_matrix.append(item)
        for term in self.test:
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_matrix.append(item)
        self.len_words += 1


class Glove_embedding():
    def __init__(self, data,trained_dict,test_rate=0.3):
        self.dict_words = dict()
        self.trained_dict=trained_dict
        data.sort(key=lambda x:len(x[2].split()))
        self.data = data
        self.len_words = 0
        self.train, self.test = data_split(data, test_rate=test_rate)
        self.train_y = [int(term[3]) for term in self.train]  # Categories in training set
        self.test_y = [int(term[3]) for term in self.test]  # Categories in test set
        self.train_matrix = list()
        self.test_matrix = list()
        self.longest=0
        self.embedding=list()

    def get_words(self):
        self.embedding.append([0] * 50)
        for term in self.data:
            s = term[2]
            s = s.upper()
            words = s.split()
            for word in words:  # Process every word
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)+1
                    if word in self.trained_dict:
                        self.embedding.append(self.trained_dict[word])
                    else:
                        # print(word)
                        # raise Exception("words not found!")
                        self.embedding.append([0]*50)
        self.len_words=len(self.dict_words)

    def get_id(self):
        for term in self.train:
            s = term[2]
            s = s.upper()
            words = s.split()
            item=[self.dict_words[word] for word in words]
            self.longest=max(self.longest,len(item))
            self.train_matrix.append(item)
        for term in self.test:
            s = term[2]
            s = s.upper()
            words = s.split()
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_matrix.append(item)
        self.len_words += 1


class ClsDataset(Dataset):
    def __init__(self, sentence, emotion):
        self.sentence = sentence
        self.emotion= emotion

    def __getitem__(self, item):
        return self.sentence[item], self.emotion[item]

    def __len__(self):
        return len(self.emotion)


def collate_fn(batch_data):
    sentence, emotion = zip(*batch_data)
    sentences = [torch.LongTensor(sent) for sent in sentence]
    padded_sents = pad_sequence(sentences, batch_first=True, padding_value=0)
    return torch.LongTensor(padded_sents), torch.LongTensor(emotion)


def get_batch(x,y,batch_size):
    dataset = ClsDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True,collate_fn=collate_fn)
    return dataloader
