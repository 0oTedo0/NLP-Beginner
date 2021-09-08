import random
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def data_split(data, test_rate=0.3):
    """ Take some data , and split them into training set and test set."""
    train = list()
    test = list()
    i = 0
    for datum in data:
      i += 1
      if random.random() > test_rate:
          train.append(datum)
      else:
          test.append(datum)
    return train, test


class Random_embedding():
    def __init__(self, data, test_rate=0.3):
        self.dict_words = dict()
        _data = [item.split('\t') for item in data]
        self.data = [[item[5], item[6], item[0]] for item in _data]
        self.data.sort(key=lambda x:len(x[0].split()))
        self.len_words = 0
        self.train, self.test = data_split(self.data, test_rate=test_rate)
        self.type_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.train_y = [self.type_dict[term[2]] for term in self.train]  # Relation in training set
        self.test_y = [self.type_dict[term[2]] for term in self.test]  # Relation in test set
        self.train_s1_matrix = list()
        self.test_s1_matrix = list()
        self.train_s2_matrix = list()
        self.test_s2_matrix = list()
        self.longest = 0

    def get_words(self):
        pattern = '[A-Za-z|\']+'
        for term in self.data:
            for i in range(2):
                s = term[i]
                s = s.upper()
                words = re.findall(pattern, s)
                for word in words:  # Process every word
                    if word not in self.dict_words:
                        self.dict_words[word] = len(self.dict_words)+1
        self.len_words = len(self.dict_words)

    def get_id(self):
        pattern = '[A-Za-z|\']+'
        for term in self.train:
            s = term[0]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_s1_matrix.append(item)
            s = term[1]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_s2_matrix.append(item)
        for term in self.test:
            s = term[0]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_s1_matrix.append(item)
            s = term[1]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_s2_matrix.append(item)
        self.len_words+=1


class Glove_embedding():
    def __init__(self, data, trained_dict, test_rate=0.3):
        self.dict_words = dict()
        _data = [item.split('\t') for item in data]
        self.data = [[item[5], item[6], item[0]] for item in _data]
        self.data.sort(key=lambda x:len(x[0].split()))
        self.trained_dict = trained_dict
        self.len_words = 0
        self.train, self.test = data_split(self.data, test_rate=test_rate)
        self.type_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.train_y = [self.type_dict[term[2]] for term in self.train]  # Relation in training set
        self.test_y = [self.type_dict[term[2]] for term in self.test]  # Relation in test set
        self.train_s1_matrix = list()
        self.test_s1_matrix = list()
        self.train_s2_matrix = list()
        self.test_s2_matrix = list()
        self.longest = 0
        self.embedding = list()

    def get_words(self):
        self.embedding.append([0]*50)
        pattern = '[A-Za-z|\']+'
        for term in self.data:
            for i in range(2):
                s = term[i]
                s = s.upper()
                words = re.findall(pattern, s)
                for word in words:  # Process every word
                    if word not in self.dict_words:
                        self.dict_words[word] = len(self.dict_words)
                        if word in self.trained_dict:
                            self.embedding.append(self.trained_dict[word])
                        else:
                            # print(word)
                            # raise Exception("words not found!")
                            self.embedding.append([0] * 50)
        self.len_words = len(self.dict_words)

    def get_id(self):
        pattern = '[A-Za-z|\']+'
        for term in self.train:
            s = term[0]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_s1_matrix.append(item)
            s = term[1]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_s2_matrix.append(item)
        for term in self.test:
            s = term[0]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_s1_matrix.append(item)
            s = term[1]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_s2_matrix.append(item)
        self.len_words+=1


class ClsDataset(Dataset):
    """ 文本分类数据集 """
    def __init__(self, sentence1,sentence2, relation):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.relation = relation

    def __getitem__(self, item):
        return self.sentence1[item], self.sentence2[item],self.relation[item]

    def __len__(self):
        return len(self.relation)


def collate_fn(batch_data):
    """ 自定义一个batch里面的数据的组织方式 """

    sents1,sents2, labels = zip(*batch_data)
    sentences1 = [torch.LongTensor(sent) for sent in sents1]
    padded_sents1 = pad_sequence(sentences1, batch_first=True, padding_value=0)
    sentences2 = [torch.LongTensor(sent) for sent in sents2]
    padded_sents2 = pad_sequence(sentences2, batch_first=True, padding_value=0)
    return torch.LongTensor(padded_sents1), torch.LongTensor(padded_sents2),  torch.LongTensor(labels)


def get_batch(x1,x2,y,batch_size):
    dataset = ClsDataset(x1,x2, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True,collate_fn=collate_fn)
    return dataloader
