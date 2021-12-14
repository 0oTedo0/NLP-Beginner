from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence


class Word_Embedding():
    def __init__(self, data):
        self.data = data
        self.word_dict = {'<pad>': 0, '<begin>': 1, '<end>': 2}
        self.tag_dict={0:'<pad>',1:'<begin>',2:'<end>'}
        self.matrix = list()

    def form_poem(self):
        data_utf8 = list(map(lambda x, y: str(x, encoding=y), self.data, ['utf-8'] * len(self.data)))
        poems = list()
        new_poem = ""
        for item in data_utf8:
            if item == '\n':
                if new_poem:
                    poems.append(new_poem)
                new_poem = ""
            else:
                if item[-2]==' ':
                  position=-2
                else:
                  position=-1
                new_poem = ''.join([new_poem, item[:position]])
        self.data = poems

    def get_words(self):
        for poem in self.data:
            for word in poem:
                if word not in self.word_dict:
                    self.tag_dict[len(self.word_dict)]=word
                    self.word_dict[word] = len(self.word_dict)

    def get_id(self):
        for poem in self.data:
            self.matrix.append([self.word_dict[word] for word in poem])

    def data_process(self):
        self.form_poem()
        self.data.sort(key=lambda x: len(x))
        self.get_words()
        self.get_id()


class ClsDataset(Dataset):
    def __init__(self, poem):
        self.poem = poem

    def __getitem__(self, item):
        return self.poem[item]

    def __len__(self):
        return len(self.poem)


def collate_fn(batch_data):
    poems = batch_data
    poems = [torch.LongTensor([1, *poem]) for poem in poems]

    padded_poems = pad_sequence(poems, batch_first=True, padding_value=0)
    padded_poems = [torch.cat([poem, torch.LongTensor([2])]) for poem in padded_poems]
    padded_poems = list(map(list,padded_poems))
    return torch.LongTensor(padded_poems)


def get_batch(x, batch_size):
    dataset = ClsDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return dataloader
