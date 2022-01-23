from random import shuffle

import torch
import time
import tqdm
from transformers import BertTokenizer


class DatasetIterator(object):
    def __init__(self, data, batch_size, device):
        self.batch_size = batch_size
        self.batches = data
        self.num_batches = len(data) // batch_size
        self.residue = False
        # 记录batch数量是否为整数
        if len(data) % self.num_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, data):
        x = torch.LongTensor([rec[0] for rec in data]).to(self.device)
        y = torch.LongTensor([rec[1] for rec in data]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([rec[2] for rec in data]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.num_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.num_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.num_batches + 1
        else:
            return self.num_batches


def build_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter



def load_data(filename, tokenizer:BertTokenizer, pad_size=32):
    data = []
    with open(filename, 'r', encoding='utf-8') as reader:
        for line in tqdm.tqdm(reader):
            if line.strip() == '': continue
            text, label = line.strip().split('\t')
            tokens = tokenizer.tokenize(text)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            seq_len = len(input_ids)
            if pad_size:
                if len(input_ids) < pad_size:
                    input_ids = input_ids + [0] * (pad_size - len(input_ids))
                else:
                    input_ids = input_ids[:pad_size]
                    seq_len = pad_size
            label = int(label)
            data.append((input_ids, label, seq_len))
    return data


def create_small_dataset():
    def sample_data(input_file, output_file, ratio=0.3):
        data = []
        with open(input_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                if line.strip() != '':
                    data.append(line.strip())

        shuffle(data)
        data_len = len(data)
        with open(output_file, 'w', encoding='utf-8') as writer:
            writer.write('\n'.join(data[:int(data_len * ratio)]) + '\n')
    sample_data("./train.txt", "./mini-train.txt")
    sample_data("./dev.txt", "./mini-dev.txt")
    sample_data("./test.txt", "./mini-test.txt")





if __name__ == '__main__':
    create_small_dataset()









