# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import json

class Config(object):

    """配置参数"""
    def __init__(self,  train_file, dev_file, test_file, vocab_file,
                 embedding_file=None, model_dir="TextCNN", model_name="TextCNN", mode=None):
        self.model_name = model_name
        # 训练集
        self.train_file = train_file
        # 验证集
        self.dev_file   = dev_file
        # 测试集
        self.test_file  = test_file
        # 类别名单
        self.class_list = ['finance', 'realty', 'stocks', 'education',
                           'science', 'society', 'politics', 'sports',
                           'game', 'entertainment']
        # 词表
        self.vocab_file = vocab_file
        # 模型训练结果
        self.model_dir  = model_dir
        # 预训练词向量
        self.embedding_file = embedding_file
        # 设备
        # 随机失活
        self.dropout = 0.9
        self.require_improvement = 600                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 10                        # 类别数
        self.vocab_size = 21128                                             # 词表大小，在运行时赋值
        self.num_epochs = 30                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embedding_dim = 128                                        # 字向量维度
        self.filter_w = 2
        self.filter_h = 3
        self.filter_d = 4
        self.num_filters = 256                                          # 卷积核数量(channels数)


        self.log_dir = "./LOG"
        self.mode = mode

        if self.mode is None: path = "PLAIN"
        else: path = '-' + self.mode
        self.output_fir = "./TCNN" + path
        self.model_dir = os.path.join(self.output_fir, self.model_dir)
        self.optim_dir = os.path.join(self.output_fir, "optim")
        self.params = os.path.join(self.output_fir, "config.json")




    # @staticmethod
    # def load_json(filename):
    #     with open(filename, 'r', encoding='utf-8') as reader:
    #         params = json.loads(reader)
    #     conf = Config()
    #

'''Convolutional Neural Networks for Sentence Classification'''

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if config.embedding_file is not None:
            embeddings = torch.tensor(pickle.load(open(config.embedding_file, 'rb')), dtype=torch.float) \
                .to(self.device)
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size,
                                          config.embedding_dim,
                                          padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_dim)) for k in [config.filter_w, config.filter_h, config.filter_d]])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * 3, config.num_classes)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, noise=None):
        # print(x[1])
        out = self.embedding(x[0])
        # print(out.shape)
        if noise is not None:
            out = out + noise
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
