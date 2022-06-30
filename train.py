# coding: UTF-8
from datetime import timedelta
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer
from sklearn import metrics
import time
from tensorboardX import SummaryWriter
import tqdm

from data import load_data, DatasetIterator
from model import Config, Model
from params import epsilon, alpha_1, FREE_OPT_NUM, PGD_OPT_NUM


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def load_checkpoint(model, optimizer, model_dir, optim_dir):
    if model_dir is not None:
        model_dict = torch.load(model_dir)
        print(type(model_dict))
        print(model_dict.keys())
        model.load_state_dict(model_dict)
        print('Loading checkpoint!')
        # optimizer.load_state_dict(model_dict = torch.load(optim_dir))
    return model, optimizer



def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if os.path.exists(config.model_dir):
        model, optimizer = load_checkpoint(model, optimizer, config.model_dir, config.optim_dir)


    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0       # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0      # 记录上次验证集loss下降的batch数
    flag = False          # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_dir + '/' + config.mode)

    if config.mode in['FREE', 'PGD', "FGSM", "MIX"]:
        delta = torch.zeros(config.batch_size, 32, config.embedding_dim)
        delta.requires_grad = True

    for epoch in range(config.num_epochs):
        print('\nEpoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for trains, labels in train_iter:
            rand = np.random.random()
            if config.mode == 'FREE' or (config.mode == 'MIX' and rand < 0.25):
                for _ in range(FREE_OPT_NUM):
                    outputs = model(trains, delta[:trains[0].size(0)])
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    grad = delta.grad.detach()
                    delta.data = delta + epsilon * torch.sign(grad)
                    delta.data[:trains[0].size(0)] = clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                    delta.grad.zero_()
            elif config.mode == 'PGD' or (config.mode == 'MIX' and rand < 0.5):
                # 附加PGD操作的优化步骤
                for _ in range(PGD_OPT_NUM):
                    outputs = model(trains, delta[:trains[0].size(0)])
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    grad = delta.grad.detach()
                    delta.data.uniform_(-epsilon, epsilon)
                    delta.data = delta + alpha_1 * torch.sign(grad)
                    delta.data[:trains[0].size(0)] = clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                    delta.grad.zero_()
            elif config.mode == "FGSM" or (config.mode == 'MIX' and rand < 0.75):
                # 无任何操作的优化步骤
                outputs = model(trains, delta[:trains[0].size(0)])
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                grad = delta.grad.detach()
                delta.data.uniform_(-epsilon, epsilon)
                delta.data = delta + alpha_1 * torch.sign(grad)
                delta.data[:trains[0].size(0)] = clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                delta.grad.zero_()

                outputs = model(trains, delta[:trains[0].size(0)])
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

                grad = delta.grad.detach()
                delta.data.uniform_(-epsilon, epsilon)
                delta.data = delta + alpha_1 * torch.sign(grad)
                delta.data[:trains[0].size(0)] = clamp(delta[:trains[0].size(0)], -epsilon, epsilon)
                delta.grad.zero_()
            else:
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.model_dir)
                    torch.save(model.state_dict(), config.optim_dir)

                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print("Epoch:", epoch + 1, msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, Auto-Early-Stopping...")
                flag = True
                break
        if flag:
            break

    writer.close()
    tt(config, model, test_iter, config.mode)


def tt(config, model, test_iter, output_file):
    # test
    model.load_state_dict(torch.load(config.model_dir))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    with open(output_file, "w", encoding='utf-8') as writer:
        writer.write(str(msg.format(test_loss, test_acc)) + '\n')
        writer.write("\nPrecision, Recall and F1-Score...\n")
        writer.write(str(test_report) + '\n')
        writer.write("\nConfusion Matrix...\n")
        writer.write(str(test_confusion) + "\n")

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)



def main():

    tokenizer = BertTokenizer("./vocab.txt")
    train_data = load_data('./mini-train.txt', tokenizer)
    dev_data = load_data('./mini-dev.txt', tokenizer)
    test_data = load_data('./mini-test.txt', tokenizer)
    config = Config('./mini-train.txt', './mini-dev.txt', './mini-test.txt', "./vocab.txt", mode="MIX")
    assert config.mode in ["FGSM", "PLAIN", "PGD", "FREE", "MIX"]
    os.makedirs(config.output_fir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    model = Model(config)
    train_data_iter = DatasetIterator(train_data, 128, model.device)
    dev_data_iter = DatasetIterator(dev_data, 128, model.device)
    test_data_iter = DatasetIterator(test_data, 128, model.device)
    train(config, model, train_data_iter, dev_data_iter, test_data_iter)


if __name__ == '__main__':
    main()