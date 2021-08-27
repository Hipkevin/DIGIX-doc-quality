import torch
import random

from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import StratifiedShuffleSplit

CLS = "[CLS]"
SEP = "[SEP]"
PAD = "[PAD]"
tag_dict = {"人物专栏": 0,
            "情感解读": 1,
            "科普知识文": 2,
            "攻略文": 3,
            "物品评测": 4,
            "治愈系文章": 5,
            "推荐文": 6,
            "深度事件": 7,
            "作品分析": 8,
            "行业解读": 9}


class TitleDataSet(Dataset):

    def __init__(self, config):
        super(TitleDataSet, self).__init__()

        tokenizer = BertTokenizer.from_pretrained(config.bert_path)

        with open(config.data_path, encoding='utf-8') as file:
            samples = file.read().strip().split('\n')

        X = list()
        Y = list()
        for s in samples:
            data = eval(s)

            text = data['title']
            tag = tag_dict.get(data['doctype'])

            text = self.concatenation(text, config.pad_size)

            tokens = tokenizer.tokenize(text)
            tokens += [PAD] * (config.pad_size - len(tokens))

            X.append(tokenizer.convert_tokens_to_ids(tokens))
            Y.append(tag)

        # 训练-测试集划分的采样器
        self.train_index, self.test_index = self.buildSamplerIndex(X, Y, config.cv)

        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def getSampler(self):
        return SubsetRandomSampler(self.train_index), SubsetRandomSampler(self.test_index)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    @staticmethod
    def concatenation(text, pad_size):
        # BERT输入文本拼接

        # available = pad_size - 1 * CLS - n * SEP
        available_len = pad_size - 2

        if len(text) > available_len:
            text = text[0: available_len]

        return CLS + text + SEP

    @staticmethod
    def buildSamplerIndex(X, Y, cv=0.15):
        # 对数据集进行分层抽样，返回划分后的索引
        # 通过该索引构造SubsetRandomSampler，完成数据集划分
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=cv, random_state=0)
        train_index, test_index = list(stratified_split.split(X, Y))[0]

        return train_index.tolist(), test_index.tolist()


class ContrastiveTitleDataset(Dataset):

    def __init__(self, config, sampling_range=[]):
        super(ContrastiveTitleDataset, self).__init__()

        tokenizer = BertTokenizer.from_pretrained(config.bert_path)

        with open(config.data_path) as file:
            samples = file.read().strip().split('\n')

        # tag_map保存各个tag中，样本对应的idx
        X = list()
        tag_map = dict()

        # 过滤测试集中的样本
        if sampling_range:
            samples = [samples[r] for r in sampling_range]

        for idx, s in enumerate(samples):

            data = eval(s)

            text = data['title']
            tag = tag_dict.get(data['doctype'])

            text = self.concatenation(text, config.pad_size)

            tokens = tokenizer.tokenize(text)
            tokens += [PAD] * (config.pad_size - len(tokens))

            X.append(tokenizer.convert_tokens_to_ids(tokens))
            tag_map[tag] = tag_map.get(tag, []) + [idx]

        X = torch.tensor(X, dtype=torch.long)

        # 计算样本分布，用于概率抽样
        distribute = [len(tag_map[k]) for k in range(config.tag_num)]
        self.contrastText = self.buildContrast(X, tag_map, config.sampling_times, distribute)

    @staticmethod
    def buildContrast(X, Y_map, sampling_times, distribute):
        res = list()
        tag = list(range(len(distribute)))

        for i in range(sampling_times):
            sampling_tag = random.choices(tag, weights=distribute)[0]

            query = X[random.choice(Y_map[sampling_tag])]
            k_positive = X[random.choice(Y_map[sampling_tag])]
            k_negative = X[
                # 随机选择除当前tag以外的标签
                # 由于tag为int型，故 new_tag = tag + (tag + offset) % tag_num
                # 其中offset为随机数，且小于tag_num，保证不会回到tag
                random.choice(Y_map[(int(sampling_tag) + random.randint(0, len(distribute) - 1)) % len(distribute)])]

            res.append((query, k_positive, k_negative))

        return res

    @staticmethod
    def concatenation(text, pad_size):
        # available = pad_size - 1 * CLS - n * SEP
        available_len = pad_size - 2

        if len(text) > available_len:
            text = text[0: available_len]

        return CLS + text + SEP

    def __getitem__(self, index):
        return self.contrastText[index]

    def __len__(self):
        # shuffle需要实现该方法
        return len(self.contrastText)