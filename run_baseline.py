import torch

from util.model import SimCSE
from util.dataTool import TitleDataSet
from config import Config

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from time import time


if __name__ == '__main__':
    config = Config()

    print("Loading model...")
    tik = time()
    model = SimCSE(config).to(config.device)
    tok = time()
    print(f"times: {tok - tik}\n")

    print("Loading data...")
    tik = time()
    data_set = TitleDataSet(config)
    train_sampler, test_sampler = data_set.getSampler()  # 使用sampler划分数据集

    train_loader = DataLoader(data_set, batch_size=config.batch_size, sampler=train_sampler)
    test_loader = DataLoader(data_set, batch_size=config.batch_size, sampler=test_sampler)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    tok = time()
    print(f"times: {tok - tik}\n")

    print("Training...")
    tik = time()
    model.train()
    for epoch in range(config.epoch_size):

        for idx, data in enumerate(train_loader):

            x, y = data[0].to(config.device), data[1].to(config.device)

            predict = model.predict(x)
            loss = criterion(predict, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch: {epoch+1} batch: {idx} | loss: {loss}")
    tok = time()
    print(f"times: {tok - tik}\n")

    print("Testing...")
    tik = time()

    model.eval()
    p_collection = list()
    y_collection = list()
    for idx, data in enumerate(test_loader):
        x, y = data[0].to(config.device), data[1].to('cpu')

        predict = torch.argmax(model.predict(x), dim=1).to('cpu')
        p_collection += predict.tolist()
        y_collection += y.tolist()

    tok = time()
    print(f"times: {tok - tik}\n")

    print(classification_report(y_true=y_collection, y_pred=p_collection))

    macro_f1 = f1_score(y_true=y_collection, y_pred=p_collection, average="macro")
    micro_f1 = f1_score(y_true=y_collection, y_pred=p_collection, average="micro")
    print("macro-F1: ", macro_f1)
    print("micro-F1: ", micro_f1)