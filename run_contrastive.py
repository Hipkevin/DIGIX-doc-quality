import torch

from util.model import SimCSE
from util.infoNCE import SupervisedInfoNCELoss
from util.dataTool import TitleDataSet, ContrastiveTitleDataset
from config import Config

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from time import time


def pretrain(model, sampling_range, config):
    print("Pre-training...")
    contrast_dataset = ContrastiveTitleDataset(config, sampling_range)
    contrast_loader = DataLoader(contrast_dataset, batch_size=config.pretrain_batch_size, shuffle=True)

    contrastive_criterion = SupervisedInfoNCELoss(config)
    contrastive_optimizer = torch.optim.AdamW(params=model.parameters(),
                                              lr=config.pretrain_learning_rate,
                                              weight_decay=config.pretrain_weight_decay)
    scheduler = CosineAnnealingWarmRestarts(contrastive_optimizer, T_0=config.T_0, T_mult=config.T_multi, eta_min=1e-8)

    tik = time()
    model.train()
    for epoch in range(config.pretrain_epoch):

        for idx, data in enumerate(contrast_loader):

            q, k_p, k_n = data[0].to(config.device), data[1].to(config.device), data[2].to(config.device)

            q_vec, kp_vec, kn_vec = model(q, k_p, k_n)
            pre_loss = contrastive_criterion(q_vec, kp_vec, kn_vec)

            contrastive_optimizer.zero_grad()
            pre_loss.backward()
            contrastive_optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch: {epoch + 1} batch: {idx} | loss: {pre_loss}")

        scheduler.step()

    tok = time()
    print(f"times: {tok - tik}\n")

    return model


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
    train_sampler, test_sampler = data_set.getSampler()

    train_loader = DataLoader(data_set, batch_size=config.batch_size, sampler=train_sampler)
    test_loader = DataLoader(data_set, batch_size=config.batch_size, sampler=test_sampler)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.learning_rate, weight_decay=config.weight_decay)

    tok = time()
    print(f"times: {tok - tik}\n")

    model = pretrain(model, data_set.train_index, config)

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
                print(f"Epoch: {epoch + 1} batch: {idx} | loss: {loss}")

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