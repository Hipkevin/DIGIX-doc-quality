import torch


class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert_path = "../bert"
        self.data_path = "data/train_title.txt"

        self.tag_num = 10
        self.cv = 0.15

        # contrastive learning 预训练任务参数
        self.pretrain_batch_size = 64
        self.pretrain_epoch = 1
        self.pretrain_learning_rate = 5e-5
        self.pretrain_weight_decay = 1e-1

        # contrastive pair 采样次数
        self.sampling_times = 140000

        # 预训练过程中，学习率余弦退火+热重启
        self.T_0 = 5
        self.T_multi = 2

        # infoNCE loss 控制数值稳定的变量
        self.taf = 1e3

        self.batch_size = 64
        self.epoch_size = 1

        self.pad_size = 70
        self.learning_rate = 5e-5
        self.weight_decay = 1e-1
        self.dropout = 0.5