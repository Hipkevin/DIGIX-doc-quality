import torch.nn as nn

from pytorch_pretrained_bert import BertModel

class SimCSE(nn.Module):
    def __init__(self, config):
        super(SimCSE, self).__init__()

        self.encoder = BertBasedEncoder(config.bert_path)

        self.classifier = nn.Linear(768, config.tag_num)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, kp, kn):

        # contrastive pair 编码
        q_vec = self.encoder(q)
        kp_vec = self.encoder(kp)
        kn_vec = self.encoder(kn)

        return q_vec, kp_vec, kn_vec

    def predict(self, x):

        # 句子分类
        x_vec = self.encoder(x)
        output = self.classifier(x_vec)
        output = self.dropout(output)

        return output

class BertBasedEncoder(nn.Module):
    def __init__(self, bert_path):
        super(BertBasedEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(bert_path)

    def forward(self, x):
        # all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
        # output_all_encoded_layers为False时，all_encoder_layers为最后一层输出
        # pooled_output为最后一层输出的pooling
        # 该pooling对CLS做线性变换+激活 (Linear+tanh)
        _, CLS_output = self.bert(x, output_all_encoded_layers=False)

        return CLS_output