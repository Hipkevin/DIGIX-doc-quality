# 文章质量类型判别

[NOTE]代码未重构

## 数据描述
训练集包含字段

    id
    title - 文章标题
    body - 文章内容
    category - 文章分类（无明确的指示含义）
    doctype - 质量类型

测试集不含doctype

## 主要方法
baseline: BERT(title)*fine-tune

contrastive learning: 
BERT-pretrain(title_train, pos, neg) + BERT-pretrained(title)*fine-tune

采样方法：

训练测试划分：分层抽样
contrastive pair 采样：按照标签概率分布抽样tag，在tag内的样本中随机选择query和key_positive，随机在tag以外的标签中随机选择key_negative

[NOTE] contrastive learning 需要在train上做采样，防止data leakage