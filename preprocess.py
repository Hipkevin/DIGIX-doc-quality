import random
from tqdm import tqdm

# 抽取带标签的数据
def extract():
    with open('data/doc_quality_data_train.json', encoding='utf-8') as file:
        samples = file.readlines()

    with open('data/train.txt', 'a+', encoding='utf-8') as file:

        for s in tqdm(samples):
            data = eval(s)

            if data['doctype']:
                file.write(str(data).replace('\n', '') + '\n')
                print(data)

# EDA-统计文本长度
def statistic():
    with open('data/train.txt', encoding='utf-8') as file:
        samples = file.read().strip().split('\n')

    tag_dict = dict()
    with open('data/statistic.txt', 'a+', encoding='utf-8') as file:
        for s in samples:
            data = eval(s)
            file.write(str(len(data['title'])) + ' ' + str(len(data['body'])) + '\n')

            tag_dict[data['doctype']] = tag_dict.get(data['doctype'], 0) + 1

    print(tag_dict)

# 只取标题
def getTitle():
    with open('data/train.txt', encoding='utf-8') as file:
        samples = file.read().strip().split('\n')

    with open('data/train_title.txt', 'a+', encoding='utf-8') as file:

        for s in tqdm(samples):
            data = eval(s)
            data.pop('body')
            file.write(str(data).replace('\n', '') + '\n')


if __name__ == '__main__':
    pass



