import pickle
import torch
import os

cur_dir = os.path.dirname(__file__)

def load_dataset(path, task_id, pad_size=35):
    task_prefix = f'task{task_id}_'
    contents = []

    with open(os.path.join(cur_dir, '../feature/cache_E/id_imageFeat_ViT-L_14.pkl'), 'rb') as f:
        id_imgFeat = pickle.load(f)

    with open(os.path.join(cur_dir, '../feature/cache_E/id_textFeat_ViT-L_14.pkl'), 'rb') as f:
        id_textFeat = pickle.load(f)

    with open(os.path.join(cur_dir, f'../feature/cache_E/{task_prefix}id_promptTokenID.pkl'), 'rb') as f:
        id_promptTokenID = pickle.load(f)

    with open(os.path.join(cur_dir, f'../feature/cache_E/{task_prefix}id_promptMask.pkl'), 'rb') as f:
        id_promptMask = pickle.load(f)

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split(',')
            id = int(line[0])
            label = int(line[1])
            imgfeature = id_imgFeat[id]
            textfeature = id_textFeat[id]
            promptTokenID = id_promptTokenID[id]
            promptMask = id_promptMask[id]
            contents.append([imgfeature, textfeature, promptTokenID, promptMask, label])
    return contents

def build_dataset(task_id, pad_size = 35):
    task_prefix = f'task{task_id}_'

    train_path = os.path.join(cur_dir, f'../data/E_split/{task_prefix}SintTrain6.csv')
    val_path = os.path.join(cur_dir, f'../data/E_split/{task_prefix}SintVal2.csv')
    test_path = os.path.join(cur_dir, f'../data/E_split/{task_prefix}SintTest2.csv')

    train = load_dataset(train_path, task_id, pad_size=pad_size)
    val = load_dataset(val_path, task_id, pad_size=pad_size)
    test = load_dataset(test_path, task_id, pad_size=pad_size)

    return train, val, test

# 构建迭代器
class DatasetIterater(object):  # 自定义数据集迭代器
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches  # 构建好的数据集
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:  # 不能整除
            self.residue = True #True表示不能整除
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        image_feat = torch.FloatTensor([_[0] for _ in datas]).to(self.device)
        text_feat = torch.FloatTensor([_[1].tolist() for _ in datas]).to(self.device)
        prompt_token_id = torch.LongTensor([_[2].tolist() for _ in datas]).to(self.device)
        prompt_mask = torch.LongTensor([_[3].tolist() for _ in datas]).to(self.device)
        y = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        return (image_feat, text_feat, prompt_token_id, prompt_mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:  #当数据集大小不整除 batch_size时，构建最后一个batch
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)  # 把最后一个batch转换为tensor 并 to(device)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:  # 构建每一个batch
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)  # 把当前batch转换为tensor 并 to(device)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue: #不能整除
            return self.n_batches + 1 #batch数+1
        else:
            return self.n_batches

def build_iterator(dataset, batch_size, device): #构建数据集迭代器
    iter = DatasetIterater(dataset, batch_size, device)
    return iter
