import logging
import numpy as np
import torch
from torch import tensor
from os.path import join
import utils
from typing import Tuple, List
from torch.utils.data import Dataset
from itertools import chain
import dgl
from collections import defaultdict
import math


def construct_dict(dir_path):
    """
    construct the entity, relation dict 构造实体，关系字典
    :param dir_path: data directory path
    :return:
    """
    ent2id, rel2id = dict(), dict()

    # index entities / relations in the occurence order in train, valid and test set 以在train，valid和test集中出现的顺序对实体/关系进行索引
    train_path, valid_path, test_path = join(dir_path, 'train.txt'), join(dir_path, 'valid.txt'), join(dir_path, 'test.txt')
    for path in [train_path, valid_path, test_path]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.split('\t')
                t = t[:-1]  # remove \n
                if h not in ent2id:
                    ent2id[h] = len(ent2id)
                if t not in ent2id:
                    ent2id[t] = len(ent2id)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)

    # arrange the items in id order 以id顺序排列项目
    ent2id, rel2id = dict(sorted(ent2id.items(), key=lambda x: x[1])), dict(sorted(rel2id.items(), key=lambda x: x[1]))
    
    return ent2id, rel2id


def read_data(set_flag):
    """
    read data from file
    :param set_flag: train / valid / test set flag
    :return:
    """
    assert set_flag in [
        'train', 'valid', 'test',
        ['train', 'valid'], ['train', 'valid', 'test']
    ]
    cfg = utils.get_global_config()
    dir_p = join(cfg.dataset_dir, cfg.dataset)
    ent2id, rel2id = construct_dict(dir_p)

    # read the file
    if set_flag in ['train', 'valid', 'test']:
        path = join(dir_p, '{}.txt'.format(set_flag))
        file = open(path, 'r', encoding='utf-8')
    elif set_flag == ['train', 'valid']:
        path1 = join(dir_p, 'train.txt')
        path2 = join(dir_p, 'valid.txt')
        file1 = open(path1, 'r', encoding='utf-8')
        file2 = open(path2, 'r', encoding='utf-8')
        file = chain(file1, file2)
    elif set_flag == ['train', 'valid', 'test']:
        path1 = join(dir_p, 'train.txt')
        path2 = join(dir_p, 'valid.txt')
        path3 = join(dir_p, 'test.txt')
        file1 = open(path1, 'r', encoding='utf-8')
        file2 = open(path2, 'r', encoding='utf-8')
        file3 = open(path3, 'r', encoding='utf-8')
        file = chain(file1, file2, file3)
    else:
        raise NotImplementedError

    src_list = []
    dst_list = []
    rel_list = []
    pos_tails = defaultdict(set)
    pos_heads = defaultdict(set)
    pos_rels = defaultdict(set)

    for i, line in enumerate(file):
        h, r, t = line.strip().split('\t') # line.strip()去掉字符串首尾的空格，\t制表符，\n换行符等等
        h, r, t = ent2id[h], rel2id[r], ent2id[t]
        src_list.append(h)
        dst_list.append(t)
        rel_list.append(r)

        # format data in query-answer form 以查询-答案的形式格式化数据
        # (h, r, ?) -> t, (?, r, t) -> h
        pos_tails[(h, r)].add(t)
        pos_heads[(r, t)].add(h)
        pos_rels[(h, t)].add(r)  # edge relations 边关系
        pos_rels[(t, h)].add(r+len(rel2id))  # inverse relations 逆关系

    output_dict = {
        'src_list': src_list, # source entity 源实体
        'dst_list': dst_list, # destination entity 目标实体
        'rel_list': rel_list, # relation 关系
        'pos_tails': pos_tails, # positive tails
        'pos_heads': pos_heads, # positive heads
        'pos_rels': pos_rels # positive relations
    }

    return output_dict


def construct_kg(set_flag, directed=False):
    """
    construct kg.
    :param set_flag: train / valid / test set flag, use which set data to construct kg. 训练/验证/测试集标志，使用哪个数据集构建图
    :param directed: whether add inverse version for each edge, to make a undirected graph. 是否添加反向边，以构建无向图
    False when training SE-GNN model, True for comuting SE metrics. 训练SE-GNN模型时为False，计算SE指标时为True
    :return:
    """
    assert directed in [True, False]
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

    d = read_data(set_flag)
    src_list, dst_list, rel_list = [], [], []

    # eid: record the edge id of queries, for randomly removing some edges when training 记录查询的边id，以便在训练时随机删除一些边
    eid = 0
    hr2eid, rt2eid = defaultdict(list), defaultdict(list)
    for h, t, r in zip(d['src_list'], d['dst_list'], d['rel_list']):
        if directed:
            src_list.extend([h])
            dst_list.extend([t])
            rel_list.extend([r])
            hr2eid[(h, r)].extend([eid])
            rt2eid[(r, t)].extend([eid])
            eid += 1
        else:
            # include the inverse edges 包括反向边
            # inverse rel id: original id + rel num 反向关系id：原始id+关系数
            src_list.extend([h, t])
            dst_list.extend([t, h])
            rel_list.extend([r, r + n_rel])
            hr2eid[(h, r)].extend([eid, eid + 1])
            rt2eid[(r, t)].extend([eid, eid + 1])
            eid += 2

    src, dst, rel = tensor(src_list), tensor(dst_list), tensor(rel_list)

    return src, dst, rel, hr2eid, rt2eid


def get_kg(src, dst, rel, device):
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
    kg = dgl.graph((src, dst), num_nodes=n_ent)
    kg.edata['rel_id'] = rel
    kg = kg.to(device)
    return kg


class TrainDataset(Dataset):
    """
    Training data is in query-answer format: (h, r) -> tails, (r, t) -> heads 训练数据以查询-答案格式：(h, r) -> tails, (r, t) -> heads
    """
    def __init__(self, set_flag, hr2eid, rt2eid):
        assert set_flag in ['train', 'valid', 'test']
        logging.info('---Load Train Data---')
        self.cfg = utils.get_global_config()
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent'] # 实体数
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel'] # 关系数

        self.d = read_data(set_flag)
        self.query = [] # (h, r, ?) -> t, (?, r, t) -> h 查询
        self.label = [] # tails, heads 真实尾实体，真实头实体
        self.rm_edges = [] # 记录查询的边id，以便在训练时随机删除一些边
        self.set_scaling_weight = [] # 用于计算SE指标的权重（未使用）

        # pred tails 预测尾实体
        for k, v in self.d['pos_tails'].items():
            self.query.append((k[0], k[1], -1))
            self.label.append(list(v))
            # randomly removing edges later
            self.rm_edges.append(hr2eid[k])

        # pred heads 预测头实体
        for k, v in self.d['pos_heads'].items():
            # inverse relation 逆关系
            self.query.append((k[1], k[0] + self.n_rel, -1))
            self.label.append(list(v))
            self.rm_edges.append(rt2eid[k])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        h, r, t = self.query[item] # 查询
        label = self.get_onehot_label(self.label[item]) # one-hot编码

        rm_edges = torch.tensor(self.rm_edges[item], dtype=torch.int64)
        rm_num = math.ceil(rm_edges.shape[0] * self.cfg.rm_rate) # 随机删除的边数
        rm_inds = torch.randperm(rm_edges.shape[0])[:rm_num] # 随机删除的边的索引
        rm_edges = rm_edges[rm_inds] # 随机删除的边

        return (h, r, t), label, rm_edges # 查询，真实尾实体，随机删除的边

    def get_onehot_label(self, label):
        onehot_label = torch.zeros(self.n_ent)
        onehot_label[label] = 1
        if self.cfg.label_smooth != 0.0:
            onehot_label = (1.0 - self.cfg.label_smooth) * onehot_label + (1.0 / self.n_ent) # 平滑标签，防止模型过拟合

        return onehot_label

    # 这个函数是用来获取标签对应的正样本节点的索引的。
    # 输入: label 表示正样本的标签，即表示正确关系的节点在节点列表中的索引。
    # 返回: 一个长度为 n_ent 的布尔类型的张量，其中对应正样本节点的位置为 True，其他位置为 False。
    def get_pos_inds(self, label):
        pos_inds = torch.zeros(self.n_ent).to(torch.bool)
        pos_inds[label] = True
        return pos_inds
    
    @staticmethod
    def collate_fn(data):
        src = [d[0][0] for d in data]
        rel = [d[0][1] for d in data]
        dst = [d[0][2] for d in data]
        label = [d[1] for d in data]  # list of list
        rm_edges = [d[2] for d in data]

        src = torch.tensor(src, dtype=torch.int64)
        rel = torch.tensor(rel, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)  # (bs, )
        label = torch.stack(label, dim=0)  # (bs, n_ent)
        rm_edges = torch.cat(rm_edges, dim=0)  # (n_rm_edges, )

        return (src, rel, dst), label, rm_edges


class EvalDataset(Dataset):
    """
    Evaluating data is in triple format. Keep one for head-batch and tail-batch respectively,
    for computing each direction's metrics conveniently. 评估数据以三元组格式。分别保留一个用于头批次和尾批次，
    """
    def __init__(self, set_flag, mode):
        assert set_flag in ['train', 'valid', 'test']
        assert mode in ['head_batch', 'tail_batch']
        self.cfg = utils.get_global_config()
        dataset = self.cfg.dataset
        self.mode = mode
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent'] # 实体数
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel'] # 关系数

        self.d = read_data(set_flag)
        self.trip = [_ for _ in zip(self.d['src_list'], self.d['rel_list'], self.d['dst_list'])] # 数据集中的三元组
        self.d_all = read_data(['train', 'valid', 'test']) # 读取所有数据
        self.pos_t = self.d_all['pos_tails'] # 所有查询尾实体的正样本
        self.pos_h = self.d_all['pos_heads'] # 所有查询头实体的正样本

    def __len__(self):
        return len(self.trip)

    def __getitem__(self, item):
        h, r, t = self.trip[item]

        if self.mode == 'tail_batch':
            # filter_bias, remove other ground truthes when ranking 
            filter_bias = np.zeros(self.n_ent, dtype=np.float)
            filter_bias[list(self.pos_t[(h, r)])] = -float('inf')
            filter_bias[t] = 0. # filter_bias 中正样本的尾实体对应的位置被设置为负无穷，而当前三元组中的尾实体位置被设置为零
        elif self.mode == 'head_batch':
            filter_bias = np.zeros(self.n_ent, dtype=np.float)
            filter_bias[list(self.pos_h[(r, t)])] = -float('inf')
            filter_bias[h] = 0. # filter_bias 中正样本的头实体对应的位置被设置为负无穷，而当前三元组中的头实体位置被设置为零
            h, r, t = t, r+self.n_rel, h # 交换头实体和尾实体，这样就可以使用同一个模型来计算头实体的预测值
        else:
            raise NotImplementedError

        return (h, r, t), filter_bias.tolist(), self.mode # 返回查询实体，查询关系，过滤偏置，模式

    @staticmethod
    def collate_fn(data: List[Tuple[tuple, list, str]]):
        h = [d[0][0] for d in data]
        r = [d[0][1] for d in data]
        t = [d[0][2] for d in data]
        filter_bias = [d[1] for d in data]
        mode = data[0][-1]

        h = torch.tensor(h, dtype=torch.int64)
        r = torch.tensor(r, dtype=torch.int64)
        t = torch.tensor(t, dtype=torch.int64) # (bs, )
        filter_bias = torch.tensor(filter_bias, dtype=torch.float) # (bs, n_ent)

        return (h, r, t), filter_bias, mode


class BiDataloader(object):
    """
    Combine the head-batch and tail-batch evaluation dataloader. 将头批次和尾批次评估数据加载器组合在一起。
    """
    def __init__(self, h_loader: iter, t_loader: iter):
        self.h_loader_len = len(h_loader) # 头批次评估数据加载器的长度
        self.t_loader_len = len(t_loader) # 尾批次评估数据加载器的长度
        self.h_loader_step = 0 # 头批次评估数据加载器的步数
        self.t_loader_step = 0 # 尾批次评估数据加载器的步数
        self.total_len = self.h_loader_len + self.t_loader_len # 总长度
        self.h_loader = self.inf_loop(h_loader) # 无限循环的头批次评估数据加载器
        self.t_loader = self.inf_loop(t_loader) # 无限循环的尾批次评估数据加载器
        self._step = 0 # 当前步数

    def __next__(self):
        if self._step == self.total_len: # 如果当前步数等于总长度
            # ensure that all the data of two dataloaders is accessed 确保两个数据加载器的所有数据都被访问
            assert self.h_loader_step == self.h_loader_len
            assert self.t_loader_step == self.t_loader_len
            self._step = 0
            self.h_loader_step = 0
            self.t_loader_step = 0
            raise StopIteration # 抛出停止迭代异常
        if self._step % 2 == 0: # 如果当前步数是偶数
            # head-batch
            if self.h_loader_step < self.h_loader_len: # 如果头批次评估数据加载器的步数小于头批次评估数据加载器的长度
                data = next(self.h_loader)
                self.h_loader_step += 1
            else:
                # if head-batch complets, return tail-batch 如果头批次完成，则返回尾批次
                data = next(self.t_loader)
                self.t_loader_step += 1
        else:  # 如果当前步数是奇数
            # tail-batch
            if self.t_loader_step < self.t_loader_len: # 如果尾批次评估数据加载器的步数小于尾批次评估数据加载器的长度
                data = next(self.t_loader)
                self.t_loader_step += 1
            else:
                # if tail-batch complets, return head-batch 如果尾批次完成，则返回头批次
                data = next(self.h_loader)
                self.h_loader_step += 1
        self._step += 1
        return data

    def __iter__(self):
        return self

    def __len__(self):
        return self.total_len

    @staticmethod
    def inf_loop(dataloader): # 无限循环的数据加载器
        """
        infinite loop
        :param dataloader:
        :return:
        """
        while True:
            for data in dataloader:
                yield data
