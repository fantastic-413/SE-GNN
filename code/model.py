#!/usr/bin/python3
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import utils
from utils import get_param
from decoder import ConvE


class SE_GNN(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']

        # entity embedding
        self.ent_emb = get_param(self.n_ent, h_dim)# n_ent * dim [14541,450] / [40943,200]

        # gnn layer
        self.kg_n_layer = self.cfg.kg_layer # 2 / 1
        # relation SE layer
        self.edge_layers = nn.ModuleList([EdgeLayer(h_dim) for _ in range(self.kg_n_layer)])
        # entity SE layer
        self.node_layers = nn.ModuleList([NodeLayer(h_dim) for _ in range(self.kg_n_layer)])
        # triple SE layer
        self.comp_layers = nn.ModuleList([CompLayer(h_dim) for _ in range(self.kg_n_layer)])

        # relation embedding for aggregation 聚合的关系嵌入
        # 创建一个包含 self.kg_n_layer 个 PyTorch 参数对象的列表，其中每个参数对象都是一个形状为 (self.n_rel * 2, h_dim) 的张量
        self.rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])# 2n_rel * dim

        # relation embedding for prediction 预测的关系嵌入
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * self.kg_n_layer, h_dim) # [200,200]
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim) # [474,450]

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)# ker_sz是卷积核的大小
        # loss
        # nn.BCELoss() 是一个 PyTorch 中的二元交叉熵损失函数，它的输入是一个形状为 (N, *) 的张量，其中 N 是 batch size，* 是任意形状，输出是一个形状为 (1, ) 的张量
        self.bce = nn.BCELoss()

        # nn.Dropout() 是一个 PyTorch 中的正则化层，用于在训练过程中对输入张量的随机单元进行丢弃（设置为0），以防止过拟合。
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)
        # 激活函数
        self.act = nn.Tanh()

    def forward(self, h_id, r_id, kg):
        """
        matching computation between query (h, r) and answer t. (h,r)与t的匹配计算
        :param h_id: head entity id, (bs, )
        :param r_id: relation id, (bs, )
        :param kg: aggregation graph
        :return: matching score, (bs, n_ent)
        """
        # aggregate embedding 聚合嵌入
        ent_emb, rel_emb = self.aggragate_emb(kg) # (n_ent, h_dim), (n_rel * 2, h_dim)

        head = ent_emb[h_id] # (bs, h_dim)
        rel = rel_emb[r_id] # (bs, h_dim)
        # (bs, n_ent)
        score = self.predictor(head, rel, ent_emb) # (bs, n_ent)

        return score

    def loss(self, score, label):
        # (bs, n_ent)
        loss = self.bce(score, label) # (bs, n_ent)

        return loss

    def aggragate_emb(self, kg):
        """
        aggregate embedding. 
        :param kg:
        :return:
        """
        ent_emb = self.ent_emb # (n_ent, h_dim)
        rel_emb_list = []
        for edge_layer, node_layer, comp_layer, rel_emb in zip(self.edge_layers, self.node_layers, self.comp_layers, self.rel_embs):
            ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb) # (n_ent, h_dim), (n_rel * 2, h_dim)
            edge_ent_emb = edge_layer(kg, ent_emb, rel_emb) # (n_ent, h_dim)
            node_ent_emb = node_layer(kg, ent_emb) # (n_ent, h_dim)
            comp_ent_emb = comp_layer(kg, ent_emb, rel_emb) # (n_ent, h_dim)
            ent_emb = ent_emb + edge_ent_emb + node_ent_emb + comp_ent_emb # (n_ent, h_dim)
            rel_emb_list.append(rel_emb) # [(n_rel * 2, h_dim), ...]

        # 在这个条件分支中，如果模型的配置参数pred_rel_w为True，则模型将拼接多层图卷积层的输出作为每个关系对应的向量表示，然后再用一个线性变换矩阵self.rel_w对这些向量表示进行变换；否则，模型直接使用预定义的关系向量表示self.pred_rel_emb。
        # 在使用FB15k-237数据集时，作者选择不使用self.rel_w进行线性变换，这是因为该数据集的关系比较多，许多关系只出现了很少的次数，因此将它们的向量表示用一个线性变换矩阵来表示可能会导致过拟合的情况。相反，使用预定义的向量表示可以更好地泛化到未见过的关系。
        # 而在使用WN18RR数据集时，作者选择使用self.rel_w进行线性变换，这是因为该数据集中的关系较少，每个关系都有足够的训练样本，因此使用线性变换矩阵可以更好地捕捉关系之间的语义差异。
        if self.cfg.pred_rel_w:
            pred_rel_emb = torch.cat(rel_emb_list, dim=1) # (n_rel * 2, h_dim * kg_n_layer)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w) # (n_rel * 2, h_dim)
        else:
            pred_rel_emb = self.pred_rel_emb # (n_rel * 2, h_dim)

        return ent_emb, pred_rel_emb # (n_ent, h_dim), (n_rel * 2, h_dim)


class CompLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.comp_op = self.cfg.comp_op
        assert self.comp_op in ['add', 'mul']

        self.neigh_w = get_param(h_dim, h_dim) # (h_dim, h_dim) 线性变换矩阵
        self.act = nn.Tanh() # 非线性激活函数
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb # 节点的特征存储到图中
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id] # 边的特征存储到图中
            # neihgbor entity and relation composition 邻居实体和关系组合
            if self.cfg.comp_op == 'add':
                kg.apply_edges(fn.u_add_e('emb', 'emb', 'comp_emb')) # 将源节点的特征和边的特征相加，然后将结果存储到名为“comp_emb”的边的特征中
            elif self.cfg.comp_op == 'mul':
                kg.apply_edges(fn.u_mul_e('emb', 'emb', 'comp_emb')) # 将源节点的特征和边的特征相乘，然后将结果存储到名为“comp_emb”的边的特征中
            else:
                raise NotImplementedError

            # attention 注意力机制
            kg.apply_edges(fn.e_dot_v('comp_emb', 'emb', 'norm'))  # (n_edge, 1) # 将边的特征与目标节点的特征相乘，然后将结果存储到名为“norm”的边的特征中
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm']) # 对边的特征进行归一化得到边的权重
            
            # agg 聚合
            kg.edata['comp_emb'] = kg.edata['comp_emb'] * kg.edata['norm'] # 将边特征与边权重相乘得到加权边特征
            kg.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh')) 
            # 将每个边的 'emb' 特征传递到目标节点，存储在目标节点的 mailbox 中
            # 将 mailbox 中的特征聚合到目标节点的 'neigh' 特征中

            neigh_ent_emb = kg.ndata['neigh'] # 获得聚合后每个目标节点的特征

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w) # 线性变换

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb) # 批归一化

            neigh_ent_emb = self.act(neigh_ent_emb) # 非线性激活函数

        return neigh_ent_emb # 返回聚合后每个目标节点的特征


class NodeLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.neigh_w = get_param(h_dim, h_dim) # (h_dim, h_dim) 线性变换矩阵
        self.act = nn.Tanh() # 非线性激活函数
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb # 节点的特征存储到图中

            # attention 注意力机制
            kg.apply_edges(fn.u_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1) # 对于每条边，计算其两端节点的特征的点积，并将结果存储在名为norm的边特征中
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm']) # 对边的特征进行归一化得到边的权重

            # agg 聚合
            kg.update_all(fn.u_mul_e('emb', 'norm', 'm'), fn.sum('m', 'neigh')) 
            # 将源节点特征与边权重相乘得到加权边特征
            # 然后将这些特征发送到目标节点，最后对目标节点的所有加权边特征求和得到该节点的特征
            
            neigh_ent_emb = kg.ndata['neigh'] # 获得聚合后每个目标节点的特征

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w) # 线性变换

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb) # 批归一化

            neigh_ent_emb = self.act(neigh_ent_emb) # 非线性激活函数

        return neigh_ent_emb # 返回聚合后每个目标节点的特征


class EdgeLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.neigh_w = utils.get_param(h_dim, h_dim) # (h_dim, h_dim) 线性变换矩阵
        self.act = nn.Tanh() # 非线性激活函数
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        assert kg.number_of_nodes() == ent_emb.shape[0]
        assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope(): # 部范围用于避免上一次计算的影响
            kg.ndata['emb'] = ent_emb # 节点的特征存储到图中
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id] # 边的特征存储到图中

            # attention 注意力机制
            kg.apply_edges(fn.e_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1) # 对于每条边，计算边的特征和目标节点的特征的点积，并将结果存储在名为norm的边特征中
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm']) # 对边的特征进行归一化得到边的权重

            # agg 聚合
            kg.edata['emb'] = kg.edata['emb'] * kg.edata['norm'] # 将边的特征与边的权重相乘
            kg.update_all(fn.copy_e('emb', 'm'), fn.sum('m', 'neigh'))
            # fn.copy_e('emb', 'm') 的作用是将每个边的 'emb' 特征传递到目标节点，存储在目标节点的 mailbox 中
            # fn.sum('m', 'neigh') 的作用是将 mailbox 中的特征求和，存储在目标节点的 'neigh' 特征中

            neigh_ent_emb = kg.ndata['neigh'] # 获得聚合后每个目标节点的特征

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w) # 线性变换

            if callable(self.bn): # 批归一化
                neigh_ent_emb = self.bn(neigh_ent_emb) # 批归一化

            neigh_ent_emb = self.act(neigh_ent_emb) # 激活函数

        return neigh_ent_emb # 返回聚合后每个目标节点的特征 
