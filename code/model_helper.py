import torch
import utils
from torch.utils.data import DataLoader
from data_helper import BiDataloader, EvalDataset


def train_step(model, data, kg, optimizer, scheduler):
    """
    A single train step. Apply back-propation and return the loss. 单个训练步骤。应用反向传播并返回损失值。
    :param model:
    :param data:
    :param kg:
    :param optimizer:
    :param scheduler:
    :return:
    """

    cfg = utils.get_global_config()
    device = torch.device(cfg.device)
    model.train() # 设置为训练模式
    optimizer.zero_grad() # 梯度清零

    (src, rel, _), label, rm_edges = data # rm_edges: (bs, n_rm_edges)
    src, rel, label, rm_edges = src.to(device), rel.to(device), label.to(device), rm_edges.to(device) # 将数据转移到GPU上
    # randomly remove the training edges to avoid overfitting. 随机删除训练边以避免过拟合。
    if cfg.rm_rate > 0:
        kg.remove_edges(rm_edges) # 移除边
    score = model(src, rel, kg) # (bs, n_ent)
    loss = model.loss(score, label) # 计算损失
    loss.backward() # 反向传播
    optimizer.step() # 更新参数
    if scheduler:
        scheduler.step() # 更新学习率

    log = {
        'loss': loss.item()
    }

    return log


def evaluate(model, set_flag, kg, record=False) -> dict:
    """
    Evaluate the dataset.
    :param model: model to be evaluated.
    :param set_flag: train / valid / test set data to be evaluated.
    :param kg: kg used to aggregate the embedding.
    :param record: whether to record the rank of all the data.
    :return:
    """
    assert set_flag in ['train', 'valid', 'test']
    model.eval() # 设置为评估模式
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
    device = torch.device(cfg.device)

    eval_h_loader = DataLoader(
        dataset=EvalDataset(set_flag, 'tail_batch'),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.cpu_worker_num,
        collate_fn=EvalDataset.collate_fn
    ) # 评估尾实体预测数据集的数据加载器
    eval_t_loader = DataLoader(
        EvalDataset(set_flag, 'head_batch'),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.cpu_worker_num,
        collate_fn=EvalDataset.collate_fn
    ) # 评估头实体预测数据集的数据加载器
    eval_loader = BiDataloader(eval_h_loader, eval_t_loader)

    hb_metrics, tb_metrics, avg_metrics = {}, {}, {} # 分别记录头实体预测、尾实体预测和平均预测的指标
    metrics = {
        'head_batch': hb_metrics,
        'tail_batch': tb_metrics,
        'average': avg_metrics,
        'ranking': []
    } # 记录所有指标
    hits_range = [1, 3, 10, 100, 1000, round(0.5*n_ent)] # 计算HITS@1、HITS@3、HITS@10、HITS@100、HITS@1000、HITS@(0.5*n_ent)
    with torch.no_grad():
        # aggregate the embedding of entities and relations. 聚合实体和关系的嵌入。
        ent_emb, rel_emb = model.aggragate_emb(kg) # (n_ent, emb_dim), (n_rel, emb_dim)
        for i, data in enumerate(eval_loader):
            # filter_bias: (bs, n_ent)
            (src, rel, dst), filter_bias, mode = data
            src, rel, dst, filter_bias = src.to(device), rel.to(device), dst.to(device), filter_bias.to(device)
            # (bs, n_ent)
            score = model.predictor(ent_emb[src], rel_emb[rel], ent_emb) # 计算得分
            score = score + filter_bias # 加上过滤器偏置

            pos_inds = dst
            batch_size = filter_bias.shape[0]
            pos_score = score[torch.arange(batch_size), pos_inds].unsqueeze(dim=1)
            # compare the positive value with negative values to compute rank and hits. 将正样本与负样本进行比较以计算排名和命中率。
            # when values equal, take the mean of upper and lower bound as the rank. 当值相等时，将上下界的平均值作为排名。
            compare_up = torch.gt(score, pos_score)  # (bs, entity_num), >
            compare_low = torch.ge(score, pos_score)  # (bs, entity_num), >=
            ranking_up = compare_up.to(dtype=torch.float).sum(dim=1) + 1  # (bs, ) 比正样本大的个数+1
            ranking_low = compare_low.to(dtype=torch.float).sum(dim=1)  # include the pos one itself, no need to +1 包括正样本本身，不需要+1
            ranking = (ranking_up + ranking_low) / 2
            if record:
                rank = torch.stack([src, rel, dst, ranking], dim=1)  # (bs, 4)
                metrics['ranking'].append(rank)

            results = metrics[mode] # 记录指标
            results['MR'] = results.get('MR', 0.) + ranking.sum().item() # sum the rank of all the data. 将所有数据的排名相加。
            results['MRR'] = results.get('MRR', 0.) + (1 / ranking).sum().item() # sum the reciprocal rank of all the data. 将所有数据的倒数排名相加。
            for k in hits_range:
                results['HITS@{}'.format(k)] = results.get('HITS@{}'.format(k), 0.) + \
                                               (ranking <= k).to(torch.float).sum().item() # sum the hits@k of all the data. 将所有数据的hits@k相加。
            results['n_data'] = results.get('n_data', 0) + batch_size # sum the number of data. 将所有数据的数量相加。

        assert metrics['head_batch']['n_data'] == metrics['tail_batch']['n_data']

        for k, results in metrics.items():
            if k in ['ranking', 'average']:
                continue
            results['MR'] /= results['n_data'] # 计算平均排名
            results['MRR'] /= results['n_data'] # 计算平均倒数排名
            for j in hits_range:
                results['HITS@{}'.format(j)] /= results['n_data'] # 计算平均hits@k

        # average the hb and tb values to get the final reports of the model. 将hb和tb的值平均以获得模型的最终报告。
        for k, value in metrics['head_batch'].items():
            metrics['average'][k] = (metrics['head_batch'][k] + metrics['tail_batch'][k]) / 2 # 计算平均值

        # sort in the ranking order and return the top-k results. 按排名顺序排序并返回前k个结果。
        if record:
            metrics['ranking'] = torch.cat(metrics['ranking'], dim=0).tolist() # 将所有数据的排名拼接起来。
            metrics['ranking'] = sorted(metrics['ranking'], key=lambda x: x[3], reverse=True) # 按排名降序排序。

    return metrics

