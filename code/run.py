#!/usr/bin/python3

import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils
from data_helper import TrainDataset, construct_kg, get_kg
import hydra
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
import pickle
from os.path import join
from model_helper import train_step, evaluate
from model import SE_GNN

# 保存模型的参数
def save_model(model, save_variables):
    """
    Save the parameters of the model
    :param model:
    :param save_variables:
    :return:
    """
    cfg = utils.get_global_config()
    pickle.dump(cfg, open('config.pickle', 'wb'))

    state_dict = {
        'model_state_dict': model.state_dict(),  # model parameters
        **save_variables
    }

    torch.save(state_dict, 'checkpoint.torch')

#创建在 warn up 期间线性增加后线性降低的学习率的调度程序。
def get_linear_scheduler_with_warmup(optimizer, warmup_steps: int, max_steps: int):
    """
    Create scheduler with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    #根据当前步骤计算一个比率，优化器的学习率将被乘以该比率。
    def lr_lambda(current_step):
        """
        Compute a ratio according to current step,
        by which the optimizer's lr will be mutiplied.
        :param current_step:
        :return:
        """
        assert current_step <= max_steps
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            return (max_steps - current_step) / (max_steps - warmup_steps)

    assert max_steps >= warmup_steps

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def format_metrics(name, h_metric, t_metric):
    msg_h = name + ' (head) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_t = name + ' (tail) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_avg = name + ' (avg) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_h = msg_h.format(h_metric['MRR'], h_metric['MR'],
                         h_metric['HITS@1'], h_metric['HITS@3'], h_metric['HITS@10'])
    msg_t = msg_t.format(t_metric['MRR'], t_metric['MR'],
                         t_metric['HITS@1'], t_metric['HITS@3'], t_metric['HITS@10'])
    msg_avg = msg_avg.format(
        (h_metric['MRR'] + t_metric['MRR']) / 2,
        (h_metric['MR'] + t_metric['MR']) / 2,
        (h_metric['HITS@1'] + t_metric['HITS@1']) / 2,
        (h_metric['HITS@3'] + t_metric['HITS@3']) / 2,
        (h_metric['HITS@10'] + t_metric['HITS@10']) / 2
    )
    return msg_h, msg_t, msg_avg


@hydra.main(config_path=join('..', 'config'), config_name="config")
def main(config: DictConfig):
    utils.set_global_config(config)
    cfg = utils.get_global_config()
    assert cfg.dataset in cfg.dataset_list

    # remove randomness 移除随机性
    utils.remove_randomness()

    # print configuration
    logging.info('\n------Config------\n {}'.format(utils.filter_config(cfg)))

    # backup the code and configuration 备份代码和配置
    code_dir_path = os.path.dirname(__file__) # 获取当前 Python 文件所在的目录的路径。
    project_dir_path = os.path.dirname(code_dir_path) # 获取当前 Python 文件所在目录的上级目录，即项目的根目录。
    config_dir_path = os.path.join(project_dir_path, 'config') # 将项目根目录和子目录 'config' 的路径组合成一个完整的目录路径。
    hydra_current_dir = os.getcwd() # 获取当前工作目录。
    logging.info('Code dir path: {}'.format(code_dir_path))
    logging.info('Config dir path: {}'.format(config_dir_path))
    logging.info('Model save path: {}'.format(hydra_current_dir))
    os.system('cp -r {} {}'.format(code_dir_path, hydra_current_dir)) # 将当前 Python 文件所在目录的代码复制到当前工作目录中。
    os.system('cp -r {} {}'.format(config_dir_path, hydra_current_dir)) # 将项目根目录下的 'config' 子目录复制到当前工作目录中。

    device = torch.device(cfg.device) # 获取设备。
    model = SE_GNN(cfg.h_dim) # 创建模型。
    model = model.to(device) # 将模型放到设备上。

    # load the knowledge graph 加载知识图谱
    src, dst, rel, hr2eid, rt2eid = construct_kg('train', directed=False) # 从训练集中构建知识图谱。
    kg = get_kg(src, dst, rel, device) #
    kg_out_deg = kg.out_degrees(torch.arange(kg.number_of_nodes(), device=device)) # 获取每个节点的出度。
    kg_zero_deg_num = torch.sum(kg_out_deg < 1).to(torch.int).item() # 统计出度为 0 的节点的数量。
    logging.info('kg # node: {}'.format(kg.number_of_nodes()))
    logging.info('kg # edge: {}'.format(kg.number_of_edges()))
    logging.info('kg # zero deg node: {}'.format(kg_zero_deg_num))

    train_loader = DataLoader(
        TrainDataset('train', hr2eid, rt2eid),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.cpu_worker_num,
        collate_fn=TrainDataset.collate_fn
    )# 生成训练数据集的迭代器。

    logging.info('-----Model Parameter Configuration-----')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' %
                     (name, str(param.size()), str(param.requires_grad)))

    # set optimizer and scheduler 设置优化器和学习率调度器
    n_epoch = cfg.epoch # 获取训练的轮数。
    single_epoch_step = len(train_loader) # 获取每个 epoch 中的步数。
    max_steps = n_epoch * single_epoch_step # 获取训练的总步数。
    warm_up_steps = int(single_epoch_step * cfg.warmup_epoch) # 获取 warm up 的步数。
    init_lr = cfg.learning_rate # 获取初始学习率。
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=init_lr
    ) # 创建优化器。
    scheduler = get_linear_scheduler_with_warmup(optimizer, warm_up_steps, max_steps) # 创建学习率调度器。

    logging.info('Training... total epoch: {0}, step: {1}'.format(n_epoch, max_steps))
    last_improve_epoch = 0 # 上一次提升的 epoch。
    best_mrr = 0. # 最好的 MRR。

    for epoch in range(n_epoch):
        loss_list = [] # 损失列表。
        for batch_data in train_loader:
            train_log = train_step(model, batch_data, kg, optimizer, scheduler)
            loss_list.append(train_log['loss'])
            # get a new kg, since in previous kg some edges are removed. 获取一个新的知识图谱，因为在之前的知识图谱中有一些边被移除了。
            if cfg.rm_rate > 0:
                kg = get_kg(src, dst, rel, device)

        val_metrics = evaluate(model, set_flag='valid', kg=kg)['average'] # 在验证集上评估模型。
        if val_metrics['MRR'] > best_mrr:
            best_mrr = val_metrics['MRR'] # 更新最好的 MRR。
            save_variables = {
                'best_val_metrics': val_metrics
            } # 保存的变量。
            save_model(model, save_variables) # 保存模型。
            improvement_flag = '*' # 标记提升。
            last_improve_epoch = epoch # 更新上一次提升的 epoch。
        else:
            improvement_flag = ''

        val_msg = 'Val - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f} | ' # 验证集上的评估结果。
        val_msg = val_msg.format(
            val_metrics['MRR'], val_metrics['MR'],
            val_metrics['HITS@1'], val_metrics['HITS@3'], val_metrics['HITS@10']
        ) # 格式化验证集上的评估结果。

        if improvement_flag == '*':
            test_metrics = evaluate(model, set_flag='test', kg=kg)['average'] # 在测试集上评估模型。
            val_msg += 'Test - MRR: {:5.4f} | '.format(test_metrics['MRR']) # 测试集上的评估结果。

        val_msg += improvement_flag # 将提升标记添加到验证集上的评估结果中。

        msg = 'Epoch: {:3d} | Loss: {:5.4f} | ' # 训练日志。
        msg = msg.format(epoch, np.mean(loss_list)) # 格式化训练日志。
        msg += val_msg # 将验证集上的评估结果添加到训练日志中。
        logging.info(msg) # 打印训练日志。

        # whether early stopping 是否提前停止训练。
        if epoch - last_improve_epoch > cfg.max_no_improve: # 如果上一次提升的 epoch 与当前 epoch 的差值 大于 最大的不提升的 epoch 数量。
            logging.info("Long time no improvenment, stop training...")
            break

    logging.info('Training end...')

    # evaluate train and test set 在训练集和测试集上评估模型。
    # load best model parameters 加载最好的模型参数。
    checkpoint = torch.load('checkpoint.torch') # 加载模型。
    model.load_state_dict(checkpoint['model_state_dict']) # 加载模型参数。

    logging.info('Train metrics ...')
    train_metrics = evaluate(model, 'train', kg=kg) # 在训练集上评估模型。
    train_msg = format_metrics('Train', train_metrics['head_batch'], train_metrics['tail_batch']) # 格式化训练集上的评估结果。
    logging.info(train_msg[0])
    logging.info(train_msg[1])
    logging.info(train_msg[2] + '\n')

    logging.info('Valid metrics ...')
    valid_metrics = evaluate(model, 'valid', kg=kg) # 在验证集上评估模型。
    valid_msg = format_metrics('Valid', valid_metrics['head_batch'], valid_metrics['tail_batch'])
    logging.info(valid_msg[0])
    logging.info(valid_msg[1])
    logging.info(valid_msg[2] + '\n')

    logging.info('Test metrics...')
    test_metrics = evaluate(model, 'test', kg=kg) # 在测试集上评估模型。
    test_msg = format_metrics('Test', test_metrics['head_batch'], test_metrics['tail_batch'])
    logging.info(test_msg[0])
    logging.info(test_msg[1])
    logging.info(test_msg[2] + '\n')

    logging.info('Model save path: {}'.format(os.getcwd()))


if __name__ == '__main__':
    main()
