import torch
import torch.nn as nn
import utils
import torch.nn.functional as F


class ConvE(nn.Module):
    def __init__(self, h_dim, out_channels, ker_sz):
        super().__init__()
        cfg = utils.get_global_config()
        self.cfg = cfg
        dataset = cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']

        self.bn0 = torch.nn.BatchNorm2d(1) # 1
        self.bn1 = torch.nn.BatchNorm2d(out_channels) # 200 / 250
        self.bn2 = torch.nn.BatchNorm1d(h_dim) # 450 / 200

        self.conv_drop = torch.nn.Dropout(cfg.conv_drop)
        self.fc_drop = torch.nn.Dropout(cfg.fc_drop)
        self.k_h = cfg.k_h # 15 / 10
        self.k_w = cfg.k_w # 30 / 20
        assert self.k_h * self.k_w == h_dim
        self.conv = torch.nn.Conv2d(1, out_channels=out_channels, stride=1, padding=0,
                                    kernel_size=ker_sz, bias=False) # [200,1,8,8] / [250,1,7,7]
        flat_sz_h = int(2 * self.k_h) - ker_sz + 1 # 30 - 8 + 1 = 23 / 20 - 7 + 1 = 14
        flat_sz_w = self.k_w - ker_sz + 1 # 30 - 8 + 1 = 23 / 20 - 7 + 1 = 14
        self.flat_sz = flat_sz_h * flat_sz_w * out_channels # 23 * 23 * 200 = 105800 / / 14 * 14 *250=490000
        self.fc = torch.nn.Linear(self.flat_sz, h_dim, bias=False) # [450,105800] / [200,490000]
        self.ent_drop = nn.Dropout(cfg.ent_drop_pred)

    def forward(self, head, rel, all_ent):
        # head (bs, h_dim), rel (bs, h_dim)
        # concatenate and reshape to 2D 解释：将head和rel拼接在一起，然后转置，最后reshape成（-1,1,2*15,30）/（-1,1,2*10,20）
        c_head = head.view(-1, 1, head.shape[-1]) # (bs, 1, h_dim)
        c_rel = rel.view(-1, 1, rel.shape[-1]) # (bs, 1, h_dim)
        c_emb = torch.cat([c_head, c_rel], 1) # (bs, 2, h_dim)
        c_emb = torch.transpose(c_emb, 2, 1).reshape((-1, 1, 2 * self.k_h, self.k_w)) # (bs, 1, 2 * k_h, k_w)

        x = self.bn0(c_emb) # 对输入数据c_emb执行batch normalization操作。这个操作可以使得输入数据的均值和方差在整个batch内保持不变，从而加速网络的训练过程。
        x = self.conv(x)  # (bs, out_channels, out_h, out_w) # 对batch normalization操作后的数据进行卷积操作。这个操作通过对输入数据的不同部分执行卷积操作，可以提取出不同的特征。
        x = self.bn1(x) # 对卷积操作后的数据执行batch normalization操作。
        x = F.relu(x) # 对batch normalization后的数据执行ReLU激活函数操作。ReLU激活函数可以将所有负值置为0，从而增加网络的非线性拟合能力。
        x = self.conv_drop(x) # 对ReLU激活后的数据进行dropout操作。dropout操作可以随机地丢弃一些神经元，从而减少过拟合的风险。
        x = x.view(-1, self.flat_sz)  # (bs, out_channels * out_h * out_w) # 对dropout后的数据进行reshape操作，将数据展平成一维向量。
        x = self.fc(x)  # (bs, h_dim) # 对展平后的数据进行全连接操作，从而将数据映射到一个低维空间中。
        x = self.bn2(x) # 对全连接后的数据执行batch normalization操作。
        x = F.relu(x) # 对batch normalization后的数据执行ReLU激活函数操作。
        x = self.fc_drop(x)  # (bs, h_dim) # 对ReLU激活后的数据进行dropout操作。
        # inference all entities 将所有的实体和x做矩阵乘法，然后sigmoid
        # all_ent: (n_ent, h_dim)
        all_ent = self.ent_drop(all_ent) # 对所有实体的嵌入向量执行dropout操作。
        x = torch.mm(x, all_ent.transpose(1, 0))  # (bs, n_ent) #  将处理后的数据与所有实体的嵌入向量进行矩阵乘法操作。
        x = torch.sigmoid(x) # 对乘法结果进行sigmoid操作，得到最终的输出结果。
        return x
