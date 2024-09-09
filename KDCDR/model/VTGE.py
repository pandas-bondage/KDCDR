import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from KDCDR.model.GCN import GCN

from torch.autograd import Variable
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

class VTGE(nn.Module):
    def __init__(self, opt):
        super(VTGE, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number-1):
            self.encoder.append(DGCNLayer(opt))
        self.encoder.append(LastLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]
    def forward(self,ufea, vfea, UV_adj, VU_adj,UU_adj,VV_adj):

        learn_user = ufea
        learn_item = vfea
        for layer in self.encoder:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_item = F.dropout(learn_item, self.dropout, training=self.training)
            learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj,UU_adj,VV_adj)
        return learn_user, learn_item

class DGCNLayer(nn.Module):
    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc5 = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc6 = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        alpha = opt["leakey"]
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.user_union1 = nn.Linear( opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union1 = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
    def forward(self, ufea, vfea, UV_adj,VU_adj,uu,vv):
        User_ho1=self.gc3(vfea,UV_adj)
        Item_ho1=self.gc4(ufea,VU_adj)
        User_ho = self.gc1(ufea, uu)
        Item_ho = self.gc2(vfea, vv)
        User_ho =self.user_union1(torch.cat((User_ho,User_ho1),dim=1))
        Item_ho=self.item_union1(torch.cat((Item_ho,Item_ho1),dim=1))
        User = torch.cat((User_ho, ufea), dim=1)
        Item = torch.cat((Item_ho, vfea), dim=1)
        User = self.user_union(User)
        Item = self.item_union(Item)
        return self.leakyrelu(User), self.leakyrelu(Item)

class LastLayer(nn.Module):
    def __init__(self, opt):
        super(LastLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc5 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc6 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_mean = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3_logstd = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4_mean = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc4_logstd = GCN(
            nfeat=opt["hidden_dim"],
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.user_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.user_union_1 = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.user_union_2 = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_1 = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.item_union_2 = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_1 = torch.clamp(sigma_1, 0.1, 5)
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        sigma_2 = torch.clamp(sigma_2, 0.1, 5)
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        sigma = torch.clamp(sigma, 0.1, 5)
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"])
        if self.gc1.training:
            sampled_z = None
            if self.opt['cuda']:
                sampled_z = gaussian_noise.cuda() * torch.exp(sigma).cuda() + mean.cuda()
            else:
                sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def forward(self, ufea, vfea, UV_adj,VU_adj,uu,vv):
        user, user_kld = self.forward_user(ufea, vfea, UV_adj,VU_adj,uu,vv)
        item, item_kld = self.forward_item(ufea, vfea, UV_adj, VU_adj,uu,vv)
        self.kld_loss = user_kld + item_kld

        return user, item

    def forward_user(self, ufea, vfea, UV_adj, VU_adj,uu,vv):
        User_ho1 = self.gc3(vfea, UV_adj)
        User_ho = self.gc1(ufea,uu)
        User_ho = self.user_union(torch.cat((User_ho,User_ho1),dim=1))
        User_ho = self.gc5(User_ho,uu)
        User_ho_mean = self.gc3_mean(User_ho,uu)
        User_ho_logstd = self.gc3_logstd(User_ho,uu)
        # User_ho_mean = self.gc3_mean(User_ho1,uu)
        # User_ho_logstd = self.gc3_logstd(User_ho1,uu)
        User_ho_mean = self.user_union_1(torch.cat((User_ho_mean,ufea),dim=1))
        User_ho_logstd = self.user_union_2(torch.cat((User_ho_logstd,ufea),dim=1))
        user, kld_loss = self.reparameters(User_ho_mean, User_ho_logstd)
        return user, kld_loss
    def forward_item(self, ufea, vfea, UV_adj,VU_adj,uu,vv):
        Item_ho_1 = self.gc4(ufea,VU_adj)
        Item_ho = self.gc2(vfea, vv)
        Item_ho = self.item_union(torch.cat((Item_ho,Item_ho_1),dim=1))
        Item_ho = self.gc6(Item_ho, vv)
        Item_ho_mean  = self.gc4_mean(Item_ho,vv)
        Item_ho_logstd  = self.gc4_logstd(Item_ho,vv)
        # Item_ho_mean  = self.gc4_mean(Item_ho_1,vv)
        # Item_ho_logstd  = self.gc4_logstd(Item_ho_1,vv)
        Item_ho_mean = self.item_union_1(torch.cat((Item_ho_mean,vfea),dim=1))
        Item_ho_logstd = self.item_union_2(torch.cat((Item_ho_logstd,vfea),dim=1))
        item, kld_loss = self.reparameters(Item_ho_mean, Item_ho_logstd)
        return item, kld_loss
