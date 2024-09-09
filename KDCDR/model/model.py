import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from KDCDR.model.VTGE import VTGE
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
class StudentNet(nn.Module):
    def __init__(self,opt):
        super(StudentNet, self).__init__()
        self.opt = opt
        self.source_share_GNN = VTGE(opt)
        self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.source_item_embedding_A = nn.Embedding(opt["source_item_num"]+opt["target_item_num"], opt["feature_dim"])
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)
        self.all_item_index = torch.arange(0, self.opt["source_item_num"]+self.opt["target_item_num"], 1)
        self.user_index = torch.arange(0, self.opt["source_user_num"], 1)
        if self.opt["cuda"]:
            self.user_index = self.user_index.cuda()
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()
            self.all_item_index= self.all_item_index.cuda()
    def forward(self, A_B_G_nonenormal_UV, A_B_G_nonenormal_VU,UU=None,VV=None):
        UU,VV=A_B_G_nonenormal_UV@A_B_G_nonenormal_VU,A_B_G_nonenormal_VU@A_B_G_nonenormal_UV
        source_user_share = self.source_user_embedding(self.user_index)
        source_item_embedding =self.source_item_embedding_A(self.all_item_index)
        source_user_share, source_item_share = self.source_share_GNN(source_user_share,source_item_embedding, A_B_G_nonenormal_UV,
                                                                                  A_B_G_nonenormal_VU,UU,VV)

        return source_user_share,source_item_share

class TeacherNet(nn.Module):
    def __init__(self, opt):
        super(TeacherNet, self).__init__()
        self.opt=opt
        self.source_specific_GNN = VTGE(opt)
        self.target_specific_GNN = VTGE(opt)
        self.dropout = opt["dropout"]
        self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], opt["feature_dim"])
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], opt["feature_dim"])
        self.user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)
        if self.opt["cuda"]:
            self.user_index = self.user_index.cuda()
            self.source_user_index = self.source_user_index.cuda()
            self.target_user_index = self.target_user_index.cuda()
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()

    def source_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.source_predict_1(fea)
        out = F.relu(out)
        out = self.source_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def target_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.target_predict_1(fea)
        out = F.relu(out)
        out = self.target_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def source_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def target_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"])
        if self.share_mean.training:
            sampled_z = gaussian_noise.cuda() * torch.exp(sigma).cuda() + mean.cuda()
        else:
            sampled_z = mean.cuda()
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, (1 - self.opt["beta"]) * kld_loss

    def forward(self, source_UV, source_VU, target_UV, target_VU,source_UU_adj,source_VV_adj,target_UU_adj,target_VV_adj):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)
        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item, source_UV, source_VU,source_UU_adj,source_VV_adj)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item, target_UV, target_VU,target_UU_adj,target_VV_adj)

        return source_learn_specific_user, source_learn_specific_item, target_learn_specific_user, target_learn_specific_item

