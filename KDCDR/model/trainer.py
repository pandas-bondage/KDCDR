import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from KDCDR.model.model import TeacherNet,StudentNet
from KDCDR.utils import torch_utils
class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class CrossTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        if self.opt["model"] == "KDCDR":
            self.model = TeacherNet(opt)
            self.Student = StudentNet(opt)
        else :
            print("please input right model name!")
            exit(0)
        self.source_item_num= opt['source_item_num']
        self.criterion = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.T = self.opt['tau']
        self.w = self.opt['w']
        self.rho = self.opt['rho']
        self.softloss = nn.BCELoss()
        if opt['cuda']:
            self.model.cuda()
            self.Student.cuda()
            self.criterion.cuda()
            self.softloss.cuda()

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.optimizer_student = torch_utils.get_optimizer(opt['optim'], self.Student.parameters(), opt['lr'])
        self.epoch_rec_loss = []

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
        else:
            inputs = [Variable(b) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
        return user_index, item_index

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            user = inputs[0]
            source_pos_item = inputs[1]
            source_neg_item = inputs[2]
            target_pos_item = inputs[3]
            target_neg_item = inputs[4]
        else:
            inputs = [Variable(b) for b in batch]
            user = inputs[0]
            source_pos_item = inputs[1]
            source_neg_item = inputs[2]
            target_pos_item = inputs[3]
            target_neg_item = inputs[4]
        return user, source_pos_item, source_neg_item, target_pos_item, target_neg_item

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def source_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)
        user_feature = self.my_index_select(self.source_user, user_index)
        item_feature = self.my_index_select(self.source_item, item_index)

        user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        user_feature = user_feature.repeat(1, item_feature.size()[1], 1)


        score = self.model.source_predict_dot(user_feature, item_feature)

        #/////////////////////////////////////////////////////////////////
        user_feature_share = self.my_index_select(self.user_share, user_index)
        item_feature_share = self.my_index_select(self.item_share, item_index)

        user_feature_share = user_feature_share.view(user_feature_share.size()[0], 1, -1)
        user_feature_share = user_feature_share.repeat(1, item_feature_share.size()[1], 1)

        score_share = self.model.source_predict_dot(user_feature_share, item_feature_share)
        score = score_share+score
        # score = score
        return score.view(score.size()[0], score.size()[1])

    def target_predict(self, batch):
        user_index, item_index = self.unpack_batch_predict(batch)

        user_feature = self.my_index_select(self.target_user, user_index)
        item_feature = self.my_index_select(self.target_item, item_index)

        user_feature = user_feature.view(user_feature.size()[0], 1, -1)
        user_feature = user_feature.repeat(1, item_feature.size()[1], 1)

        score = self.model.target_predict_dot(user_feature, item_feature)
        #//////////////////////////////////////////////////////////////////////

        user_feature_share = self.my_index_select(self.user_share, user_index)
        item_feature_share = self.my_index_select(self.item_share,
                                                  item_index + torch.ones_like(item_index) * self.source_item_num)
        user_feature_share = user_feature_share.view(user_feature_share.size()[0], 1, -1)
        user_feature_share = user_feature_share.repeat(1, item_feature_share.size()[1], 1)
        score_share = self.model.target_predict_dot(user_feature_share, item_feature_share)

        score =  score_share+score
        # score =  score
        return score.view(score.size()[0], score.size()[1])

    def my_index_select_embedding(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = memory(index)
        ans = ans.view(tmp)
        return ans
    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def evaluate_embedding(self,A_B_G_nonenormal_UV,A_B_G_nonenormal_VU,UU_S,VV_S,UU_T,VV_T, source_UV, source_VU, target_UV, target_VU,source_UU_HG=None, source_VV_HG=None,AB_UU_HG=None,AB_VV_HG=None,target_UU_HG=None,target_VV_HG=None, source_adj=None, target_adj=None):
        self.source_user, self.source_item, self.target_user, self.target_item= self.model(source_UV,source_VU, target_UV,target_VU,UU_S,VV_S,UU_T,VV_T)
        self.user_share, self.item_share = self.Student(A_B_G_nonenormal_UV, A_B_G_nonenormal_VU)


    def for_bcelogit(self, x):
        y = 1 - x
        return torch.cat((x,y), dim = -1)
    def reconstruct_graph(self,batch, A_B_G_nonenormal_UV,A_B_G_nonenormal_VU,UU_S,VV_S,UU_T,VV_T, source_UV, source_VU, target_UV, target_VU,source_UU_HG=None, source_VV_HG=None,AB_UU_HG=None,AB_VV_HG=None,target_UU_HG=None,target_VV_HG=None,source_adj=None, target_adj=None, epoch = 100):
        self.model.train()
        self.Student.train()
        self.optimizer.zero_grad()
        self.optimizer_student.zero_grad()
        user, source_pos_item, source_neg_item, target_pos_item, target_neg_item = self.unpack_batch(batch)
        self.source_user, self.source_item, self.target_user, self.target_item = self.model(source_UV, source_VU,
                                                                                            target_UV, target_VU, UU_S,
                                                                                            VV_S, UU_T, VV_T)

        self.user_share,self.item_share =self.Student(A_B_G_nonenormal_UV,A_B_G_nonenormal_VU)

        source_user_feature = self.my_index_select(self.source_user, user)
        source_item_pos_feature = self.my_index_select(self.source_item, source_pos_item)
        source_item_neg_feature = self.my_index_select(self.source_item, source_neg_item)
        target_user_feature = self.my_index_select(self.target_user, user)
        target_item_pos_feature = self.my_index_select(self.target_item, target_pos_item)
        target_item_neg_feature = self.my_index_select(self.target_item, target_neg_item)

        user_feature_share = self.my_index_select(self.user_share, user)

        source_item_pos_feature_share = self.my_index_select(self.item_share,source_pos_item)
        source_item_neg_feature_share = self.my_index_select(self.item_share,source_neg_item)
        target_item_pos_feature_share = self.my_index_select(self.item_share,target_pos_item+torch.ones_like(target_pos_item)*self.source_item_num)
        target_item_neg_feature_share = self.my_index_select(self.item_share,target_neg_item+torch.ones_like(target_neg_item)*self.source_item_num)

        pos_source_score = self.model.source_predict_dot(source_user_feature, source_item_pos_feature).reshape([-1,1])
        neg_source_score = self.model.source_predict_dot(source_user_feature, source_item_neg_feature).reshape([-1,1])
        pos_target_score = self.model.target_predict_dot(target_user_feature, target_item_pos_feature).reshape([-1,1])
        neg_target_score = self.model.target_predict_dot(target_user_feature, target_item_neg_feature).reshape([-1,1])


        pos_source_score_student = self.model.source_predict_dot(user_feature_share, source_item_pos_feature_share).reshape([-1,1])
        neg_source_score_student = self.model.source_predict_dot(user_feature_share, source_item_neg_feature_share).reshape([-1,1])
        pos_target_score_student = self.model.target_predict_dot(user_feature_share, target_item_pos_feature_share).reshape([-1,1])
        neg_target_score_student = self.model.target_predict_dot(user_feature_share, target_item_neg_feature_share).reshape([-1,1])


        pos_labels, neg_labels = torch.ones(pos_source_score.size()), torch.zeros(
            neg_source_score.size())
        if self.opt["cuda"]:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        KDLOSS_A = self.softloss(torch.sigmoid(pos_source_score_student / self.T),
                      torch.sigmoid(pos_source_score.detach() / self.T))
        KDLOSS_B=self.softloss(torch.sigmoid(neg_source_score_student/self.T),torch.sigmoid(neg_source_score.detach()/self.T))
        KDLOSS_C= self.softloss(torch.sigmoid(pos_target_score_student / self.T),
                      torch.sigmoid(pos_target_score.detach() / self.T))
        KDLOSS_D=self.softloss(torch.sigmoid(neg_target_score_student/self.T),torch.sigmoid(neg_target_score.detach()/self.T))

        #softtarget
        self.KDLOSS=KDLOSS_A+KDLOSS_B+KDLOSS_C+KDLOSS_D
        a1= torch.sigmoid(pos_source_score)
        a2=torch.sigmoid(neg_source_score)
        b1 = torch.sigmoid(pos_target_score)
        b2 =torch.sigmoid(neg_target_score)
        a1 = F.binary_cross_entropy(a1, pos_labels,reduction='none')
        a2 = F.binary_cross_entropy(a2, neg_labels,reduction='none')
        b1 = F.binary_cross_entropy(b1, pos_labels,reduction='none')
        b2 = F.binary_cross_entropy(b2, neg_labels,reduction='none')

        loss1 = self.w*torch.exp(a1/self.rho)
        loss2 =self.w*torch.exp(a2/self.rho)
        loss3 =self.w*torch.exp(b1/self.rho)
        loss4 =self.w*torch.exp(b2/self.rho)
        self.hardloss = loss1*self.criterion(pos_source_score_student, pos_labels) + \
               loss2*self.criterion(neg_source_score_student, neg_labels) + \
               loss3*self.criterion(pos_target_score_student, pos_labels) + \
               loss4*self.criterion(neg_target_score_student, neg_labels)

        loss = self.criterion(pos_source_score, pos_labels) + \
               self.criterion(neg_source_score, neg_labels) + \
               self.criterion(pos_target_score, pos_labels) + \
               self.criterion(neg_target_score, neg_labels)+self.model.source_specific_GNN.encoder[-1].kld_loss + \
               self.model.target_specific_GNN.encoder[-1].kld_loss\
               +self.KDLOSS+ \
                self.hardloss.mean()+self.Student.source_share_GNN.encoder[-1].kld_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_student.step()
        return (loss).item()