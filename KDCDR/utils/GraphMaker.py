import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import copy
#生成边集矩阵
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphMaker(object):
    def __init__(self, opt, filename):
        self.opt = opt
        self.user = set()
        self.item = set()
        user_map = {}
        item_map = {}
        data=[]
        with codecs.open(filename) as infile:
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if user_map.get(line[0], "zxczxc") is "zxczxc":
                    user_map[line[0]] = len(user_map)
                if item_map.get(line[1], "zxczxc") is "zxczxc":
                    item_map[line[1]] = len(item_map)
                line[0] = user_map[line[0]]
                line[1] = item_map[line[1]]
                data.append((int(line[0]),int(line[1]),float(line[2])))
                self.user.add(int(line[0]))
                self.item.add(int(line[1]))

        opt["number_user"] = len(self.user)
        opt["number_item"] = len(self.item)

        print("number_user", len(self.user))
        print("number_item", len(self.item))
        self.user_map = user_map
        self.item_map = item_map
        self.raw_data = data
        self.UV,self.VU, self.adj,self.UV_adj_ori,self.VU_adj_ori = self.preprocess( data, opt)

    def heropgraph(self,uv,vu):
        heoedge=[]
        for i in range(2):
            # tempeedge=None
            temp=None
            if i==0:
                tempeedge = uv
            else:
                temp = vu @vu.T
                for j in range(i-1):
                    temp = temp@vu @ vu.T
                temp = torch.min(torch.tensor(temp.toarray()),torch.ones_like(torch.tensor(temp.toarray())))
                temp = sp.csr_matrix(temp.numpy())
                tempeedge=uv@temp
            heoedge.append(tempeedge)
        horizontal_stack = sp.hstack(heoedge)
        heoedge = self.HGnoraml(horizontal_stack)
        heoedge = sparse_mx_to_torch_sparse_tensor(heoedge)
        return  heoedge
    def HGnoraml(self,mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1/2).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        #-----------------------
        cloum=np.array(mx.sum(axis=0))
        c_inv = np.power(cloum,-1).flatten()
        c_inv[np.isinf((c_inv))]=0.
        c_mat_inv =sp.diags(c_inv)
        #----------------------
        mx_temp = r_mat_inv.dot(mx)
        mx_temp = mx_temp.dot(c_mat_inv)
        mx_temp = mx_temp.dot(mx.T)
        mx_temp = mx_temp.dot(r_mat_inv)
        return mx_temp



    def preprocess(self,data,opt):
        UV_edges = []
        VU_edges = []
        all_edges = []
        real_adj = {}

        user_real_dict = {}
        item_real_dict = {}
        for edge in data:
            UV_edges.append([edge[0],edge[1]])
            if edge[0] not in user_real_dict.keys():
                user_real_dict[edge[0]] = set()
            user_real_dict[edge[0]].add(edge[1])

            VU_edges.append([edge[1], edge[0]])
            if edge[1] not in item_real_dict.keys():
                item_real_dict[edge[1]] = set()
            item_real_dict[edge[1]].add(edge[0])

            all_edges.append([edge[0],edge[1] + opt["number_user"]])
            all_edges.append([edge[1] + opt["number_user"], edge[0]])
            if edge[0] not in real_adj :
                real_adj[edge[0]] = {}
            real_adj[edge[0]][edge[1]] = 1

        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)
        UV_adj_sp = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(opt["number_user"], opt["number_item"]),
                               dtype=np.float32)
        VU_adj_sp = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(opt["number_item"], opt["number_user"]),
                               dtype=np.float32)
        all_adj_sp = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),shape=(opt["number_item"]+opt["number_user"], opt["number_item"]+opt["number_user"]),dtype=np.float32)
        UV_adj = normalize(UV_adj_sp)
        VU_adj = normalize(VU_adj_sp)
        all_adj = normalize(all_adj_sp)
        UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)
        print("real graph loaded!")
        return UV_adj, VU_adj, all_adj,UV_adj_sp,VU_adj_sp

