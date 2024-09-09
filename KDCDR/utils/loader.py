"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import codecs

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, evaluation,label=None):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation
        self.label= label
        # ************* source data *****************
        source_train_data = "../dataset/" + filename + "/train.txt"
        source_test_data = "../dataset/" + filename + "/test.txt"
        self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_test_data, self.source_user, self.source_item ,self.all_dict_A= self.read_data(source_train_data, source_test_data)

        opt["source_user_num"] = len(self.source_user)
        opt["source_item_num"] = len(self.source_item)
        # ************* target data *****************
        filename = filename.split("_")
        filename = filename[1] + "_" + filename[0]
        target_train_data = "../dataset/" + filename + "/train.txt"
        target_test_data = "../dataset/" + filename + "/test.txt"
        self.target_ma_set, self.target_ma_list, self.target_train_data, self.target_test_data, self.target_user, self.target_item ,self.all_dict_B= self.read_data(target_train_data, target_test_data)
        opt["target_user_num"] = len(self.target_user)
        opt["target_item_num"] = len(self.target_item)


        a_0_15, a_16_30, a_31_45, a_46 = [], [], [], []
        b_0_10, b_11_20, b_21_30, b_31 = [], [], [], []

        for key, value in self.all_dict_A.items():
            if len(value) <= 10:
                a_0_15.append(key)
            if len(value) > 10 and len(value) <= 20:
                a_16_30.append(key)
            if len(value) > 20 and len(value) <= 25:
                a_31_45.append(key)
            if len(value) > 25:
                a_46.append(key)
        for key, value in self.all_dict_B.items():
            if len(value) <= 35:
                b_0_10.append(key)
            if len(value) > 35 and len(value) <= 40:
                b_11_20.append(key)
            if len(value) > 40 and len(value) <= 55:
                b_21_30.append(key)
            if len(value) > 55:
                b_31.append(key)
        self.A1 ,self.A2,self.A3,self.A4= a_0_15, a_16_30, a_31_45, a_46
        self.B1, self.B2, self.B3, self.B4 =  b_0_10, b_11_20, b_21_30, b_31
        self.A1B1,self.A1B2,self.A1B3,self.A1B4 = set( self.A1)&set(self.B1) ,set( self.A1)&set(self.B2),set( self.A1)&set(self.B3),set( self.A1)&set(self.B4)
        self.B1A1,self.B1A2,self.B1A3,self.B1A4 =  set( self.B1)&set(self.A1), set( self.B1)&set(self.A2), set( self.B1)&set(self.A3) ,set( self.B1)&set(self.A4)
        opt["rate"] = self.rate()

        assert opt["source_user_num"] == opt["target_user_num"]
        if evaluation == -1:
            data = self.preprocess()
        else :
            data = self.preprocess_for_predict()
        # shuffle for training
        if evaluation == -1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)
        print(len(data))
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def read_data(self, train_file, test_file):
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user = {}
            item = {}
            ma = {}
            ma_list = {}
            all_list = {}
            for line in infile:
                line=line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if user.get(line[0], "zxczxc") is "zxczxc":
                    user[line[0]] = len(user)
                if item.get(line[1], "zxczxc") is "zxczxc":
                    item[line[1]] = len(item)
                line[0] = user[line[0]]
                line[1] = item[line[1]]
                train_data.append([line[0],line[1]])
                if line[0] not in ma:
                    ma[line[0]] = set()
                    ma_list[line[0]] = []
                    all_list[line[0]]=[]
                ma[line[0]].add(line[1])
                ma_list[line[0]].append(line[1])
                all_list[line[0]].append(line[1])

        with codecs.open(test_file,"r",encoding="utf-8") as infile:
            test_data=[]
            for line in infile:
                line=line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if user.get(line[0], "zxczxc") is "zxczxc":
                    continue
                if item.get(line[1], "zxczxc") is "zxczxc":
                    continue
                line[0] = user[line[0]]
                line[1] = item[line[1]]
                all_list[line[0]].append(line[1])
                ret = [line[1]]
                for i in range(999):
                    while True:
                        rand = random.randint(0, len(item)-1)
                        if rand in ma[line[0]]:
                            continue
                        ret.append(rand)
                        break
                test_data.append([line[0],ret])

        return ma, ma_list, train_data, test_data, user, item ,all_list

    def rate(self):
        ret = []
        for i in range(len(self.source_ma_set)):
            ret = len(self.source_ma_set[i]) / (len(self.source_ma_set[i]) + len(self.target_ma_set[i]))
        return ret

    def preprocess_for_predict(self):
        processed=[]
        if self.eval == 1:
            for d in self.source_test_data:
                if self.label==None:
                    processed.append([d[0],d[1]])
                if self.label==1 and d[0] in self.A1:# user, item_list(pos in the first node)
                    processed.append([d[0],d[1]])
                if self.label==2 and d[0] in self.A2:
                    processed.append([d[0], d[1]])
                if self.label==3 and d[0] in self.A3:
                    processed.append([d[0], d[1]])
                if self.label==4 and d[0] in self.A4:
                    processed.append([d[0], d[1]])

        else :
            for d in self.target_test_data:
                if self.label==None:
                    processed.append([d[0],d[1]])
                if self.label==1 and d[0] in self.B1:# user, item_list(pos in the first node)
                    processed.append([d[0],d[1]])
                if self.label==2 and d[0] in self.B2:
                    processed.append([d[0], d[1]])
                if self.label==3 and d[0] in self.B3:
                    processed.append([d[0], d[1]])
                if self.label==4 and d[0] in self.B4:
                    processed.append([d[0], d[1]])
        # processed=[processed_A1,processed_A2,processed_A3,processed_A4]
        return processed
    def preprocess(self):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in self.source_train_data:
            d = [d[1], d[0]]
            processed.append(d + [-1])
        for d in self.target_train_data:
            processed.append([-1] + d)
        return processed

    def find_pos(self,ma_list, user):
        rand = random.randint(0, 1000000)
        rand %= len(ma_list[user])
        return ma_list[user][rand]

    def find_neg(self, ma_set, user, type):
        # neg=[]
        n = 500
        while n:
            n -= 1
            rand = random.randint(0, self.opt[type] - 1)
            if rand not in ma_set[user]:
                return rand
            # if len(neg)==9:
            #     return neg
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval!=-1 :
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))

        else :
            source_neg_tmp = []
            target_neg_tmp = []
            source_pos_tmp = []
            target_pos_tmp = []
            user = []
            for b in batch:
                if b[0] == -1:
                    source_pos_tmp.append(self.find_pos(self.source_ma_list, b[1]))
                    target_pos_tmp.append(b[2])
                else:
                    source_pos_tmp.append(b[0])
                    target_pos_tmp.append(self.find_pos(self.target_ma_list, b[1]))
                source_neg_tmp.append(self.find_neg(self.source_ma_set, b[1], "source_item_num"))
                target_neg_tmp.append(self.find_neg(self.target_ma_set, b[1], "target_item_num"))
                user.append(b[1])
            return (torch.LongTensor(user), torch.LongTensor(source_pos_tmp), torch.LongTensor(source_neg_tmp), torch.LongTensor(target_pos_tmp), torch.LongTensor(target_neg_tmp))
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
