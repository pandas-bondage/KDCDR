import  pandas as pd
import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import tqdm
import copy
class DataProcess():
    def __init__(self,filename):
        filename_A = filename
        filename = filename.split("_")
        filename_B = filename[1] + "_" + filename[0]
        self.A_train_data = "../../dataset/" + filename_A + "/train.txt"
        self.B_train_data = "../../dataset/" + filename_B + "/train.txt"
        self.A_B_train_data = "../../dataset/" + filename_A + "/A_B_train.txt"
    def process(self):
        self.user_A = set()
        self.user_B = set()
        self.item_A = set()
        self.item_B = set()
        data = []

        with codecs.open(self.A_train_data) as infile:
            for line in tqdm.tqdm(infile,smoothing=0,desc='A_train_data',mininterval=1.0):
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                data.append([int(line[0]), int(line[1]), float(line[2])])
                self.user_A.add(int(line[0]))
                self.item_A.add(int(line[1]))
        temp = list(self.item_A)
        temp.sort()
        with codecs.open(self.B_train_data) as infile:
            for line in tqdm.tqdm(infile,smoothing=0,desc='B_train_data',mininterval=1.0):
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                data.append([int(line[0]), temp[-1]+int(line[1])+1, float(line[2])])
                self.user_B.add(int(line[0]))
                self.item_B.add(temp[-1]+int(line[1])+1)
        writer = open(self.A_B_train_data,'w')
        for line in data:
            for i in range(3):
                writer.write(str(line[i]))
                if i != 2:
                    writer.write('\t')
            writer.write('\n')
        writer.close()
        return data
if __name__ == '__main__':

    Data = DataProcess('Movie_book')
    data = Data.process()


