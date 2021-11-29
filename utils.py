from PIL import Image
import numpy as np
import os
import argparse
import glob
import time
import pickle
import pyflann
from tqdm import tqdm

#该方法旨在使用训练有素的线框编码器将 UI设计嵌入潜在向量空间，并支持基于线框的kNNUI设计搜索
#返回前十个搜索结果索引值

# 存储数据
def save_data(data, save_root):
	pickle.dump(data[0], open(os.path.join(save_root,'sfile_list.pk'),'wb'))
	pickle.dump(data[1], open(os.path.join(save_root,'wfile_list.pk'),'wb'))
	np.save(os.path.join(save_root,'torchlist_nn.npy'),data[2])
		
#从字节流中上传数据
def load_data_from_pickle(load_root):
	sfile_list = pickle.load(open(os.path.join(load_root,'sfile_list.pk'),'rb'))
	wfile_list = pickle.load(open(os.path.join(load_root,'wfile_list.pk'),'rb'))
	torchlist = np.load(os.path.join(load_root,'torchlist_nn.npy'))

	return sfile_list, wfile_list, torchlist


#返回前十个结果的索引号
def findtTopMinimalIndex(num, list):
    list_sort = []
    list_minimal = []
    for i in range(len(list)):
        if i == 0:
            list_sort.append(0)
        else:
            count = 0
            for j in range(i):
                if list[i] <= list[j]:
                    list_sort[j] += 1
                else:
                    count += 1
            list_sort.append(count)
    for i in range(num):
        list_minimal.append(list_sort.index(i))
    return list_minimal