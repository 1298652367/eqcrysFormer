import os
import json
import zipfile
import requests
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata
import dgl

from .utils import get_db_info

#将数据集分割为训练集、验证集和测试集
def split_train_val_test(
    total_size,
    ratio_train_val_test=None,#用于指定训练、验证、测试集的比例
    n_train_val_test=None,#用于指定训练、验证、测试集的比例，与上述互斥，是整数型
    keep_order=False,#是否保持原始数据集中样本的顺序
    split_seed=7,#一个整数，用于设置随机数生成器的种子，以确保可复现的随机分割结果
):
    if ratio_train_val_test:
        assert len(ratio_train_val_test) == 3
        ratio_train, ratio_val, ratio_test = ratio_train_val_test
        n_train = int(ratio_train * total_size)
        n_val = int(ratio_val * total_size) if ratio_val else None
        n_test = int(ratio_test * total_size) if ratio_test else None
    elif n_train_val_test:
        assert len(n_train_val_test) == 3
        n_train, n_val, n_test = n_train_val_test
    else:
        raise ValueError("Please Specify the dataset division.")

    ids = list(np.arange(total_size))#ids是一个包含从0到total_size-1的整数的列表，表示数据集的索引
    if not keep_order:
        random.seed(split_seed)
        random.shuffle(ids)

    train_idx = ids[:n_train]
    val_idx = ids[n_train: n_train+n_val] if n_val else None
    test_idx = ids[-n_test:] if n_test else None
    return train_idx, val_idx, test_idx


def get_dataset(
    dataset, #数据集的名称，用于指定要获取的数据集
    data_dir, #数据集存储的目录路径。
    ratio_train_val_test=[0.6, 0.2, 0.2], 
    n_train_val_test=None,
    transforms=None,#用于对获取的数据集进行预处理操作
    graph=True, #指示是否使用图形数据集
    line_graph=True, #指示是否使用线图（line graph）表示图形数据集。
):
    database_info = get_db_info()
    assert dataset in database_info
    url, filename, message, _ = database_info[dataset]
    print(message)
    print("Data dir:", str(data_dir))

    if not os.path.isfile(os.path.join(data_dir, filename+ '.zip')):
        print("Downloading the dataset ...")
        zipfilename = filename + '.zip'
        r = requests.get(url, stream=True)
        with open(os.path.join(data_dir, zipfilename), 'wb') as f:
            for data in r.iter_content(chunk_size=1024):
                f.write(data)
        print("Extracting the zipfile ...")
        assert zipfile.is_zipfile(os.path.join(data_dir, zipfilename))
        zf = zipfile.ZipFile(os.path.join(data_dir, zipfilename))
        for name in zf.namelist():
            zf.extract(name, data_dir)
        zf.close()

    assert os.path.isfile(os.path.join(data_dir, filename))
    with open(os.path.join(data_dir, filename)) as f:
        data = json.load(f)
    print("Loading completed.")

    train_idx, val_idx, test_idx = split_train_val_test(
                                        total_size=len(data),
                                        ratio_train_val_test=ratio_train_val_test,
                                        n_train_val_test=n_train_val_test)
    train_data = [data[idx] for idx in train_idx]
    val_data = [data[idx] for idx in val_idx] if val_idx else None
    test_data = [data[idx] for idx in test_idx] if val_idx else None

    if isinstance(transforms, list): #判断transforms是否为列表数据类型
        assert len(transforms) <= 3 #确保 transforms 列表的长度不超过 3，避免超出范围。
        if len(transforms) == 3: #如果 transforms 的长度为 3，将列表中的三个元素分别赋值给 train_transform、val_transform 和 test_transform。
            train_transform, val_transform, test_transform = transforms
        elif len(transforms) == 2: #如果 transforms 的长度为 2，将列表中的第一个元素赋值给 train_transform，并将列表中的最后一个元素同时赋值给 val_transform 和 test_transform。
            train_transform = transforms[0]
            val_transform = test_transform = transforms[-1]
        else:
            train_transform = val_transform = test_transform = transforms
    else:
        train_transform = val_transform = test_transform = transforms

    if graph:
        train_dataset = CrystalGraphDataset(data=train_data, transforms=train_transform, line_graph=line_graph)
        val_dataset = CrystalGraphDataset(data=val_data, transforms=val_transform, line_graph=line_graph) if val_data else None
        test_dataset = CrystalGraphDataset(data=test_data, transforms=test_transform, line_graph=line_graph) if test_data else None
    else:
        train_dataset = CrystalDataset(data=train_data, transforms=train_transform)
        val_dataset = CrystalDataset(data=val_data, transforms=val_transform) if val_data else None
        test_dataset = CrystalDataset(data=test_data, transforms=test_transform) if test_data else None

    return train_dataset, val_dataset, test_dataset


class CrystalDataset(Dataset):
    def __init__(self, data, transforms):
        super().__init__()
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index): #如何获取数据集中指定索引位置的样本
        info = self.data[index]
        crystal = dict()#创建一个字典crystal，用于存储样本的相关信息
        crystal['info'] = info #样本的原始信息将存储在crystal字典中的'info'键下
        #将info字典中的'atoms'键的值转换为Atoms对象，并将其赋值给crystal字典中的'structure'键。
        crystal['structure'] = Atoms.from_dict(info['atoms'])
        #对crystal字典进行预处理或转换操作，并将返回的结果赋值给inputs和targets变量
        inputs, targets = self.transforms(crystal)
        return inputs, targets

    def __len__(self):
        return len(self.data)


class CrystalGraphDataset(Dataset):
    def __init__(self, data, transforms, cutoff=8, max_neighbors=12, line_graph=False):
        self.data = data
        self.max_neighbors = max_neighbors
        self.cutoff = cutoff
        self.transforms = transforms
        self.line_graph = line_graph

    def __getitem__(self, index):
        info = self.data[index]
        crystal = dict()
        crystal['info'] = info
        crystal['structure'] = Atoms.from_dict(info['atoms'])
        crystal['graph'] = self.build_graph(crystal['structure'])
        if self.line_graph:
            crystal['line_graph'] = self.build_line_graph(crystal['graph'])
        inputs, targets = self.transforms(crystal)
        return inputs, targets

    def __len__(self):
        return len(self.data)
    
    def build_graph(self, atoms):
        edges = nearest_neighbor_edges(
                atoms=atoms,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                use_canonize=True,
            )
        u, v, r = build_undirected_edgedata(atoms, edges)
        g = dgl.graph((u, v))
        g.edata['offset'] = r
        return g

    def build_line_graph(self, g):
        lg = g.line_graph(shared=True)
        return lg

    def collect(self):
        def collect_graph(samples):#接收一个样本列表samples作为输入，其中每个样本由一个图对象和一个目标值组成。
            graphs, targets = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            return batched_graph, torch.stack(targets)
        def collect_line_graph(samples):
            graphs, labels = map(list, zip(*samples))
            batched_graph = dgl.batch([g[0] for g in graphs])
            batched_line_graph = dgl.batch([g[1] for g in graphs])
            return [batched_graph, batched_line_graph], torch.stack(labels)

        if self.line_graph:
            return collect_line_graph
        return collect_graph