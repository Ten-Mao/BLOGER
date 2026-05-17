from torch.utils.data import Dataset
import os
import json
import numpy as np


class SeqRecDataset(Dataset):
    def __init__(self, data_path, dataset, max_his_len, mode):
        super().__init__()
        self.data_path = data_path
        self.dataset = dataset
        self.max_his_len = max_his_len
        self.mode = mode

        with open(
            os.path.join(self.data_path, self.dataset, self.dataset + ".inter.json"),
            "r",
        ) as f:
            self.inters = json.load(f)

        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.inter_data = self._process_valid_data()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        sample = self.inter_data[index]
        return sample

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]  # remove the last two
            items = [i + 1 for i in items]  # zero for padding
            for i in range(1, len(items)):
                his = items[:i]
                if len(his) <= self.max_his_len:
                    his = his + [0] * (self.max_his_len - len(his))
                else:
                    his = his[-self.max_his_len :]
                sample = {
                    "his": np.array(his),
                    "tgt": np.array(items[i]),
                    "user": np.array(int(uid)),
                }
                inter_data.append(sample)
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-1]  # remove the last one
            items = [i + 1 for i in items]  # zero for padding
            his = items[:-1]
            if len(his) <= self.max_his_len:
                his = his + [0] * (self.max_his_len - len(his))
            else:
                his = his[-self.max_his_len :]
            sample = {
                "his": np.array(his),
                "tgt": np.array(items[-1]),
                "user": np.array(int(uid)),
            }
            inter_data.append(sample)
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid]  # remove none
            items = [i + 1 for i in items]  # zero for padding
            his = items[:-1]
            if len(his) <= self.max_his_len:
                his = his + [0] * (self.max_his_len - len(his))
            else:
                his = his[-self.max_his_len :]
            sample = {
                "his": np.array(his),
                "tgt": np.array(items[-1]),
                "user": np.array(int(uid)),
            }
            inter_data.append(sample)
        return inter_data
