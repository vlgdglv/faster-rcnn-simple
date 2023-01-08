import os
import math
import torch
import collections
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup

from torch.utils.data import Dataset, DataLoader


class VOCDataset(Dataset):
    def __init__(self, VOC2012_dir):

        self.idx2class = [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        self.class2idx = {
            "background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5,
            "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12,
            "horse": 13,  "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17,
            "sofa": 18, "train": 19, "tvmonitor": 20}

        image_dir = os.path.join(VOC2012_dir, "JPEGImages")
        file_names = os.listdir(image_dir)
        self.images = [os.path.join(image_dir, x) for x in file_names]

        target_dir = os.path.join(VOC2012_dir, "Annotations")
        file_names = os.listdir(target_dir)
        self.annotations = [os.path.join(target_dir, x) for x in file_names]

        assert len(self.images) == len(self.annotations)

    def __getitem__(self, index):
        assert index < self.__len__()

        image = Image.open(self.images[index]).convert("RGB")
        # Image shape: H W C
        image = np.asarray(image)
        # to [0,1]
        image = image / 255.0
        # permute axes: C H W
        image = np.transpose(image, (2, 0, 1))
        # To tensor
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        
        target = self.parse_voc_xml(self.annotations[index])

        return image, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, xml_path):
        soup = BeautifulSoup(open(xml_path), features="html.parser").annotation
        o = soup.find_all("object")
        labels = []
        bboxes = []
        for obj in o:
            obj_class = obj.find("name").string
            obj_class_idx = self.class2idx[obj_class]
            xmin = int(float(obj.find("xmin").string))
            xmax = int(float(obj.find("xmax").string))
            ymin = int(float(obj.find("ymin").string))
            ymax = int(float(obj.find("ymax").string))
            labels.append(obj_class_idx)
            bboxes.append([xmin, ymin, xmax, ymax])
        targets = {
            "boxes": torch.tensor(bboxes),
            "labels": torch.tensor(labels),
        }
        return targets


class VOCLoader:
    def __init__(self, dataset, batch_size, shuffle, val_proportion=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_len = len(dataset)
        self.train_total_iters = math.ceil(self.dataset_len / batch_size)
        self.train_current = 0
        self.val_current = 0
        self.training = True
        self.val_proportion = val_proportion
        self.set_index()
        print("train total iters:" + str(self.train_total_iters))
        print("train total iters:" + str(self.val_total_iters))

    def set_index(self):
        self.train_current = 0
        self.val_current = 0
        idx_pool = np.arange(self.dataset_len)
        if self.shuffle:
            np.random.shuffle(idx_pool)
        self.train_idx = idx_pool
        self.val_idx = None
        self.num_train = num_train = self.dataset_len
        self.num_val = 0
        if self.val_proportion is not None:
            num_val = int(self.val_proportion * self.dataset_len)
            num_train = self.dataset_len - num_val
            self.num_train = num_train
            self.num_val = num_val
            self.train_idx = idx_pool[:num_train]
            self.val_idx = idx_pool[num_train:]
            self.train_total_iters = math.ceil(num_train / self.batch_size)
            self.val_total_iters = math.ceil(num_val / self.batch_size)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.training:
            if self.train_current < self.train_total_iters:
                sidx = self.train_current * self.batch_size
                self.train_current += 1
                if self.train_current == self.train_total_iters:
                    sampled_idx = self.train_idx[sidx:]
                else:
                    sampled_idx = self.train_idx[sidx: sidx + self.batch_size]
                images = []
                targets = []
                for idx in sampled_idx:
                    image, target = self.dataset[idx]
                    images.append(image)
                    targets.append(target)
                return images, targets
            else:
                raise StopIteration
        else:
            if self.val_current < self.val_total_iters:
                sidx = self.val_current * self.batch_size
                self.val_current += 1
                if self.val_current == self.val_total_iters:
                    sampled_idx = self.val_idx[sidx:]
                else:
                    sampled_idx = self.val_idx[sidx: sidx + self.batch_size]
                images = []
                targets = []
                for idx in sampled_idx:
                    image, target = self.dataset[idx]
                    images.append(image)
                    targets.append(target)
                return images, targets
            else:
                raise StopIteration

    def train(self):
        self.set_index()
        self.training = True

    def val(self):
        self.set_index()
        self.training = False


if __name__ == "__main__":
    pass