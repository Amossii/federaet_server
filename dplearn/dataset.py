import pickle
from torch.utils.data import Dataset
import torchvision as tv
import numpy as np
import torch
from exts import *
from models import *
class MyDataset(Dataset):
    def __getitem__(self, index) :
        image = self.x_data[index]
        label = self.y_data[index]
        if self.transform:
            image = self.transform(image)
        return image, label

#files=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    def __init__(self, transform,files=['data_batch_1']) -> None:
        user=g.user
        userID=user.id
        dictlist=[]
        for filename in files:
            file=File.query.filter_by(user_id=userID,filename=filename).first()
            text=file.content
            dictlist.append(pickle.loads(text, encoding='bytes'))
        dicti = {b'data': np.empty([0, 3072],dtype=np.uint8), b'labels': []}
        for dic in dictlist:
            dicti[b'data'] = np.concatenate((dicti[b'data'], dic[b'data']), axis=0)
            dicti[b'labels'] += dic[b'labels']

        self.transform = transform
        self.x_data = dicti[b'data'].reshape(-1, 32, 32,3)
        self.y_data = torch.from_numpy(np.array(dicti[b'labels'])).type(torch.LongTensor)

    def __len__(self):
        return len(self.x_data)
def get_dataset():
    transform_train = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        # transforms.RandomCrop： 切割中心点的位置随机选取
        tv.transforms.RandomCrop(32, padding=4), tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        # transforms.Normalize： 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_datasets = MyDataset(transform=transform_train)
    eval_datasets = MyDataset(transform=transform_test)
    return train_datasets,eval_datasets

