import torchvision as tv
from dplearn.dataset import MyDataset

class Dataloader:

    def __init__(self):
        self.transform_train = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            # transforms.RandomCrop： 切割中心点的位置随机选取
            tv.transforms.RandomCrop(32, padding=4), tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            # transforms.Normalize： 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    def getTrainData(self,files):
        # filename

        train_datasets = MyDataset(transform=self.transform_train,files=files)
        return train_datasets
    def getEvalData(self,files):

        eval_datasets = MyDataset(transform=self.transform_test,files=files)
        return eval_datasets