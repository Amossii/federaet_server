import pickle

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
from torchvision import models
import torch.nn as nn



class Server:
    # 定义构造函数
    def __init__(self):
      self.global_model=None
    def myInit(self,conf,eval_dataset,model_init):
        # 导入配置文件
        self.conf = conf
        # 根据配置获取模型文件
        self.global_model=None
        if model_init==None:
            self.global_model = models.get_model(self.conf["model_name"])
            self.global_model.fc = nn.Linear(512, 10)
        else:
            self.global_model = pickle.loads(model_init)
        # 生成一个测试集合加载器

        #this is a change

        # conv1 = self.global_model.conv1
        #
        # # 修改第一个卷积层的参数以保留原始输入图像的尺寸
        # # 假设输入图像的尺寸是 [3, 32, 32]
        # # 计算填充大小，使得特征图的高度和宽度与输入图像相同
        # # 注意：这里假设使用了卷积核大小为3x3，步长为1的卷积操作
        # padding = (1, 1)  # 由于卷积核大小为3x3，填充大小为1
        # conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=padding, bias=False)
        #
        # # 将修改后的第一个卷积层设置回ResNet-18模型中
        # self.global_model.conv1 = conv1

        #upon is a change
        self.eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            # 设置单个批次大小32
            batch_size=self.conf["batch_size"],
            # 打乱数据集
            shuffle=True)

    # 全局聚合模型
    # weight_accumulator 存储了每一个客户端的上传参数变化值/差值
    def model_aggregate(self, weight_accumulator):
      # 遍历服务器的全局模型
      for name, data in self.global_model.state_dict().items():
        # 更新每一层乘上学习率
        update_per_layer = weight_accumulator[name] * self.conf["lambda"]
        # 累加和
        if data.type() != update_per_layer.type():
            # 因为update_per_layer的type是floatTensor，所以将起转换为模型的LongTensor（有一定的精度损失）
            data.add_(update_per_layer.to(torch.int64))
        else:
            data.add_(update_per_layer)

            # 评估函数
    def model_eval(self):
        self.global_model.eval()    # 开启模型评估模式（不修改参数）
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        # 遍历评估数据集合
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            # 获取所有的样本总量大小
            dataset_size += data.size()[0]
            output = self.global_model(data)
            # 聚合所有的损失 cross_entropy交叉熵函数计算损失
            total_loss += torch.nn.functional.cross_entropy(
                output,
                target,
                reduction='sum'
            ).item()
            # 获取最大的对数概率的索引值， 即在所有预测结果中选择可能性最大的作为最终的分类结果
            pred = output.data.max(1)[1]
            # 统计预测结果与真实标签target的匹配总个数
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))    # 计算准确率
        total_1 = total_loss / dataset_size                     # 计算损失值
        self.global_model.train()
        return acc, total_1
    def getModel(self):
        return pickle.dumps(self.global_model)
    def model_predict(self,predict_dataset):
        self.global_model.eval()
        for batch_id, batch in enumerate(torch.utils.data.DataLoader(predict_dataset)):
            data, target = batch
            output = self.global_model(data)
            pred = output.data.max(1)[1]
        self.global_model.train()
        return pred.item(),target.item()

    def heat(self):
        from torchcam.methods import GradCAM
        target_layer = self.global_model.layer4[-1]  # 选择目标层
        cam_extractor = GradCAM(self.global_model, target_layer)
        img_path = 'D:\Python\myapp\\file\img\Figure_2.jpg'
        img_pil = Image.open(img_path)

        from torchvision import transforms
        # # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                 mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                                             ])
        print(type(img_pil))
        input_tensor = test_transform(img_pil).unsqueeze(0)  # 预处理
        pred_logits = self.global_model(input_tensor)
        # # topk()方法用于返回输入数据中特定维度上的前k个最大的元素
        pred_top1 = torch.topk(pred_logits, 1)
        # # pred_id 为图片所属分类对应的索引号，分类和索引号存储在imagenet_class_index.csv
        pred_id = pred_top1[1].detach().cpu().numpy().squeeze().item()
        print(pred_id)

        activation_map = cam_extractor(pred_id, pred_logits)
        activation_map = activation_map[0][0].detach().cpu().numpy()
        #
        # # 可视化
        from torchcam.utils import overlay_mask
        #
        # # overlay_mask 用于构建透明的叠加层
        # # fromarray 实现array到image的转换
        result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)
        import matplotlib.pyplot as plt
        plt.imshow(result)
        plt.show()