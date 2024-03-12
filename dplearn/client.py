import torch
import pickle
class Client:
    # 构造函数
    def __init__(self, conf, model, train_dataset,eval_dataset, id=1):
        # 配置文件
        self.conf = conf
        # 客户端本地模型(一般由服务器传输)
        self.local_model = model
        # 客户端ID
        self.client_id = id
        # 客户端本地数据集
        self.train_dataset = train_dataset
        self.eval_dataset= eval_dataset
        # 按ID对训练集合的拆分
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        indices = all_range[id * data_len: (id + 1) * data_len]
        # 生成一个数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            # 制定父集合
            self.train_dataset,
            # batch_size每个batch加载多少个样本(默认: 1)
            batch_size=conf["batch_size"],
            # 指定子集合
            # sampler定义从数据集中提取样本的策略
            # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices)
        )
        self.eval_loader = torch.utils.data.DataLoader(
            self.eval_dataset,
            # 设置单个批次大小32
            batch_size=self.conf["batch_size"],
            # 打乱数据集
            shuffle=True)

    # 模型本地训练函数
    def local_train(self, model):
        # 整体的过程：拉取服务器的模型，通过部分本地数据集训练得到
        for name, param in model.state_dict().items():
            # 客户端首先用服务器端下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())
        # 定义最优化函数器用于本地模型训练
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])

        # 本地训练模型
        self.local_model.train()  # 设置开启模型训练（可以更改参数）
        # 开始训练模型
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                # 梯度
                optimizer.zero_grad()
                # 训练预测
                output = self.local_model(data)
                # 计算损失函数 cross_entropy交叉熵误差
                loss = torch.nn.functional.cross_entropy(output, target)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()
            print("Epoch %d done" % e)
        # 创建差值字典（结构与模型参数同规格），用于记录差值
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            # 计算训练后与训练前的差值
            diff[name] = (data - model.state_dict()[name])
        print("Client %d local train done" % self.client_id)
        # 客户端返回差值
        return diff
    def getModel(self):
        return self.local_model
    def modelSave(self):
        with open('file/models/model.pkl', 'wb') as f:
            pickle.dump(self.local_model, f)
    def modelLoad(self,content):
        self.local_model = pickle.loads(content)
    def model_eval(self):
        self.local_model.eval()  # 开启模型评估模式（不修改参数）
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        # 遍历评估数据集合
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            # 获取所有的样本总量大小
            dataset_size += data.size()[0]
            # 加载到模型中训练
            output = self.local_model(data)
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
        acc = 100.0 * (float(correct) / float(dataset_size))  # 计算准确率
        total_1 = total_loss / dataset_size  # 计算损失值
        return acc, total_1