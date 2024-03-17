from flask import Blueprint
from dplearn import dataset
import json
from dplearn.server import *
from dplearn.client import *
import random
from dplearn.dataloader import Dataloader
from dplearn.utils.globalConf import conf
bp=Blueprint('test',__name__,url_prefix='/test')

a=[100]
@bp.route('/')
def test():
    dataloader=Dataloader()
    train_datasets=dataloader.getTrainData(['data_batch_1'])
    eval_datasets=dataloader.getEvalData(['test_batch'])
    with open("./dplearn/conf.json", 'r') as f:
        conf = json.load(f)

    server = Server()
    server.myInit(conf, eval_datasets)
    # 客户端列表
    clients = []

    # 添加10个客户端到列表
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    print("\n\n")

    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练都是从clients列表中随机采样k个进行本轮训练
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        for c in candidates:
            print(c.client_id)

        # 权重累计
        weight_accumulator = {}

        # 初始化空模型参数weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)

        # 遍历客户端，每个客户端本地训练模型
        for c in candidates:
            diff = c.local_train(server.global_model)

            # 根据客户端的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        # 模型参数聚合
        server.model_aggregate(weight_accumulator)

        # 模型评估
        acc, loss = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
    return "This is a test!"

@bp.route('/olda')
def olda():
    print(conf)
    return str(a[0])

@bp.route('/newa')
def newa():
    a[0]=1000
    return str(a[0])

