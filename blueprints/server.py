import json
from exts import *
from models import *
import random
import torch
from dplearn.client import Client
from dplearn.utils.myServer import server as server
from dplearn.utils.globalConf import conf
from dplearn.utils.myClient import clients
from dplearn.utils.myServer import weight_accumulator
from dplearn.dataloader import Dataloader
bp=Blueprint('server',__name__,url_prefix='/server')

#全局一个server对象，包括创建和重置以及全局训练的方法

# @bp.route('/init')
# def serverInit():
#     filename = request.args.get('filename', default='admin')
#     dataloader=Dataloader()
#     if not conf:
#         with open("./dplearn/conf.json", 'r') as f:
#             conf.update(json.load(f))
#     eval_datasets = dataloader.getEvalData([filename])
#     server.myInit(conf, eval_datasets)
#
#     for name, params in server.global_model.state_dict().items():
#         # 生成一个和参数矩阵大小相同的0矩阵
#         weight_accumulator[name] = torch.zeros_like(params)
#
#     return packMassage(200,'初始化成功！',{})
@bp.route('/clear')
def serverClear():
    conf.clear()
    weight_accumulator.clear()
    pass

@bp.route('/aggregate')
def serverAggregate():
    server.model_aggregate(weight_accumulator)
    weight_accumulator.clear()
    for name, params in server.global_model.state_dict().items():
        # 生成一个和参数矩阵大小相同的0矩阵
        weight_accumulator[name] = torch.zeros_like(params)
    return "done"

@bp.route('/conf',methods=['POST'])
def getConf():
    conf_data=request.json
    if not conf_data:
        return packMassage(400,'初始化全局参数失败，请重试！',conf_data)
    conf.update(conf_data)
    #初始化server
    filename = conf['test_file']
    dataloader = Dataloader()
    eval_datasets = dataloader.getEvalData([filename])
    server.myInit(conf, eval_datasets)
    #初始化weight
    weight_accumulator.clear()
    for name, params in server.global_model.state_dict().items():
        # 生成一个和参数矩阵大小相同的0矩阵
        weight_accumulator[name] = torch.zeros_like(params)
    #添加client到clients里
    clients.clear()
    dataloader = Dataloader()
    all_clients=g.user.clients
    for client in all_clients:
        index = client.number
        train_datasets = dataloader.getTrainData([client.filename])
        clients.append(Client(conf, server.global_model, train_datasets, eval_datasets,index))
    return packMassage(200, '初始化成功！', conf)
@bp.route('/eval')
def eval():
    acc, loss = server.model_eval()
    print(" acc: %f, loss: %f\n" % ( acc, loss))
    return packMassage(200,"acc: %f, loss: %f" % ( acc, loss),{
        "acc":acc,
        "loss":loss })
@bp.route('/test')
def test():
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

