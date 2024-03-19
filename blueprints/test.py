from flask import Blueprint
from dplearn import dataset
from exts import  *
import json
from dplearn.server import *
from dplearn.client import *
import random
from dplearn.dataloader import Dataloader
from dplearn.utils.globalConf import conf
from models import Dpmodel_model

bp=Blueprint('test',__name__,url_prefix='/test')
from dplearn.utils.myServer import server as server
from dplearn.utils.globalConf import conf
from dplearn.utils.myClient import clients
from dplearn.utils.myServer import weight_accumulator
@bp.route('/')
def test():
    user=g.user
    # 全局模型训练
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        # 每次训练都是从clients列表中随机采样k个进行本轮训练
        candidates = random.sample(clients, conf["k"])
        print("select clients is: ")
        for c in candidates:
            print(c.client_id)
        weight_accumulator.clear()
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            weight_accumulator[name] = torch.zeros_like(params)
        # 遍历客户端，每个客户端本地训练模型
        for c in candidates:
            print("the train of %d is start...."% c.client_id)
            diff = c.local_train(server.global_model)
            acc, loss = server.model_eval()
            print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
            # 根据客户端的参数差值字典更新总体权重
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        # 模型参数聚合
        server.model_aggregate(weight_accumulator)

        # 模型评估
        acc, loss = server.model_eval()
        content = server.getModel()
        model = Dpmodel_model(content=content, model_name='global_model', user_id=user.id, file_size=len(content), acc=acc,
                              loss=loss)
        db.session.add(model)
        db.session.commit()
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

