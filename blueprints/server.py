import json
import pickle
import time

import cv2
import numpy as np
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
from encrypt.homo import SimpleAdditiveHomomorphic
from encrypt.shamir import *
bp=Blueprint('server',__name__,url_prefix='/server')

@bp.route('/init')
def serverInit():
    pass
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
@bp.route('/test')
def serverSave():
    for name, params in server.global_model.state_dict().items():
        # 生成一个和参数矩阵大小相同的0矩阵
        weight_accumulator[name] = torch.zeros_like(params)
        print(params.shape)
        print(torch.zeros_like(params).shape)
    return 'done'
@bp.route('/clear')
def serverClear():
    conf.clear()
    weight_accumulator.clear()
    pass
@bp.route('encrypt')
def encrypt():
    s=time.time()
    #创建一个同态加密实例
    crypto = SimpleAdditiveHomomorphic()

    user = g.user
    epoch = conf['epoch']
    filename = request.args.get("filename", default="global_aggregate")
    filename = filename + '_epoch' + str(epoch)
    # 初始化weight
    weight_accumulator.clear()
    for name, params in server.global_model.state_dict().items():
        print(params)
        weight_accumulator[name] = torch.zeros_like(params)

    for client in clients:
        model_id = client.model_id
        if model_id == -1:
            return packMassage(400, "client model is not exist", {})
        print(model_id)
        client_model = Dpmodel_model.query.get(model_id)
        client.modelLoad(client_model.content)
        diff = client.encryptDiff(server.global_model,crypto)
        for name, params in server.global_model.state_dict().items():
            # print(diff[name])
            weight_accumulator[name].add_(diff[name])

        # save epoch info into database
        epoch_client = Epoch(epoch=epoch, is_server=client.server_id, model_id=client_model.id,
                             model_name=client_model.model_name,
                             user_id=user.id)
        db.session.add(epoch_client)


    # for name, data in server.global_model.state_dict().items():
    #     # 更新每一层乘上学习率
    #     print(weight_accumulator[name])

    decryption=clients[0].decryptDiff(server.global_model,crypto,weight_accumulator)

    for name, data in server.global_model.state_dict().items():
        # 更新每一层乘上学习率
        print(decryption[name])


    server.model_aggregate(decryption)

    # save the model
    content = server.getModel()
    acc, loss = server.model_eval()
    model = Dpmodel_model(content=content, model_name=filename, user_id=user.id, file_size=len(content), acc=acc,
                          loss=loss)
    db.session.add(model)
    db.session.commit()

    model_server = Dpmodel_model.query.filter_by(model_name=filename, user_id=user.id).first()
    epoch_server = Epoch(epoch=epoch, is_server=0, model_id=model_server.id, model_name=model_server.model_name,
                         user_id=user.id)
    db.session.add(epoch_server)
    db.session.commit()
    end=time.time()
    return packMassage(200, "the acc is %f,loss is %f" % (acc, loss), {"acc": acc, "loss": loss,"time":end-s})

@bp.route('/aggregate')
def serverAggregate():
    start=time.time()
    user=g.user
    epoch = conf['epoch']
    filename=request.args.get("modelName",default="global_aggregate")
    filename=filename+'_epoch'+str(epoch)
    # 初始化weight
    weight_accumulator.clear()
    weight_accumulator_rand={}
    for name, params in server.global_model.state_dict().items():
        # 生成一个随机参数矩阵,防止服务器知道每个客户端的参数梯度信息
        # randn=torch.randn(params.shape)
        # weight_accumulator[name] = randn
        # weight_accumulator_rand[name]=randn.clone()
        weight_accumulator[name]=torch.randn(params.shape)

    for client in clients:
        model_id=client.model_id
        if model_id==-1:
            return packMassage(400,"client model is not exist",{})
        print(model_id)
        client_model=Dpmodel_model.query.get(model_id)
        client.modelLoad(client_model.content)
        # acc,loss=client.model_eval()
        # print(acc,loss)
        diff=client.getDiff(server.global_model)
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name].add_(diff[name])


        #save epoch info into database
        epoch_client=Epoch(epoch=epoch,is_server=client.server_id,model_id=client_model.id,model_name=client_model.model_name,
                           user_id=user.id,acc=client_model.acc,loss=client_model.loss)
        db.session.add(epoch_client)



    # for name, params in server.global_model.state_dict().items():
    #     # 还原平均薪水问题中加上的平均数
    #     weight_accumulator[name].sub_(weight_accumulator_rand[name])



    server.model_aggregate(weight_accumulator)

    #save the model
    content=server.getModel()
    acc,loss=server.model_eval()
    model = Dpmodel_model(content=content, model_name=filename, user_id=user.id, file_size=len(content), acc=acc,
                          loss=loss)
    db.session.add(model)
    db.session.commit()

    model_server=Dpmodel_model.query.filter_by(model_name=filename,user_id=user.id).first()
    epoch_server=Epoch(epoch=epoch,is_server=0,model_id=model_server.id,model_name=model_server.model_name,
                       user_id=user.id,acc=acc,loss=loss)
    db.session.add(epoch_server)
    db.session.commit()
    end=time.time()
    return packMassage(200,"the acc is %f,loss is %f" % (acc,loss),{"acc":acc,"loss":loss,"time":end-start})



@bp.route('/conf',methods=['POST'])
def getConf():
    user=g.user
    conf_data=request.json
    if not conf_data:
        return packMassage(400,'初始化全局参数失败，请重试！',conf_data)
    conf.update(conf_data)
    #初始化server
    filename = conf['test_file']
    dataloader = Dataloader()
    eval_datasets = dataloader.getEvalData([filename])
    # init global model with the previous model
    epoch=conf['epoch']
    if conf['model_init']!="None":
        model=Dpmodel_model.query.filter_by(user_id=user.id, model_name=conf['model_init']).first()
        if not model:
            return packMassage(400,'init with a model name not exist',{})
        server.myInit(conf,eval_datasets,model.content,epoch)
    else:
        server.myInit(conf, eval_datasets,None,epoch)

    #添加client到clients里



    clients.clear()
    dataloader = Dataloader()
    all_clients=g.user.clients


    for client in all_clients:

        index = client.number
        train_datasets = dataloader.getTrainData([client.filename])
        if client.flag!='success':
            new_client=Client(conf,server.global_model,train_datasets,eval_datasets,index,client.model_name,
                              client.model_id,client.number)
        else:
            client_model=pickle.loads(Dpmodel_model.query.get(client.model_id).content)
            new_client = Client(conf, client_model, train_datasets, eval_datasets, index, client.model_name,
                                client.model_id, client.number)
        clients.append(new_client)

    return packMassage(200, '初始化成功！', conf)

@bp.route('/eval')
def eval():
    user=g.user
    model_name=request.args.get('model_name',default='model_global')
    acc, loss = server.model_eval()
    print(" acc: %f, loss: %f\n" % ( acc, loss))
    content = server.getModel()
    model = Dpmodel_model(content=content, model_name=model_name, user_id=user.id, file_size=len(content), acc=acc,
                          loss=loss)
    db.session.add(model)
    db.session.commit()
    return packMassage(200,"acc: %f, loss: %f,model has already saved" % ( acc, loss),{
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

@bp.route('/predict')
def predict():
    user=g.user
    filename = request.args.get('filename', default='admin')
    dataloader = Dataloader()
    predict_datasets = dataloader.getEvalData([filename])
    predict,target=server.model_predict(predict_datasets)
    return packMassage(200,'predict is done,and target %d answer is %d'%(target,predict),{'target':target,'predict':predict})
@bp.route('/heat')
def heat():

    user = g.user
    filename = request.args.get('filename', default='admin')
    file=File.query.filter_by(filename=filename,user_id=user.id).first()
    nd=pickle.loads(file.content)[b'data']

    nd=nd.reshape(3,32,32)
    cv2.imwrite('D:\Python\myapp\\file\img\Figure_3.jpg'
        , np.transpose(nd, (1, 2, 0)))
    server.heat('D:\Python\myapp\\file\img\Figure_3.jpg')
    return "done"

@bp.route('key',methods=['get'])
def generate_key():
    user=g.user
    myClients=user.clients

    client_num = len(g.user.clients)

    crypto = SimpleAdditiveHomomorphic()
    conf['key'] = crypto.k
    shares = shamir(conf['key'], client_num, client_num)
    for client,share in zip(myClients,shares):
        client.xi = str(share[0])
        client.yi = str(share[1])
        client.cipher = str(conf['key'])
    db.session.commit()
    print(shares)
    return packMassage(200,'the key is generating',{'data':shares})