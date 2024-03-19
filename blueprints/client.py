from exts import *
from dplearn.dataloader import Dataloader
from dplearn.client import Client
bp=Blueprint('client',__name__,url_prefix='/client')
from dplearn.utils.globalConf import conf
from dplearn.utils.myServer import *
from dplearn.utils.myClient import clients
from models import Client_model
from models import Dpmodel_model

#client对象


@bp.route('/train')
def clientTrain():
    if server.global_model==None:
        return packMassage(400,'the server has not been initial!',{})
    id = int(request.args.get('id', default='8888'))
    candidates=[obj for obj in clients if obj.client_id==id]
    if len(candidates)==0:
        return packMassage(400,"不存在请求的主机号！",{"id":id,"model_name":"..."})

    candidate=candidates[0]
    print("选中的主机号为%d" % candidate.client_id)

    print("client %d local train start..."% id )
    diff = candidate.local_train(server.global_model)
    # 根据客户端的参数差值字典更新总体权重
    for name, params in server.global_model.state_dict().items():
        weight_accumulator[name].add_(diff[name])

    acc, loss = candidate.model_eval()
    print(" acc: %f, loss: %f\n" % (acc, loss))
    return packMassage(200, "主机%d训练完毕acc: %f, loss: %f" % (id, acc, loss), {
        "number":id,
        "acc": acc,
        "loss": loss})
@bp.route('/train_save')
def clientTrain_Save():
    user=g.user
    id = int(request.args.get('id', default='8888'))
    model_name = request.args.get('model_name', default='model.pkl')
    if server.global_model==None:
        return packMassage(400,'the server has not been initial!',{})
    candidates=[obj for obj in clients if obj.client_id==id]
    if len(candidates)==0:
        return packMassage(400,"不存在请求的主机号！",{"id":id,"model_name":model_name})

    candidate=candidates[0]
    print("选中的主机号为%d" % candidate.client_id)

    print("client %d local train start..."% id )

    diff = candidate.local_train(server.global_model)

    # 根据客户端的参数差值字典更新权重
    for name, params in server.global_model.state_dict().items():
        weight_accumulator[name].add_(diff[name])

    acc, loss = candidate.model_eval()
    print(" acc: %f, loss: %f\n" % (acc, loss))
    content=candidate.getModel()
    model_name=str(id)+'_'+"{:.3f}".format(acc)+"_"+"{:.3f}".format(loss)
    model = Dpmodel_model(content=content, model_name=model_name, user_id=user.id, file_size=len(content),acc=acc,loss=loss)

    client=Client_model.query.get(id)
    client.model_name=model_name
    client.model_id=model.id
    client.flag="success"

    db.session.add(model)
    db.session.commit()



    return packMassage(200, "主机%d训练完毕acc: %f, loss: %f,并已保存" % (id, acc, loss), {
        "number":id,
        "acc": acc,
        "loss": loss})
@bp.route('/init',methods=['POST'])
def clientInit():
    pass

@bp.route('/clear')
def clientClearall():
    clients.clear()
    return "hello"

@bp.route('/add')
def clientAdd():
    number = request.args.get('number', default='8888')
    filename=request.args.get('filename',default='hahha')
    model_name = request.args.get('model_name', default='Null')

    if model_name!="Null":
        model=Dpmodel_model.query.filter_by(model_name=model_name).first()

    user = g.user
    client = Client_model.query.filter_by(user_id=user.id, number=number).first()
    if client:
        return packMassage(400,'添加主机失败，因为主机number重复！',{})

    if model_name!="Null":
        flag="success"
        model_id = model.id
    else:
        flag="fail"
        model_id = -1

    client = Client_model(user_id=user.id, filename=filename, number=number,model_name=model_name,flag=flag,model_id=model_id)
    db.session.add(client)
    db.session.commit()
    return packMassage(200,'添加主机成功！',{})
@bp.route('/query')
def clientQuery():
    user = g.user
    data = []
    for client in user.clients:
        data.append({
            'id': client.id,
            'filename': client.filename,
            'username': user.username,
            'number':client.number,
            'join_time': client.join_time,
            "model_name":client.model_name,
            "flag":client.flag,
            "model_id":client.model_id})
    return packMassage(200, "获取用户主机成功!", {'fileInfo': data})

@bp.route('/delete')
def clientDelete():
    number = request.args.get('number',default=-1)
    user = g.user
    client = Client_model.query.filter_by(user_id=user.id, number=number).first()
    if client:
        db.session.delete(client)
        db.session.commit()
        return packMassage(200, "删除主机成功", {})
    else:
        return packMassage(400, "删除失败！主机不存在", {})
@bp.route('/load')
def clientLoad():
    user=g.user

    id = int(request.args.get('id', default='8888'))
    model_name = request.args.get('model_name', default='')
    candidates = [obj for obj in clients if obj.client_id == id]
    if len(candidates) == 0:
        return packMassage(400, "不存在请求的主机号！", {})
    candidate = candidates[0]

    dpmodel =Dpmodel_model.query.filter_by(user_id=user.id,model_name=model_name).first()
    if dpmodel==None:
        return packMassage(400, "不存在模型！", {})

    candidate.modelLoad(dpmodel.content)
    acc, loss = candidate.model_eval()
    print(" acc: %f, loss: %f\n" % (acc, loss))
    return " acc: %f, loss: %f\n" % (acc, loss)

@bp.route('/eval')
def clientEval():
    id = int(request.args.get('id', default='8888'))
    candidates = [obj for obj in clients if obj.client_id == id]
    if len(candidates) == 0:
        return packMassage(400, "不存在请求的主机号！", {})

    candidate = candidates[0]
    acc, loss = candidate.model_eval()
    print(" acc: %f, loss: %f\n" % (acc, loss))
    return " acc: %f, loss: %f\n" % (acc, loss)