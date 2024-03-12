from exts import *
from dplearn.dataloader import Dataloader
from dplearn.client import Client
bp=Blueprint('client',__name__,url_prefix='/client')
from dplearn.utils.globalConf import conf
from dplearn.utils.myServer import *
from dplearn.utils.myClient import clients
from models import Client_model

#client对象


@bp.route('/train')
def clientTrain():
    id = int(request.args.get('id', default='8888'))
    candidates=[obj for obj in clients if obj.client_id==id]
    if len(candidates)==0:
        return packMassage(400,"不存在请求的主机号！",{})

    candidate=candidates[0]
    print("选中的主机号为%d" % candidate.client_id)

    print("client %d local train start..."% id )
    diff = candidate.local_train(server.global_model)
    # 根据客户端的参数差值字典更新总体权重
    for name, params in server.global_model.state_dict().items():
        weight_accumulator[name].add_(diff[name])

    acc, loss = candidate.model_eval()
    print(" acc: %f, loss: %f\n" % (acc, loss))
    candidate.modelSave()
    return packMassage(200, "主机%d训练完毕acc: %f, loss: %f" % (id, acc, loss), {
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
    user = g.user
    client = Client_model.query.filter_by(user_id=user.id, number=number).first()
    if client:
        return packMassage(400,'添加主机失败，因为主机number重复！',{})
    client = Client_model(user_id=user.id, filename=filename, number=number)
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
            'join_time': client.join_time})
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
    id = int(request.args.get('id', default='8888'))
    candidates = [obj for obj in clients if obj.client_id == id]
    if len(candidates) == 0:
        return packMassage(400, "不存在请求的主机号！", {})

    candidate = candidates[0]
    candidate.modelLoad()
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