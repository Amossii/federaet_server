import os
from exts import *
from models import Dpmodel_model
bp=Blueprint('dpmodel',__name__,url_prefix='/model')

@bp.route('/',methods=['get'])
def modelQuery():
    user=g.user
    data=[]
    for model in user.models:
        data.append({
                    'id':model.id,
                    'modelname': model.model_name,
                    'username': user.username,
                    'join_time': model.join_time,
                    'acc':model.acc,
                    'loss':model.loss})
    return packMassage(200,"获取用户模型成功!",{'fileInfo':data})

@bp.route('/',methods=['post'])
def modelAdd():
    user = g.user
    # 检查是否存在上传文件
    if 'model' not in request.files:
        return packMassage(400, 'no model', {})

    model = request.files['model']

    # 检查文件名是否为空
    if model.filename == '':
        return packMassage(400, 'No selected file', {})

    # 保存文件到指定路径
    content = model.read()
    model = Dpmodel_model(content=content, model_name=model.filename, user_id=user.id, file_size=len(content),acc=0,loss=0)
    db.session.add(model)
    db.session.commit()

    return packMassage(200, 'Model uploaded successfully', {})

@bp.route('/',methods=['delete'])
def modelDelete():
    model_id = request.args.get('id', default='0')
    user = g.user
    model=Dpmodel_model.query.get(model_id)
    if model:
        db.session.delete(model)
        db.session.commit()
        return packMassage(200, "删除模型成功", {})
    else:
        return packMassage(400, "模型不存在", {})