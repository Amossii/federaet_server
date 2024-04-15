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
    files = ['model.pkl']
    user=g.user
    for file_path in files:
        dir="./file/models/%s" % file_path
        file_size = os.path.getsize(dir)
        with open(dir,'rb')as f:
            text=f.read()
            model=Dpmodel_model(content=text,model_name=file_path,user_id=user.id,file_size=file_size,acc=0,loss=0)
            db.session.add(model)
            db.session.commit()
        f.close()
    return packMassage(200,"模型添加完成！",{})
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