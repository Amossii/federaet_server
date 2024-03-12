import os

from exts import *
from models import Dpmodel_model
bp=Blueprint('dpmodel',__name__,url_prefix='/model')

@bp.route('/query')
def modelQuery():
    user=g.user
    data=[]
    for model in user.models:
        data.append({
                    'id':model.id,
                    'filename': model.model_name,
                    'username': user.username,
                    'join_time': model.join_time})
    return packMassage(200,"获取用户模型成功!",{'fileInfo':data})
@bp.route('/add')
def modelAdd():
    files = ['model.pkl']
    user=g.user
    for file_path in files:
        dir="./file/models/%s" % file_path
        file_size = os.path.getsize(dir)
        with open(dir,'rb')as f:
            text=f.read()
            model=Dpmodel_model(content=text,model_name=file_path,user_id=user.id,file_size=file_size)
            db.session.add(model)
            db.session.commit()
        f.close()
    return packMassage(200,"模型添加完成！",{})