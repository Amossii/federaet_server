import os

from exts import *
from models import File
from models import User
bp=Blueprint('file',__name__,url_prefix='/file')

@bp.route('/query')
def fileQuery():
    user=g.user
    data=[]
    for file in user.files:
        data.append({
                    'id':file.id,
                    'filename': file.filename,
                      'username': user.username,
                      'join_time': file.join_time})
    return packMassage(200,"获取用户文件成功!",{'fileInfo':data})

#文件名不可重复
@bp.route('/add')
def fileAdd():
    files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    user=g.user

    for file_path in files:
        dir="./file/%s" % file_path
        file_size = os.path.getsize(dir)
        with open(dir,'rb')as f:
            text=f.read()
            file=File(content=text,filename=file_path,user_id=user.id,file_size=file_size)
            db.session.add(file)
            db.session.commit()
        f.close()
    return packMassage(200,"文件添加完成！",{})

@bp.route('/delete')
def fileDelete():
    filename = request.args.get('filename', default='admin')
    user=g.user
    file = File.query.filter_by(user_id=user.id,filename=filename).first()
    if file:
        db.session.delete(file)
        db.session.commit()
        return packMassage(200, "删除文件成功", {})
    else:
        return packMassage(400, "文件不存在", {})


