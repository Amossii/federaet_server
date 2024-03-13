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
    id = request.args.get('id', default='admin')
    user=g.user
    file = File.query.get(id)
    if file:
        db.session.delete(file)
        db.session.commit()
        return packMassage(200, "删除文件成功", {})
    else:
        return packMassage(400, "文件不存在", {})


@bp.route('/upload', methods=['POST'])
def upload_file():
    user=g.user
    # 检查是否存在上传文件
    if 'file' not in request.files:
        return packMassage(400,'no file',{})

    file = request.files['file']

    # 检查文件名是否为空
    if file.filename == '':
        return packMassage(400,'No selected file',{})

    # 保存文件到指定路径
    content=file.read()
    file = File(content=content, filename=file.filename, user_id=user.id, file_size=len(content))
    db.session.add(file)
    db.session.commit()

    return packMassage(200,'File uploaded successfully',{})
