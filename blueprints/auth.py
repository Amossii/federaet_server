from exts import *
from models import *
bp=Blueprint('auth',__name__,url_prefix="/auth")

@bp.route('/login',methods=['post'])
def login():
    username = request.args.get('username', default='admin')
    password = request.args.get('password', default='admin')
    if not username or not password:
        return packMassage(400,'lack of parameters',{})
    user = User.query.filter_by(username=username).first()
    if not user:
        return packMassage(400, "用户名不存在！", {})
    elif not user._check_password(password):
        return packMassage(400, "密码错误！", {})
    session['userID'] = user.id
    return packMassage(200, "登录成功！", {
        'token': "do ametmollit ut ex dolore",
        "routes": "*"
    })
@bp.route('/logout')
def logout():
    session.clear()
    return packMassage(200, "退出登录成功！", {})

@bp.route('/user',methods=['post'])
def register():
    username = request.args.get('username', default='admin')
    password = request.args.get('password', default='admin')
    access = int(request.args.get('access', default=0))
    email=request.args.get('email', default='helloworld@python.com')
    user = User.query.filter_by(username=username).first()
    if user:
        return packMassage(400, "注册的用户已存在！", {'username':username})
    user = User(username=username, _password=password,access=access,email=email)
    db.session.add(user)
    db.session.commit()
    return packMassage(200, "用户注册成功！", {
        'username':username,
        'password':password,
        'access':access,
        'email':email
    })

@bp.route('/check')
def check():

    if g.user:
        username = g.user.username
        return packMassage(200, "查询用户成功！", {"username": username})
    else:
        return packMassage(200,"用户未登录！",{})
@bp.route('/info')
def getInfo():
    user=g.user
    fileSize=0
    fileTotal=0
    taskTotal = 0
    transcodeTotal = 0

    files = File.query.filter_by(user_id=user.id)
    for file in files:
        fileSize+=file.file_size
        fileTotal+=1

    models=user.models
    for model in models:
        fileSize+=model.file_size
        transcodeTotal+=1

    return packMassage(200,'查询成功！',{
        "fileTotal":fileTotal,
        "fileSize":fileSize,
        "taskTotal":taskTotal,
        "transcodeTotal":transcodeTotal
    })
@bp.route('/user',methods=['PUT'])
def set():
    id=request.args.get('id', default='0')
    username = request.args.get('username', default='admin')
    password = request.args.get('password', default='admin')
    access = int(request.args.get('access', default=0))
    email = request.args.get('email', default='helloworld@python.com')
    user = User.query.get(id)
    if not user:
        return packMassage(400, "编辑的用户不存在！", {})
    user.username=username
    user._password=password
    user.access=access
    user.email=email
    user.join_time=datetime.now()
    db.session.commit()
    return packMassage(200, "用户信息修改成功！", {
        'username': username,
        'password': password,
        'access': access,
        'email': email
    })
@bp.route('/user',methods=['DELETE'])
def delete():
    id = request.args.get('id', default='0')
    user = User.query.get(id)
    if not user:
        return packMassage(400, "需要的用户不存在！", {})
    else:
        db.session.delete(user)
        db.session.commit()
        return packMassage(200, "用户%s删除成功！" % (user.username), {})

@bp.route('/user',methods=['get'])
def query():
    user=g.user
    if user.access==0:
        return packMassage(400,'你无权访问此页面，请先获取权限',{})

    users=User.query.all()
    data=[{'id':user.id,
           'username':user.username,
           'password':user.password,
           'join_time':user.join_time,
           'access':user.access,
           'email':user.email} for user in users]

    return packMassage(200,'查询用户数据成功！',{'userInfo':data})