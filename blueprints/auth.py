from exts import *
from models import *
bp=Blueprint('auth',__name__,url_prefix="/auth")

@bp.route('/login',methods=['POST','GET'])
def login():

    username = request.args.get('username', default='admin')
    password = request.args.get('password', default='admin')
    user = User.query.filter_by(username=username).first()

    if user==None:
        return packMassage(400, "用户名不存在！", {})
    elif user.password!=password:
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

@bp.route('/register')
def register():
    username = request.args.get('username', default='admin')
    password = request.args.get('password', default='admin')

    user = User.query.filter_by(username=username).first()
    if user:
        return packMassage(400, "注册的用户已存在！", {})
    user = User(username=username, password=password)
    db.session.add(user)
    db.session.commit()
    return packMassage(200, "用户注册成功！", {})

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
    files = File.query.filter_by(user_id=user.id)
    for file in files:
        fileSize+=file.file_size
        fileTotal+=1
    taskTotal=0
    transcodeTotal= 4
    return packMassage(200,'查询成功！',{
        "fileTotal":fileTotal,
        "fileSize":fileSize,
        "taskTotal":taskTotal,
        "transcodeTotal":transcodeTotal
    })