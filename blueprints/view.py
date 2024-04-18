from flask import Blueprint
bp=Blueprint('view',__name__,url_prefix='/view')
from exts import *
from models import *

@bp.route('/info',methods=['get'])
def info():
    user=g.user

    epochs=user.epochs
    info={}
    for epoch in epochs:
        state=epoch.epoch
        if state not in info:
            info[state]=[]
        model_id=epoch.model_id
        model=Dpmodel_model.query.get(model_id)
        info[state].append({
            "loss":model.loss,
            "acc":model.acc,
            "client":epoch.is_server,
            "epoch":state
        })

    return packMassage(200,'',{'info':info})


