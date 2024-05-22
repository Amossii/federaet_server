import base64
import os

from exts import *
from .modules import captcha
from models import File
import pickle
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
bp=Blueprint('resources',__name__,url_prefix='')

@bp.route('/captcha')
def getCaptcha():
    imageURL, captchaID = captcha.generate_captcha()
    return packMassage(200, '发送成功', {
        'captchaURL': imageURL,
        'captchaID': captchaID
    })
@bp.route('/photo')
def getphoto():
    file_id=request.args.get('id',default=60)

    photo=File.query.get(file_id).content

    dict=pickle.loads(photo.content)
    data=dict[b'data'][0]
    img=data.reshape(3,32,32)
    img=np.transpose(img, (1, 2, 0))

    with BytesIO() as buffer:
        plt.imsave(buffer, img, format='png')  # 将 NumPy 数组以 PNG 格式保存到缓冲区中
        img_str = base64.b64encode(buffer.getvalue())

    base64_data = b'data:image/png;base64,' + img_str
    return packMassage(200, '发送成功', {
        'photoURL': base64_data.decode(),
        'photoname':dict[b'filenames'].decode()
    })
