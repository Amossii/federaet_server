from exts import *
from .modules import captcha
bp=Blueprint('resources',__name__,url_prefix='')

@bp.route('/captcha')
def getCaptcha():
    imageURL, captchaID = captcha.generate_captcha()
    return packMassage(200, '发送成功', {
        'captchaURL': imageURL,
        'captchaID': captchaID
    })