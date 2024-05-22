import base64
import os
import pickle
from PIL import Image
import torch
from flask import Blueprint
bp=Blueprint('view',__name__,url_prefix='/view')
from exts import *
from models import *

@bp.route('/acc',methods=['get'])
def getacc():
    user=g.user
    epochs=user.epochs
    series={}

    for epoch in epochs:
        id=epoch.is_server
        if id not in series.keys():
            name='client'+str(id)
            if id==0:
                name='server'
            series[id]={
                'name': name,
                'data': [],
                'type': 'line'
            }

        series[id]['data'].append(epoch.acc)

    seri=[]
    for key in series.keys():
        seri.append(series[key])
    return packMassage(200,'',{'series':seri})
    # epochs=user.epochs
    # info={}
    # for epoch in epochs:
    #     state=epoch.epoch
    #     if state not in info:
    #         info[state]=[]
    #     model_id=epoch.model_id
    #     model=Dpmodel_model.query.get(model_id)
    #     info[state].append({
    #         "loss":model.loss,
    #         "acc":model.acc,
    #         "client":epoch.is_server,
    #         "epoch":state
    #     })

    # return packMassage(200,'',
    #                    {'series': [{
    #     'name': 'client1',
    #     'data': [37.87,
    #       41.09,
    #       43.05,
    #       45.51,
    #       47.5,
    #       48.01,
    #       49.59,
    #       51.12,
    #       51.69,
    #       53.36,
    #       54.62,
    #       55.02,
    #       54.76,
    #       55.21,
    #       56.39,
    #       57.19,
    #       57.95,
    #       58.8,
    #       59.39,
    #       59.5,
    #     ], 'type': 'line'
    #   }, {
    #     'name': 'client2',
    #     'data': [38.41,
    #       40.12,
    #       43.62,
    #       45.47,
    #       47.54,
    #       48.3,
    #       49.33,
    #       50.39,
    #       52.29,
    #       53.1,
    #       54.6,
    #       54.35,
    #       55.74,
    #       55.87,
    #       56.92,
    #       55.75,
    #       57.8,
    #       58.55,
    #       58.64,
    #       58.84,
    #     ], 'type': 'line'
    #   }, {
    #     'name': 'client3',
    #     'data': [37.13,
    #       41.54,
    #       43.28,
    #       45.93,
    #       46.42,
    #       48.99,
    #       50.46,
    #       51.47,
    #       52.25,
    #       53.64,
    #       53.43,
    #       54.16,
    #       55.27,
    #       56.33,
    #       56.69,
    #       56.89,
    #       58.45,
    #       59.07,
    #       59.88,
    #       58.79,
    #     ], 'type': 'line'
    #   }, {
    #     'name': 'server',
    #     'data': [15.14,
    #       17.57,
    #       26.53,
    #       36.15,
    #       42.44,
    #       45.76,
    #       48.36,
    #       50.33,
    #       51.87,
    #       52.94,
    #       54.18,
    #       55.14,
    #       55.84,
    #       56.88,
    #       57.64,
    #       57.97,
    #       58.67,
    #       59.27,
    #       59.76,
    #       60.49,
    #       1], 'type': 'line'
    #   }]})
@bp.route('/loss',methods=['get'])
def getloss():
    pass
@bp.route('/pic',methods=['get'])
def getPic():
    user=g.user

    pic_id = request.args.get('id', default=60)

    pic=File.query.get(pic_id).content

    base64_data = base64.b64encode(pic)
    base64_data = b'data:image/png;base64,' + base64_data
    return packMassage(200,'',{
        'photoURL': base64_data.decode()
    })
@bp.route('/predict',methods=['get'])
def predict():
    user=g.user

    pic_id=request.args.get('pic_id',default=60)

    content = File.query.get(pic_id).content
    img_path = 'D:\Python\myapp\\file\img\\bee_raw.jpg'

    with open(img_path, 'wb') as file:
        file.write(content)

    from PIL import Image
    import pickle
    import torch
    # 有 GPU 就用 GPU，没有就用 CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    with open('D:\Python\codeServer\imagenet\model\model2', 'rb') as f:
        model = pickle.load(f)
    model.to(device)

    from torchcam.methods import GradCAM
    target_layer = model.layer4[-1]  # 选择目标层
    cam_extractor = GradCAM(model, target_layer)
    #
    # # 图片预处理
    from torchvision import transforms
    # # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                         ])
    #
    # # 图片分类预测
    img_pil = Image.open(img_path)

    input_tensor = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_tensor)
    # # topk()方法用于返回输入数据中特定维度上的前k个最大的元素
    pred_top1 = torch.topk(pred_logits, 1)
    # # pred_id 为图片所属分类对应的索引号，分类和索引号存储在imagenet_class_index.csv
    pred_id = pred_top1[1].detach().cpu().numpy().squeeze().item()

    #
    # # 生成可解释性分析热力图
    activation_map = cam_extractor(pred_id, pred_logits)
    activation_map = activation_map[0][0].detach().cpu().numpy()
    #
    # # 可视化
    from torchcam.utils import overlay_mask
    #
    result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imshow(result)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    dir='bee.png'
    plt.savefig(dir)

    with open(dir, "rb") as file:
        base64_data = base64.b64encode(file.read())
    os.remove(dir)
    base64_data = b'data:image/png;base64,' + base64_data

    # classfiction='ant' if pred_id==310 else 'bee'
    return packMassage(200,'',{
        'picURL':base64_data.decode(),
        'predict':'the predict classification is : ant' if pred_id==310 else 'the predict classification is : bee'
    })
@bp.route('draw')
def draw_pic():
    user=g.user
    pic_id=request.args.get('pic_id',default=61)
    content = File.query.get(pic_id).content

    with open('pic.png','wb')as file:
        file.write(content)

    return 'done'

    # dir = 'D:\Python\myapp\\file\img\\bee_raw.jpg'
    #
    # with open(dir, "rb") as file:
    #     base64_data = base64.b64encode(file.read())
    # os.remove(dir)
    # base64_data = b'data:image/png;base64,' + base64_data
    #
    # return packMassage(200, '', {
    #     'picURL': base64_data.decode(),
    #     'predict': 'the predict classification is : bee'
    # })
@bp.route('predict2')
def predict2():
    user=g.user
    model_id=request.args.get('model_id',default=0)
    pic_id=request.args.get('pic_id',default=0)
    content = File.query.get(pic_id).content
    img_path = 'D:\Python\myapp\\file\img\\bee_raw.jpg'
    with open(img_path, 'wb') as file:
        file.write(content)
    model=pickle.loads(Dpmodel_model.query.get(model_id).content)
    #
    from torchcam.methods import GradCAM
    target_layer = model.layer4[-1]  # 选择目标层
    cam_extractor = GradCAM(model, target_layer)
    img_pil = Image.open(img_path)

    from torchvision import transforms
    # # # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                         ])
    print(type(img_pil))
    input_tensor = test_transform(img_pil).unsqueeze(0)  # 预处理
    pred_logits = model(input_tensor)
    # # topk()方法用于返回输入数据中特定维度上的前k个最大的元素
    pred_top1 = torch.topk(pred_logits, 1)
    # # pred_id 为图片所属分类对应的索引号，分类和索引号存储在imagenet_class_index.csv
    pred_id = pred_top1[1].detach().cpu().numpy().squeeze().item()
    print(pred_id)
    #
    activation_map = cam_extractor(pred_id, pred_logits)
    activation_map = activation_map[0][0].detach().cpu().numpy()
    #
    # # 可视化
    from torchcam.utils import overlay_mask
    #
    # # overlay_mask 用于构建透明的叠加层
    # # fromarray 实现array到image的转换
    result = overlay_mask(img_pil, Image.fromarray(activation_map), alpha=0.7)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imshow(result)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.savefig('myplot1.png')
    dir = 'bee.png'
    plt.savefig(dir)

    with open(dir, "rb") as file:
        base64_data = base64.b64encode(file.read())
    os.remove(dir)
    base64_data = b'data:image/png;base64,' + base64_data
    label_name = ['airplane', 'automobile', 'brid', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # classfiction='ant' if pred_id==310 else 'bee'
    return packMassage(200, '', {
        'picURL': base64_data.decode(),
        'predict': 'the predict classification is : '+label_name[pred_id]
    })