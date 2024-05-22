//model_name：模型名称
//no_models：客户端总数量
//type：数据集信息
//global_epochs：全局迭代次数，即服务端与客户端的通信迭代次数
//local_epochs：本地模型训练迭代次数
//batch_size：本地训练每一轮的样本数
//lr，momentum，lambda：本地训练的超参数设置

{
  "model_name" : "resnet18",
  "no_models" : 10,
  "type" : "cifar",
  "global_epochs" : 2,
  "local_epochs" : 1,
  "k" : 1,
  "batch_size" : 32,
  "lr" : 0.001,
  "momentum" : 0.0001,
  "lambda" : 0.1
}
