
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from math import *
import random
import time

from EncDec import Encoder,Decoder

from GauMento import Gaulikely,MonteCarlo

#新数据片段拆分
train_num=6000
valid_num=1000
test_num=800

#定义encoder网络参数
# MTCN的超参数
input_channels = 13
num_hidden = 20
levels = 2
channel_sizes = [num_hidden] * levels
kernel_size = 7
dropout = 0.1

# VIFE超参数
n_heads = 5
d_feature = 13
d_model = 20  # Embedding Size
d_ff = 40  # FeedForward dimension
d_k = d_v = 4  # dimension of K(=Q), V

# 定义decoder网络参数
input_size = 13
output_size = 1
hidden_size = 20
n_layers = 1
dropout_p = 0.1

input_length = 40
output_length=5
learning_rate=0.003
batch_size = 20
num_epoch=15
retrain=True

#定义MentoCarlo参数
num_sampling=10000

#数据条数拆分
num_trainvalid=(train_num+valid_num)+(input_length + output_length)
num_test=input_length + output_length+test_num*output_length
total_num=num_trainvalid+num_test

#对数据进行整理
data_initial=pd.read_csv(r"F:\博一\柳钢项目\烧结终点研究\原始数据前2万test.csv")
#取样本
data_initial=data_initial.iloc[:total_num,:]

data=data_initial.drop(columns=["time2","time3"])
data=data.set_index(data["time1"]).drop(columns=["time1","Unnamed: 0"])
#BRP和BTP两列交换位置
cols = list(data)
cols.insert(-1,cols.pop(cols.index("BRP")))
data = data.loc[:,cols]
data=np.asarray(data)

"处理BTP大于90的数据"
num_trainvalid=(train_num+valid_num)+(input_length + output_length)

for i in range(data.shape[0]):
        if data[i,-1]>90:
            data[i,-1]=np.mean(np.random.normal(87, 1, 1000))

data_x=data[:,:-1]
data_y=data[:,-1].reshape(-1,1)

# 归一化
min_max = MinMaxScaler()
data = min_max.fit_transform(data)

Y_min_max = MinMaxScaler()
Y_data = Y_min_max.fit_transform(data_y)

#提取片段
feature_train=[]
label_train=[]

for i in range(0,len(data) - (input_length + output_length)):
    feature_train.append(data[i:i + input_length, :])
    label_train.append(data[i + input_length:i + input_length + output_length, :])

#训练集
train_x=np.asarray(feature_train[0:train_num])
train_y=np.asarray(label_train[0:train_num])[:,:,-1]
train_y=np.expand_dims(train_y,2)
#验证集
valid_x=np.asarray(feature_train[train_num:(train_num+valid_num)])
valid_y=np.asarray(label_train[train_num:(train_num+valid_num)])[:,:,-1]
valid_y=np.expand_dims(valid_y,2)

#测试集的构造
#提取片段
feature_test=[]
label_test=[]

for i in range((train_num+valid_num)+(input_length + output_length), len(data) - (input_length + output_length), output_length):
    feature_test.append(data[i:i + input_length, :])
    label_test.append(data[i + input_length:i + input_length + output_length, :])

#测试集
test_x=np.asarray(feature_test[0:test_num])
test_y=np.asarray(label_test[0:test_num])[:,:,-1]
test_y=np.expand_dims(test_y,2)

#数据集的构造
class MYDataset(Dataset):
    def __init__(self,x_tensor,y_tensor):
        self.feature=x_tensor
        self.label=y_tensor

    def __getitem__(self, index):
        return self.feature[index],self.label[index]

    def __len__(self):
        return self.feature.size(0)

#转tensor，载入数据
#训练集
train_dataset=MYDataset(torch.from_numpy(train_x).float(),torch.from_numpy(train_y).float())
train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

#验证集
valid_dataset=MYDataset(torch.from_numpy(valid_x).float(),torch.from_numpy(valid_y).float())
valid_loader=DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

#测试集
test_dataset=MYDataset(torch.from_numpy(test_x).float(),torch.from_numpy(test_y).float())
test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#开始带有注意力机制的RNN训练
#实例化Encoder和Decoder
encoder = Encoder(input_channels, channel_sizes, kernel_size=kernel_size, dropout=dropout,
                    d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)
decoder=Decoder(hidden_size, output_size, n_layers, dropout_p, input_length)

#优化器
encoder_optimizer=torch.optim.Adam(encoder.parameters(),lr=learning_rate)
decoder_optimizer=torch.optim.Adam(decoder.parameters(),lr=learning_rate)

loss_func=nn.MSELoss()

#定义评价指标
def mape(y_true, y_pred):

    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

def HR(actual,predict,ratio):
    count=[]
    error=actual*ratio
    for i in range(actual.shape[0]):
        if abs(actual[i]-predict[i])<error[i]:
            count.append(1)
    return sum(count)/actual.shape[0]

#开始训练周期循坏
plot_losses=[]
for epoch in range(1,num_epoch+1):

    #将解码器置于训练状态，并让dropout工作
    time_start1 = time.time()
    train_loss=0
    #对训练数据进行循环
    for step,data_batch in enumerate (train_loader):

        x_train=data_batch[0]
        y_train=data_batch[1]

        loss = 0

        encoder_outputs = encoder(x_train)
        decoder_hidden = encoder_outputs[:, -1, :].unsqueeze(0)
        y_hist = x_train[:, :, -1]
        for di in range(output_length):

            gaulikely = Gaulikely(y_hist)
            mu, sigma = gaulikely(y_hist)
            decoder_input = MonteCarlo(mu, sigma, num_sampling)
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 获取解码器的预测结果，并作为下一个时刻的输入
            y_hist=torch.cat((y_hist,decoder_output),1)
            # decoder_input大小：batch_size,length_seq
            loss = loss + loss_func(decoder_output[:, -1], y_train[:, di, -1])

        train_loss += loss.data.numpy()
        # 反向传播
        # 清空梯度
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        # 开始梯度下降
        encoder_optimizer.step()
        decoder_optimizer.step()
    time_end1 = time.time()
    interval_train = time_end1 - time_start1

    #模型校验
    valid_loss=0
    #对所有的校验数据做循环
    for step,data_batch in enumerate(valid_loader):
        x_valid = data_batch[0]
        y_valid = data_batch[1]

        loss = 0

        encoder_outputs = encoder(x_valid)
        decoder_hidden = encoder_outputs[:, -1, :].unsqueeze(0)
        y_hist = x_valid[:, :, -1]

        for di in range(output_length):
            gaulikely = Gaulikely(y_hist)
            mu, sigma = gaulikely(y_hist)
            decoder_input = MonteCarlo(mu, sigma, num_sampling)

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 获取解码器的预测结果，并作为下一个时刻的输入
            y_hist = torch.cat((y_hist, decoder_output), 1)
            # decoder_input大小：batch_size,length_seq
            loss = loss + loss_func(decoder_output[:, -1], y_valid[:, di, -1])

        valid_loss += loss.data.numpy()

    print("第%d个Epoch,训练损失：%.4f,校验损失：%.4f，训练时间：%.4f" % (
        epoch, train_loss / len(train_x), valid_loss / len(valid_x), interval_train))

#测试
y_predict_all =torch.tensor([])
y_actual=torch.tensor([])
test_loss=0
for step,data_batch in enumerate(test_loader):
    x_test = data_batch[0]
    y_test = data_batch[1]

    loss = 0

    encoder_outputs = encoder(x_test)
    decoder_hidden = encoder_outputs[:, -1, :].unsqueeze(0)
    y_hist = x_test[:, :, -1]

    y_predict = torch.tensor([])
    for di in range(output_length):
        gaulikely = Gaulikely(y_hist)
        mu, sigma = gaulikely(y_hist)
        decoder_input = MonteCarlo(mu, sigma, num_sampling)
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        # 获取解码器的预测结果，并作为下一个时刻的输入
        y_hist = torch.cat((y_hist, decoder_output), 1)
        # decoder_input大小：batch_size,length_seq
        loss = loss + loss_func(decoder_output[:, -1], y_test[:, di, -1])

        y_predict = torch.cat((y_predict, decoder_output), 1)

    y_predict_all = torch.cat((y_predict_all, y_predict), 0)
    y_actual = torch.cat((y_actual, y_test.squeeze(2)), 0)

    test_loss += loss.data.numpy()
print("测试损失：%.4f" % (test_loss / len(test_x)))

#将2维张量维度转换为1维张量
rel_y_temp = y_actual
pre_y_temp = y_predict_all

rel_y=torch.tensor([])
pre_y=torch.tensor([])

for i in range(rel_y_temp.size()[0]):
    rel_y_one=rel_y_temp[i,:]
    rel_y=torch.cat((rel_y,rel_y_one))

for i in range(pre_y_temp.size()[0]):
    pre_y_one = pre_y_temp[i, :]
    pre_y = torch.cat((pre_y, pre_y_one))

rel_y=np.expand_dims(rel_y.data.numpy(),1)
pre_y=np.expand_dims(pre_y.data.numpy(),1)

rel_y = Y_min_max.inverse_transform(rel_y)
pre_y = Y_min_max.inverse_transform(pre_y)

actual_BTP=rel_y.squeeze(1)
predict_BTP=pre_y.squeeze(1)

BTP=pd.DataFrame(columns=["predicted value","actual value","R_score","HR","RMSE","MAE","MAPE","error"])
BTP["predicted value"] = predict_BTP
BTP["actual value"] = actual_BTP

R_score=r2_score(actual_BTP, predict_BTP)
HR=HR(actual_BTP,predict_BTP,ratio=0.03)
RMSE = sqrt(mean_squared_error(actual_BTP, predict_BTP))
MAE = mean_absolute_error(actual_BTP, predict_BTP)
MAPE=mape(actual_BTP, predict_BTP)
Error = actual_BTP - predict_BTP

BTP["R_score"] = R_score
BTP["HR"]=HR
BTP["RMSE"] = RMSE
BTP["MAE"] = MAE
BTP["MAPE"] = MAPE
BTP["Error"] = Error

# BTP.to_csv(r"F:\博一\柳钢项目\烧结终点研究1\论文结果\结果分析\模型消融"
#            r"\Ours-with-MT-dataprocess模型"+"步长为"+str(output_length)+"真实值与预测值对比.csv")

print("Ours模型的R_score: %f" % R_score)
print("Ours模型的HR: %f" % HR)
print("Ours模型的RMSE: %f" % RMSE)
print("Ours模型的MAE: %f" % MAE)
print("Ours模型的MAPE: %f" % MAPE)
print("Ours模型的偏差值:", Error)








