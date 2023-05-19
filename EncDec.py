"fusion"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from MTCN import TCN
from VIFE import VIAM


class Encoder(nn.Module):
    def __init__(self,input_channels, channel_sizes,kernel_size, dropout,
                 d_feature,d_model, n_heads, d_k, d_v):
        super(Encoder, self).__init__()

        self.mtcn = TCN(input_channels, channel_sizes, kernel_size=kernel_size, dropout=dropout)
        self.vife=MultiHeadAttention(d_feature,d_model, n_heads, d_k, d_v)
        self.d_model=d_model

    def forward(self, input):
        latent_MTCN=self.mtcn(input.transpose(-2,-1)).transpose(-2,-1)
        latent_VIFE,attention=self.vife(input,input,input)

        # WA = nn.Parameter(torch.randn(input.size()[0],input.size()[1], self.d_model, requires_grad=True) * 0.01)
        # WB = nn.Parameter(torch.randn(input.size()[0],input.size()[1], self.d_model, requires_grad=True) * 0.01)
        # WA = nn.Parameter(torch.rand(1))
        # WB = 1-WA
        latent_variable=latent_MTCN+latent_VIFE
        #latent_variable  =latent_MTCN + latent_VIFE
        #latent_variable = latent_MTCN #去掉变量之间的关系VIFE
        return latent_variable

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, dropout_p, input_length):
        super(Decoder, self).__init__()

        # define parameters定义参数
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.time_step = input_length

        # define layers定义层数
        # 注意力网络层
        self.attn = nn.Linear(self.hidden_size *  self.n_layers + self.hidden_size, 1)

        # 注意力机制作用之后的结果映射到后面层
        self.attn_combine = nn.Linear(self.hidden_size  + self.output_size, self.hidden_size)

        # 定义GRU层
        self.gru = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.n_layers,
                          batch_first=True)

        # dropout操作层
        self.dropout = nn.Dropout(self.dropout_p)

        # 全连接层
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,input,hidden,encoder_outputs):
        input0=input[:,-1].unsqueeze(1)

        # 将hidden张量数据转化为batch_size排在第0维的形状
        # hidden的大小：direction*n_layers,batch_size,hidden_size
        temp_for_transpose = torch.transpose(hidden, 0, 1).contiguous()
        temp_for_transpose = temp_for_transpose.view(temp_for_transpose.size()[0], -1)
        hidden_attn = temp_for_transpose

        # 注意力权重的计算
        # hidden_attn大小：batch_size,direction*n_layers*hidden_size
        score = torch.tensor([])
        for i in range(self.time_step):
            # input_to_attention的大小：batch_size,hidden_size*(1+direction*n_layers)
            input_to_attention = torch.cat((hidden_attn, encoder_outputs[:, i, :]), 1)
            # 注意力层输出的权重
            score_one = self.attn(input_to_attention)
            score = torch.cat((score, score_one), 1)

        attn_weights = F.softmax(score)
        # attn_weights大小：batch_size,Input_legnth

        attn_weights = attn_weights.unsqueeze(1)
        # attn_weights大小：batch_size,1,length_seq中间的1是为了bmm乘法用的

        # encoder_outputs大小：batch_size,seq_length,hidden_size*direction
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        # attn_applied（Cj）大小：batch_size,1,hidden*direction
        # bmmm:两个矩阵相乘，忽略第一个batch维度，缩并时间维度

        # 将输入的词向量与注意力机制作用后的结果拼接成一个大的输入向量
        output = torch.cat((input0, attn_applied[:, 0, :]), 1)
        # output大小：batch_size,hidden*(direction+1)

        # 将大的输入向量映射为GRU的隐含层
        output = self.attn_combine(output).unsqueeze(1)
        # output大小：batch_size,length_seq,hidden_size
        output = F.relu(output)

        # outputz的结果再dropout
        output = self.dropout(output)

        # 开始解码器GRU的运算
        gru_output, hidden = self.gru(output, hidden)
        # output大小：batch_size,length_seq,hidden_size*directions
        # hidden大小：n_layers*directions,batch_size,hidden_size

        # 取出GRU运算最后一步的结果，输入给最后一层全连接层
        output = self.out(gru_output[:, -1, :])

        # output大小：batch_size*output_size
        return output, hidden, attn_weights

if __name__=="__main__":

    x = torch.randn(500, 40, 13)
    y=torch.randn(500,40,1)

    #定义encoder网络参数
    # MTCN的超参数
    input_channels = 13
    num_hidden = 20
    levels = 8
    channel_sizes = [num_hidden] * levels
    kernel_size = 7
    dropout = 0.0

    # VIFE超参数
    n_heads = 5
    d_feature = 13
    d_model = 20  # Embedding Size
    d_ff = 40  # FeedForward dimension
    d_k = d_v = 4  # dimension of K(=Q), V

    encoder = Encoder(input_channels, channel_sizes, kernel_size=kernel_size, dropout=dropout,
                    d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)

    # 定义decoder网络参数
    input_size = 13
    output_size = 1
    hidden_size = 20
    n_layers = 1
    dropout_p = 0.1
    input_length = 40

    decoder=Decoder(hidden_size, output_size, n_layers, dropout_p, input_length)

    encoder_outputs=encoder(x)
    decoder_hidden=encoder_outputs[:,-1,:].unsqueeze(0)
    decoder_input=x[:,:,-1]

    decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_outputs)

    print(encoder_outputs.shape)

