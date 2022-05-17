import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d
torch.set_default_dtype(torch.float32)
from numba import jit

"""
    x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class TATT_(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_, self).__init__()

        # self.d_conv_c = Conv2d(c_in, 1, kernel_size=(3, 3), padding=(1, 1), bias=True, dilation=(1, 1))

        self.dim_reduction = Conv2d(c_in, 1, kernel_size=(1, 1))


        self.dilation_conv_f = Conv2d(c_in,      1, kernel_size=(1, 1))
        self.dilation_conv_g = Conv2d(num_nodes, 1, kernel_size=(1, 1))

        self.U = nn.Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)


        self.bn = BatchNorm1d(tem_size)
        torch.nn.init.xavier_uniform_(self.U)
        torch.nn.init.xavier_uniform_(self.v)
        torch.nn.init.xavier_uniform_(self.b)

        A = np.zeros((24, 24))
        for i in range(12):
            for j in range(12):
                A[i, j] = 1
                A[i + 12, j + 12] = 1

        self.B = (torch.tensor((-1e13) * (1 - A))).type(torch.float64).cuda()

    def forward(self, seq):
        # seq.shape = torch.Size([18, 64, 170, 24])             # b,c,n,l

        c1 = seq.permute(0, 1, 3, 2)
        f1 = self.dilation_conv_f(c1).squeeze()                 # b,l,n

        print(f1.shape)

        exit(6)





        # f_star_x = self.dim_reduction(seq)                    # b,c,n,l
        f_star_x = torch.einsum("bcnl->bnl", seq)               # b,n,l
        f_star_x = self.dilation_conv(f_star_x)                 # b,n,l

        f_star_x_tr = f_star_x.permute(0, 2, 1)                 # b,l,n

        E_s = F.leaky_relu(
            torch.matmul(
                torch.matmul(f_star_x_tr, self.U),
                f_star_x
            ) + self.b
        )

        E_p = torch.matmul(self.v, E_s)                         # shape =  b, l, l

        # E_p = E_p.permute(0, 2, 1).contiguous()
        # E_p = self.bn(E_p).permute(0, 2, 1).contiguous()

        E = torch.zeros(E_p.shape).cuda()


        for bs in range(E_p.shape[0]):
            E[bs] = torch.exp(E_p[bs]) / torch.sum(torch.exp(E_p), dim=0)

        E_p = E.permute(0, 2, 1).contiguous()
        E_p = self.bn(E_p).permute(0, 2, 1).contiguous()
        E_p = torch.softmax(E_p+self.B, -1)

        return E_p


class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.conv_d1 = Conv1d(tem_size, tem_size, kernel_size=(2, ), dilation=(2, ), padding=(1, ), bias=False)
        self.conv_d2 = Conv1d(c_in, c_in,         kernel_size=(2, ), dilation=(2, ), padding=(1, ), bias=False)

        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)

        A = np.zeros((24, 24))
        for i in range(12):
            for j in range(12):
                A[i, j] = 1
                A[i + 12, j + 12] = 1

        self.B = (torch.tensor((-1e13) * (1 - A))).type(torch.float32).cuda()

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n
        f1 = self.conv_d1(f1)

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()  # b,c,l
        f2 = self.conv_d2(f2)

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)

        # normalization
        # logits=tf_util.batch_norm_for_conv1d(logits, is_training=training,
        #                                   bn_decay=bn_decay, scope='bn')
        # a,_ = torch.max(logits, 1, True)
        # logits = logits - a

        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits + self.B, -1)
        return coefs


class TATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size, shape_size_0):
        super(TATT, self).__init__()

        # self.d_conv_c = Conv2d(c_in, 1, kernel_size=(3, 3), padding=(1, 1), bias=True, dilation=(1, 1))

        self.dim_reduction_bnl = Conv2d(c_in, 1, kernel_size=(1, 1), bias=False)
        self.dim_reduction_bcl = Conv2d(num_nodes, 1, kernel_size=(1, 1), bias=False)


        self.dilation_conv_f = Conv1d(num_nodes, num_nodes, 3, padding=2, dilation=2)
        self.dilation_conv_g = Conv1d(c_in, c_in,           3, padding=2, dilation=2)

        self.U = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)


        self.bn = BatchNorm1d(tem_size)
        torch.nn.init.xavier_uniform_(self.U)
        torch.nn.init.xavier_uniform_(self.v)
        # torch.nn.init.xavier_uniform_(self.b)

        A = np.zeros((24, 24))
        for i in range(12):
            for j in range(12):
                A[i, j] = 1
                A[i + 12, j + 12] = 1

        self.B = (torch.tensor((-1e13) * (1 - A))).type(torch.float64).cuda()

    def forward(self, seq):
        # seq.shape = torch.Size([18, 64, 170, 24])                 # b,c,n,l


        f_star_x1    = self.dim_reduction_bnl(seq).squeeze()                             # b, n, c, l -> b, n, l
        # print(f_star_x1.shape)           # b,n,l

        f_star_x1_tr = self.dim_reduction_bcl(seq.permute(0, 2, 1, 3)).squeeze()         # b, n, c, l -> b, c, l
        # print(f_star_x1_tr.shape)        # b,c,l


        # --------------- D_Conv -----------------

        f_star_x1    = self.dilation_conv_f(f_star_x1)
        f_star_x1_tr = self.dilation_conv_g(f_star_x1_tr)

        # print(f_star_x1.shape, f_star_x1_tr.shape)          # torch.Size([18, 170, 24]) torch.Size([18, 64, 24])

        E_s = torch.sigmoid(
            torch.matmul(
                torch.matmul(f_star_x1.permute(0, 2, 1), self.U),              # b, n, l   *
                f_star_x1_tr
            ) + self.b
        )


        E_p = torch.matmul(self.v, E_s)


        # E_m = torch.zeros(E_p.shape).cuda()
        # for bs in range(E_p.shape[0]):
        #     E_m[bs] = torch.exp(E_p[bs]) / torch.sum(torch.exp(E_p[bs]), dim=0)


        # E_p = E_p.permute(0, 2, 1).contiguous()
        # E_p = self.bn(E_p).permute(0, 2, 1).contiguous()
        # E_p = torch.softmax(E_p+self.B, -1)

        return E_p


class T_cheby_conv(nn.Module):


    def __init__(self, c_in, c_out, K, Kt):
        super(T_cheby_conv, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        Lap = Lap.transpose(-1, -2)
        # print(Lap)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out


class Spatial_Temporal_Corr(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt, batch_size):
        super(Spatial_Temporal_Corr, self).__init__()
        self.c_out = c_out

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv2 = Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.dim_reduction = Conv2d(c_out, 1, kernel_size=(1, 1))
        self.dim_ascending = Conv2d(1, c_out, kernel_size=(1, 1))

        self.cheby_gcn_A = T_cheby_conv(c_out, 2 * c_out, K, Kt)
        # self.temporal_attention = TATT(c_out, num_nodes, tem_size, batch_size)
        self.temporal_attention = TATT_1(c_out, num_nodes, tem_size)
        self.time_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1), stride=(1, 1), bias=True)

        self.bn = LayerNorm([c_out, num_nodes, tem_size])
        self.batch_norm = BatchNorm2d(c_in, affine=False)


    def forward(self, x, supports):
        # x = self.batch_norm(x)
        x_input = self.conv1(x)                                 # b,c,n,l   c=64
        x_1     = self.time_conv(x)
        x_1     = F.leaky_relu(x_1)
        x_1 = F.dropout(x_1, 0.5, self.training)


        # ------------- SPA ----------------
        x_1_G = self.cheby_gcn_A(x_1, supports)                 # b,c,n,l
        filter_, gate_ = torch.split(x_1_G, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate_) * F.leaky_relu(filter_)
        x_1 = F.dropout(x_1, 0.5, self.training)
        # ----------------------------------



        # ------------- TEM ----------------
        T_coef = self.temporal_attention(x_1)                   # b,n,l
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum('bcnl,blq->bcnq', x_1, T_coef)
        # ----------------------------------



        out = self.bn(F.leaky_relu(x_1) + x_input)              # b,c,n,l

        return out, supports, T_coef


class ST_BLOCK_1(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_1, self).__init__()

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.TATT_1 = TATT_1(c_out, num_nodes, tem_size)
        self.dynamic_gcn = T_cheby_conv(c_out, 2 * c_out, K, Kt)
        self.K = K
        self.time_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1), stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.c_out = c_out
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        # print("x_input.shape:", x_input.shape)
        x_1 = self.time_conv(x)
        x_1 = self.bn(x_1)
        x_1 = F.leaky_relu(x_1)
        x_1 = F.dropout(x_1, 0.5, self.training)
        x_1 = self.dynamic_gcn(x_1, supports)
        filter, gate = torch.split(x_1, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        x_1 = F.dropout(x_1, 0.5, self.training)
        # print("---- !!! ---")
        # print(x_1.shape)
        T_coef = self.TATT_1(x_1)
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum('bcnl,blq->bcnq', x_1, T_coef)
        out = self.bn(F.leaky_relu(x_1) + x_input)
        return out, supports, T_coef


@jit(nopython=True)
def hadamard(x_1_, T_coef):
    for i in range(x_1_.shape[0]):
        for j in range(x_1_.shape[1]):
            x_1_[i, j] = x_1_[i, j] * T_coef[i]
    return x_1_

#
# class T_cheby_conv_ds(nn.Module):
#     '''
#     x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
#     nSample : number of samples = batch_size
#     nNode : number of node in graph
#     tem_size: length of temporal feature
#     c_in : number of input feature
#     c_out : number of output feature
#     adj : laplacian
#     K : size of kernel(number of cheby coefficients)
#     W : cheby_conv weight [K * feat_in, feat_out]
#     '''
#
#     def __init__(self, c_in, c_out, K, Kt):
#         super(T_cheby_conv_ds, self).__init__()
#         c_in_new = (K) * c_in
#         self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, Kt), padding=(0, 1),
#                             stride=(1, 1), bias=True)
#         self.K = K
#
#     def forward(self, x, adj):
#         nSample, feat_in, nNode, length = x.shape
#
#         Ls = []
#         L1 = adj
#         L0 = torch.eye(nNode).repeat(nSample, 1, 1).cuda()
#         Ls.append(L0)
#         Ls.append(L1)
#         for k in range(2, self.K):
#             L2 = 2 * torch.matmul(adj, L1) - L0
#             L0, L1 = L1, L2
#             Ls.append(L2)
#
#         Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
#         # print(Lap)
#         Lap = Lap.transpose(-1, -2)
#         x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
#         x = x.view(nSample, -1, nNode, length)
#         out = self.conv1(x)
#         return out


# A = np.zeros((24, 24))
# for i in range(24):
#     for j in range(24):
#         A[i, j] = 1
#
# B = (-1e13) * (1 - A)
# B = (torch.tensor(B)).type(torch.float32).cuda()
#
#
# class LSTM(nn.Module):
#     def __init__(self, c_in, c_out, num_nodes, tem_size):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(c_in, c_out, num_nodes, tem_size, batch_first=True)  # b*n,l,c
#         self.c_out = c_out
#         self.tem_size = tem_size
#         self.bn = BatchNorm2d(c_in, affine=False)
#
#         self.conv1 = Conv2d(c_out, 12, kernel_size=(1, tem_size), padding=(0, 0), stride=(1, 1), bias=True)
#
#     def forward(self, x_r):
#         x_r = self.bn(x_r)
#         x = x_r
#         shape = x.shape
#         h = Variable(torch.zeros((1, shape[0] * shape[2], self.c_out))).cuda()
#         c = Variable(torch.zeros((1, shape[0] * shape[2], self.c_out))).cuda()
#         hidden = (h, c)
#
#         x = x.permute(0, 2, 3, 1).contiguous().view(shape[0] * shape[2], shape[3], shape[1])
#         x, hidden = self.lstm(x, hidden)
#         x = x.view(shape[0], shape[2], shape[3], self.c_out).permute(0, 3, 1, 2).contiguous()
#
#         x = self.conv1(x).squeeze().permute(0, 2, 1).contiguous()  # b,n,l
#
#         return x


#
# import torch
# import torch.nn as nn
# from torch.nn.utils import weight_norm
#
#
# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size
#
#     def forward(self, x):
#         return x[:, :, :-self.chomp_size].contiguous()
#
#
# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))
#         self.chomp1 = Chomp1d(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)
#
#         self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))
#         self.chomp2 = Chomp1d(padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)
#
#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
#         self.relu = nn.ReLU()
#         self.init_weights()
#
#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)
#
#
# class TemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
#
#         self.network = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.network(x)
#
